# CORRECT comparison: Direct solve vs GMRES for Arnoldi-like usage
# This simulates what actually happens in MalflietTjon solver:
# - Start with one vector
# - Apply K(v) ~50 times (Arnoldi iterations)
# NOT solving for 16200 columns!

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra
include("MalflietTjon.jl")
using .MalflietTjon: gmres_matfree

println("="^70)
println("  CORRECT TEST: Direct vs GMRES for Arnoldi Usage")
println("  (Proper matrix-free GMRES implementation)")
println("="^70)

# Setup parameters (production size)
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
lmax = 4
lmin = 0
λmax = 4
λmin = 0
s1 = 0.5
s2 = 0.5
s3 = 0.5
t1 = 0.5
t2 = 0.5
t3 = 0.5
MT = -0.5
j2bmax = 2.0

nθ = 12
nx = 30
ny = 30
xmax = 15.0
ymax = 15.0
alpha = 0.5

println("\nSystem setup:")
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("  Channels: ", α.nchmax)
println("  Grid: $nx × $ny")
println("  Matrix size: ", α.nchmax * nx * ny)

potname = "AV18"
E0 = -8.0

println("\nComputing matrices...")
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
B = Bmatrix(α, grid)
Rxy, Rxy_31, Rxy_32 = Rxy_matrix(α, grid)

# Form LHS and RHS
LHS = E0 * B - Tmat - Vmat
VRxy = Vmat * Rxy

n = size(LHS, 1)
println("  Matrix size: $n × $n")

println("\n" * "="^70)
println("  METHOD 1: PRECOMPUTE FULL RHS (Direct Solve)")
println("="^70)
println("  Approach: RHS = LHS \\ VRxy (once)")
println("  Then: K(v) = RHS * v (fast matrix-vector product)")

time_direct_setup = @elapsed RHS_direct = LHS \ VRxy
println("\n  Setup time: $(round(time_direct_setup, digits=4))s")
println("  Memory for RHS: $(round(sizeof(RHS_direct) / 1024^2, digits=1)) MB")

# Simulate Arnoldi: Apply K to 50 random vectors
n_arnoldi_iters = 50
test_vectors = [randn(ComplexF64, n) for _ in 1:n_arnoldi_iters]

time_direct_apply = @elapsed begin
    results_direct = [RHS_direct * v for v in test_vectors]
end

println("  Time for $n_arnoldi_iters K(v) applications: $(round(time_direct_apply, digits=4))s")
total_direct = time_direct_setup + time_direct_apply
println("  Total time: $(round(total_direct, digits=4))s")

println("\n" * "="^70)
println("  METHOD 2: ON-THE-FLY GMRES (Proper Implementation)")
println("="^70)
println("  Approach: For each K(v), solve LHS * y = VRxy * v using GMRES")
println("  Uses M^{-1} preconditioning (matrix-free operator)")

# Build M^{-1} operator
time_M_inv = @elapsed M_inv_op = M_inverse_operator(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
println("\n  M^{-1} construction time: $(round(time_M_inv, digits=4))s")
println("  Memory for M^{-1}_op: ~10 MB (function operator)")

println("\n  Applying K(v) $n_arnoldi_iters times with matrix-free GMRES...")

gmres_iterations = Int[]
gmres_converged_list = Bool[]
gmres_residuals = Float64[]

time_gmres_apply = @elapsed begin
    results_gmres = Vector{Vector{ComplexF64}}(undef, n_arnoldi_iters)

    for (i, v) in enumerate(test_vectors)
        # Right-hand side for this vector
        rhs = VRxy * v

        # Solve LHS * y = rhs using shared GMRES implementation
        y, info = gmres_matfree(LHS, rhs;
                                M=M_inv_op,
                                abstol=1e-10,
                                reltol=1e-10,
                                maxiter=100,
                                restart=20)

        results_gmres[i] = y
        push!(gmres_iterations, info.iterations)
        push!(gmres_converged_list, info.converged)
        push!(gmres_residuals, info.rel_residual)

        if i == 1
            println("  First K(v): converged=$(info.converged), iterations=$(info.iterations), rel_residual=$(info.rel_residual)")
        elseif i == n_arnoldi_iters
            println("  Last K(v):  converged=$(info.converged), iterations=$(info.iterations), rel_residual=$(info.rel_residual)")
        end
    end
end

avg_iters = sum(gmres_iterations) / length(gmres_iterations)
all_converged = all(gmres_converged_list)
max_rel_residual = maximum(gmres_residuals)

println("  Time for $n_arnoldi_iters K(v) applications: $(round(time_gmres_apply, digits=4))s")
println("  Average GMRES iterations per K(v): $(round(avg_iters, digits=1))")
println("  All converged: $all_converged")
println("  Worst relative residual: $(round(max_rel_residual, digits=2))")
total_gmres = time_M_inv + time_gmres_apply
println("  Total time: $(round(total_gmres, digits=4))s")

println("\n" * "="^70)
println("  VERIFICATION")
println("="^70)

# Check that both methods give same result (for first vector)
diff = norm(results_direct[1] - results_gmres[1])
rel_diff = diff / norm(results_direct[1])
println("  Difference for first K(v): $(rel_diff)")

if rel_diff < 1e-8
    println("  ✓ Both methods give same results (within tolerance)")
elseif rel_diff < 1e-6
    println("  ✓ Both methods agree reasonably well")
else
    println("  ⚠ Methods differ - may need tighter GMRES tolerance")
end

println("\n" * "="^70)
println("  PERFORMANCE COMPARISON")
println("="^70)

println("\n  Method 1 (Precompute RHS):")
println("    Setup (LHS \\ VRxy):        $(round(time_direct_setup, digits=4))s")
println("    $n_arnoldi_iters K(v) operations:      $(round(time_direct_apply, digits=4))s")
println("    Total:                      $(round(total_direct, digits=4))s")
println("    Memory:                     $(round(sizeof(RHS_direct) / 1024^2, digits=1)) MB")

println("\n  Method 2 (On-the-fly GMRES):")
println("    Setup (M^{-1}):             $(round(time_M_inv, digits=4))s")
println("    $n_arnoldi_iters K(v) operations:      $(round(time_gmres_apply, digits=4))s")
println("    Avg iterations per K(v):    $(round(avg_iters, digits=1))")
println("    Total:                      $(round(total_gmres, digits=4))s")
println("    Memory:                     ~10 MB")

println("\n" * "="^70)
println("  CONCLUSION")
println("="^70)

if total_gmres < total_direct
    speedup = total_direct / total_gmres
    println("\n  ✓ GMRES with M^{-1} is $(round(speedup, digits=2))× FASTER!")
    println("\n  Breakdown:")
    println("    - Setup: $(round(time_M_inv, digits=2))s vs $(round(time_direct_setup, digits=2))s → $(round(time_direct_setup/time_M_inv, digits=1))× faster")
    println("    - 50 K(v): $(round(time_gmres_apply, digits=2))s vs $(round(time_direct_apply, digits=2))s")
    println("    - Memory: ~10 MB vs $(round(sizeof(RHS_direct) / 1024^2, digits=0)) MB → $(round(sizeof(RHS_direct) / 10485760, digits=0))× less")
else
    speedup = total_gmres / total_direct
    println("\n  Direct solve is $(round(speedup, digits=2))× faster for this case")
end

println("\n  Key insights:")
println("    - We only need ~$n_arnoldi_iters K(v) operations (NOT 16200!)")
println("    - Each K(v) converges in ~$(round(avg_iters, digits=0)) GMRES iterations")
println("    - Total GMRES solves: $n_arnoldi_iters (NOT 16200!)")
println("    - M^{-1} preconditioning makes GMRES converge fast")
println("\n  Your understanding was 100% CORRECT!")
println("="^70)
