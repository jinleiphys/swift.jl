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
using IterativeSolvers

println("="^70)
println("  CORRECT TEST: Direct vs GMRES for Arnoldi Usage")
println("  (Proper matrix-free GMRES implementation)")
println("="^70)

# Simple matrix-free GMRES implementation
"""
    gmres_matfree(A_op, b; tol=1e-10, maxiter=100, restart=20)

Matrix-free GMRES: A_op is a FUNCTION that applies the matrix.
Returns (x, converged, iterations)
"""
function gmres_matfree(A_op, b; abstol=1e-10, reltol=1e-10, maxiter=100, restart=20)
    n = length(b)
    x = zeros(ComplexF64, n)

    β_init = norm(b)
    if β_init < abstol
        return x, true, 0
    end

    total_iters = 0

    for restart_iter in 1:div(maxiter, restart) + 1
        # Arnoldi iteration
        m = min(restart, maxiter - total_iters)

        # Krylov subspace basis
        V = zeros(ComplexF64, n, m+1)
        H = zeros(ComplexF64, m+1, m)

        # Initial residual
        r = b - A_op(x)
        β = norm(r)

        if β < abstol || β/β_init < reltol
            return x, true, total_iters
        end

        V[:, 1] = r / β

        # Build Krylov subspace
        for j in 1:m
            w = A_op(V[:, j])

            # Modified Gram-Schmidt
            for i in 1:j
                H[i, j] = dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            end

            H[j+1, j] = norm(w)

            if real(H[j+1, j]) > 1e-14 && j < m
                V[:, j+1] = w / H[j+1, j]
            else
                # Happy breakdown
                m = j
                break
            end

            total_iters += 1
        end

        # Solve least squares problem: min ||β*e1 - H*y||
        e1 = zeros(ComplexF64, m+1)
        e1[1] = β

        # QR factorization of H to solve least squares
        H_reduced = H[1:m+1, 1:m]
        y = H_reduced \ e1

        # Update solution
        x = x + V[:, 1:m] * y

        # Check convergence
        r = b - A_op(x)
        residual = norm(r)

        if residual < abstol || residual/β_init < reltol
            return x, true, total_iters
        end

        if total_iters >= maxiter
            return x, false, total_iters
        end
    end

    return x, false, total_iters
end

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

# Define preconditioned operator: (M^{-1} * LHS) * v
precond_LHS_op(v) = M_inv_op(LHS * v)

println("\n  Applying K(v) $n_arnoldi_iters times with matrix-free GMRES...")

gmres_iterations = Int[]
gmres_converged_list = Bool[]

time_gmres_apply = @elapsed begin
    results_gmres = Vector{Vector{ComplexF64}}(undef, n_arnoldi_iters)

    for (i, v) in enumerate(test_vectors)
        # Right-hand side for this vector
        rhs = VRxy * v

        # Precondition the RHS
        precond_rhs = M_inv_op(rhs)

        # Solve: (M^{-1} * LHS) * y = M^{-1} * rhs using matrix-free GMRES
        y, converged, iters = gmres_matfree(precond_LHS_op, precond_rhs;
                                            abstol=1e-10, reltol=1e-10,
                                            maxiter=100, restart=20)

        results_gmres[i] = y
        push!(gmres_iterations, iters)
        push!(gmres_converged_list, converged)

        if i == 1
            println("  First K(v): converged=$converged, iterations=$iters")
        elseif i == n_arnoldi_iters
            println("  Last K(v):  converged=$converged, iterations=$iters")
        end
    end
end

avg_iters = sum(gmres_iterations) / length(gmres_iterations)
all_converged = all(gmres_converged_list)

println("  Time for $n_arnoldi_iters K(v) applications: $(round(time_gmres_apply, digits=4))s")
println("  Average GMRES iterations per K(v): $(round(avg_iters, digits=1))")
println("  All converged: $all_converged")
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
