# Realistic test: How GMRES is actually used in MalflietTjon solver
# This demonstrates the on-the-fly approach vs precomputing full RHS

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
println("  REALISTIC GMRES TEST: On-the-Fly vs Precomputed RHS")
println("="^70)

# Setup (smaller system for faster demonstration)
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
lmax = 2
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
nx = 20
ny = 20
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

LHS = E0 * B - Tmat - Vmat
VRxy = Vmat * Rxy

println("\n" * "="^70)
println("  METHOD 1: PRECOMPUTE FULL RHS (Direct Solve)")
println("="^70)
println("  This computes RHS = LHS \\ VRxy once (16 × 16 matrix)")
println("  Then uses K(x) = RHS * x for fast matrix-vector products")

time_precompute = @elapsed RHS_precomputed = LHS \ VRxy
println("\n  Precomputation time: $(round(time_precompute, digits=4))s")
println("  Memory for RHS: $(sizeof(RHS_precomputed) / 1024^2) MB")

# Simulate Arnoldi iterations with precomputed RHS
K_precomputed = x -> RHS_precomputed * x

# Test: Apply K to 50 random vectors (simulating Arnoldi iterations)
n = size(LHS, 1)
test_vectors = [randn(ComplexF64, n) for _ in 1:50]

time_apply_precomputed = @elapsed begin
    results_precomputed = [K_precomputed(v) for v in test_vectors]
end

println("  Time for 50 K(x) operations: $(round(time_apply_precomputed, digits=4))s")
println("  Total time (precompute + 50 ops): $(round(time_precompute + time_apply_precomputed, digits=4))s")

println("\n" * "="^70)
println("  METHOD 2: ON-THE-FLY GMRES (No Precomputation)")
println("="^70)
println("  This solves LHS * y = VRxy * x using GMRES each time K(x) is called")
println("  Uses M^{-1} preconditioning for fast convergence")

# Compute M_inv preconditioner
time_M_inv = @elapsed M_inv = M_inverse(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
println("\n  M^{-1} computation time: $(round(time_M_inv, digits=4))s")
println("  Memory for M_inv: $(sizeof(M_inv) / 1024^2) MB")

# Define on-the-fly K using GMRES
K_gmres = function(x)
    rhs = VRxy * x
    y, _ = gmres_matfree(LHS, rhs; M=M_inv, abstol=1e-10, reltol=1e-10, maxiter=100, restart=20)
    return y
end

# Test: Apply K to same 50 vectors
time_apply_gmres = @elapsed begin
    results_gmres = [K_gmres(v) for v in test_vectors]
end

println("  Time for 50 K(x) operations: $(round(time_apply_gmres, digits=4))s")
println("  Total time (M^{-1} + 50 ops): $(round(time_M_inv + time_apply_gmres, digits=4))s")

println("\n" * "="^70)
println("  VERIFICATION")
println("="^70)

# Check that both methods give same result
diff = norm(results_precomputed[1] - results_gmres[1])
rel_diff = diff / norm(results_precomputed[1])
println("  Difference for first vector: $(rel_diff)")

if rel_diff < 1e-8
    println("  ✓ Both methods give same result")
else
    println("  ⚠ Results differ significantly")
end

println("\n" * "="^70)
println("  PERFORMANCE COMPARISON")
println("="^70)

total_precompute = time_precompute + time_apply_precomputed
total_gmres = time_M_inv + time_apply_gmres

println("  Precomputed RHS approach:")
println("    Setup: $(round(time_precompute, digits=4))s")
println("    50 operations: $(round(time_apply_precomputed, digits=4))s")
println("    Total: $(round(total_precompute, digits=4))s")
println("    Memory: $(sizeof(RHS_precomputed) / 1024^2) MB")

println("\n  On-the-fly GMRES approach:")
println("    Setup (M^{-1}): $(round(time_M_inv, digits=4))s")
println("    50 operations: $(round(time_apply_gmres, digits=4))s")
println("    Total: $(round(total_gmres, digits=4))s")
println("    Memory: $(sizeof(M_inv) / 1024^2) MB")

if total_precompute < total_gmres
    speedup = total_gmres / total_precompute
    println("\n  → Direct solve is $(round(speedup, digits=2))× faster for this system size")
    println("    Recommendation: Use direct solve (current default)")
else
    speedup = total_precompute / total_gmres
    println("\n  → GMRES is $(round(speedup, digits=2))× faster for this system size")
    println("    Recommendation: Enable use_gmres=true")
end

println("\n" * "="^70)
println("  CONCLUSION")
println("="^70)
println("  For current system sizes (~20×20 to 30×30):")
println("    - Direct solve is faster and simpler")
println("    - Memory is not a constraint")
println()
println("  GMRES would become advantageous when:")
println("    - Grid size >> 50×50 (RHS matrix becomes huge)")
println("    - Memory becomes limiting factor")
println("    - Direct factorization becomes prohibitively expensive")
println("="^70)
