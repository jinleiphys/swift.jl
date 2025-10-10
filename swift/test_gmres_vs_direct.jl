# Simple comparison: Direct solve vs GMRES with M^{-1} preconditioning
# Solves: LHS * x = VRxy where LHS = E*B - T - V, VRxy = V * Rxy

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra
using Kronecker
using IterativeSolvers

println("="^70)
println("  DIRECT SOLVE vs GMRES WITH M^{-1} PRECONDITIONING")
println("="^70)

# Setup parameters
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
nx = 10
ny = 10
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

# Compute matrices
println("\nComputing matrices...")
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
B = Bmatrix(α, grid)
Rxy, Rxy_31, Rxy_32 = Rxy_matrix(α, grid)

# Form LHS and RHS
LHS = E0 * B - Tmat - Vmat
VRxy = Vmat * Rxy

println("  LHS size: ", size(LHS))
println("  RHS size: ", size(VRxy))
println("  RHS is complex: ", eltype(VRxy) <: Complex)

# Compute M^{-1} for preconditioning
println("\nComputing M^{-1} preconditioner...")
M_inv = M_inverse(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)

println("\n" * "="^70)
println("  METHOD 1: DIRECT SOLVE (BACKSLASH)")
println("="^70)

time_direct = @elapsed begin
    x_direct = LHS \ VRxy
end

residual_direct = norm(LHS * x_direct - VRxy)
println("  Time: $(round(time_direct, digits=4)) seconds")
println("  Solution norm: ", norm(x_direct))
println("  Residual norm: ", residual_direct)

println("\n" * "="^70)
println("  METHOD 2: GMRES WITH M^{-1} LEFT PRECONDITIONING")
println("="^70)

# Apply left preconditioner: solve (M^{-1}*LHS)*x = M^{-1}*VRxy
precond_LHS = M_inv * LHS
precond_RHS = M_inv * VRxy

# GMRES works with vectors, so we solve column by column
n_cols = size(VRxy, 2)
x_gmres = zeros(ComplexF64, size(VRxy))

time_gmres = @elapsed begin
    for col in 1:n_cols
        rhs_vec = precond_RHS[:, col]
        x_col, history = gmres(precond_LHS, rhs_vec;
                               abstol=1e-10,
                               reltol=1e-10,
                               maxiter=1000,
                               restart=50,
                               log=true,
                               verbose=false)
        x_gmres[:, col] = x_col

        if col == 1
            println("  First column convergence:")
            println("    Converged: ", history.isconverged)
            println("    Iterations: ", history.iters)
        end
    end
end

residual_gmres = norm(LHS * x_gmres - VRxy)
error_vs_direct = norm(x_gmres - x_direct)

println("  Total time: $(round(time_gmres, digits=4)) seconds")
println("  Solution norm: ", norm(x_gmres))
println("  Residual norm: ", residual_gmres)
println("  Error vs direct: ", error_vs_direct)

println("\n" * "="^70)
println("  COMPARISON SUMMARY")
println("="^70)
println("\n  Direct solve:")
println("    Time: $(round(time_direct, digits=4))s")
println("    Residual: $(residual_direct)")
println("\n  GMRES with M^{-1} preconditioning:")
println("    Time: $(round(time_gmres, digits=4))s")
println("    Residual: $(residual_gmres)")
println("    Error vs direct: $(error_vs_direct)")

speedup = time_direct / time_gmres
println("\n  Speedup: $(round(speedup, digits=2))×")

if speedup > 1.5
    println("  ✓ GMRES with preconditioning is faster!")
elseif speedup > 0.8
    println("  ≈ Both methods have similar performance")
else
    println("  ⚠ Direct solve is faster (for this problem size)")
    println("    Note: GMRES advantage increases with larger systems")
end

println("\n" * "="^70)
println("\nTest complete!")
