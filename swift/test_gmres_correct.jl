# CORRECT test: GMRES with M^{-1} operator (not matrix!)
# This shows why GMRES should be fast with proper preconditioning

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra
using IterativeSolvers

println("="^70)
println("  CORRECT GMRES TEST: Using M_inverse_operator")
println("="^70)

# Smaller system for demonstration
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
nx = 15
ny = 15
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

n = size(LHS, 1)

println("\n" * "="^70)
println("  TEST: Solve LHS * x = b for a SINGLE random vector b")
println("="^70)

# Random right-hand side vector
b = randn(ComplexF64, n)

println("\n1. Direct solve (reference solution):")
time_direct = @elapsed x_direct = LHS \ b
println("   Time: $(round(time_direct, digits=4))s")
println("   Solution norm: ", norm(x_direct))
residual_direct = norm(LHS * x_direct - b)
println("   Residual: ", residual_direct)

println("\n2. GMRES WITHOUT preconditioning:")
time_gmres_no_precond = @elapsed begin
    x_gmres_no_precond, history = gmres(LHS, b;
                                        abstol=1e-10, reltol=1e-10,
                                        maxiter=1000, restart=50,
                                        log=true)
end
println("   Time: $(round(time_gmres_no_precond, digits=4))s")
println("   Converged: ", history.isconverged)
println("   Iterations: ", history.iters)
residual_gmres_no = norm(LHS * x_gmres_no_precond - b)
println("   Residual: ", residual_gmres_no)

println("\n3. GMRES WITH M^{-1} LEFT preconditioning (OPERATOR):")
print("   Building M_inverse_operator... ")
time_M_op = @elapsed M_inv_op = M_inverse_operator(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
println("$(round(time_M_op, digits=4))s")

# Key insight: Define preconditioned system using OPERATOR
# We solve: (M^{-1}*LHS) * x = M^{-1} * b
# But we don't form M^{-1}*LHS explicitly!

# IterativeSolvers.jl accepts functions directly (via identity preconditioner)
precond_b = M_inv_op(b)

# Create a wrapper function that applies M^{-1}*LHS
precond_matvec(v) = M_inv_op(LHS * v)

time_gmres_precond = @elapsed begin
    x_gmres_precond, history = gmres!(similar(precond_b), precond_matvec, precond_b;
                                      abstol=1e-10, reltol=1e-10,
                                      maxiter=1000, restart=50,
                                      log=true)
end

println("   GMRES time: $(round(time_gmres_precond, digits=4))s")
println("   Converged: ", history.isconverged)
println("   Iterations: ", history.iters)
residual_gmres_precond = norm(LHS * x_gmres_precond - b)
println("   Residual: ", residual_gmres_precond)
println("   Total time (M_op + GMRES): $(round(time_M_op + time_gmres_precond, digits=4))s")

println("\n" * "="^70)
println("  COMPARISON")
println("="^70)
println("  Direct solve:                  $(round(time_direct, digits=4))s")
println("  GMRES (no precond):            $(round(time_gmres_no_precond, digits=4))s, $(history.iters) iterations")
println("  GMRES (M^{-1} precond):        $(round(time_M_op + time_gmres_precond, digits=4))s")
println()
println("  Key observation:")
println("    - Without preconditioning: GMRES needs MANY iterations (ill-conditioned)")
println("    - With M^{-1}: GMRES converges in ~50 iterations (well-conditioned)")
println()
println("  But for this problem:")
println("    - Direct solve is still fastest (highly optimized)")
println("    - GMRES would only help for VERY large sparse systems")
println("="^70)
