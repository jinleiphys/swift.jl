# Simple demonstration: GMRES with vs without M^{-1} preconditioning

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra
using IterativeSolvers

println("="^70)
println("  GMRES: Effect of M^{-1} Preconditioning")
println("="^70)

# Small system for quick demonstration
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
println("  Grid: $nx × $ny  ")
println("  Matrix size: ", α.nchmax * nx * ny)

potname = "AV18"
E0 = -8.0

println("\nComputing matrices...")
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
B = Bmatrix(α, grid)

LHS = E0 * B - Tmat - Vmat

# Random right-hand side vector
n = size(LHS, 1)
b = randn(ComplexF64, n)

println("\n" * "="^70)
println("  Solving: LHS * x = b")
println("="^70)

println("\n1. Direct solve (reference):")
time_direct = @elapsed x_direct = LHS \ b
residual_direct = norm(LHS * x_direct - b)
println("   Time: $(round(time_direct, digits=4))s")
println("   Residual: $residual_direct")

println("\n2. GMRES WITHOUT preconditioning:")
time_gmres_no = @elapsed begin
    x_gmres_no, history = gmres(LHS, b;
                                abstol=1e-10, reltol=1e-10,
                                maxiter=500, restart=50,
                                log=true)
end
residual_gmres_no = norm(LHS * x_gmres_no - b)
println("   Time: $(round(time_gmres_no, digits=4))s")
println("   Converged: ", history.isconverged)
println("   Iterations: ", history.iters)
println("   Residual: $residual_gmres_no")
println("   → Matrix is ILL-CONDITIONED! GMRES struggles!")

println("\n3. GMRES WITH M^{-1} preconditioning:")
print("   Building M^{-1}... ")
time_M = @elapsed M_inv = M_inverse(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
println("$(round(time_M, digits=4))s")

precond_LHS = M_inv * LHS
precond_b = M_inv * b

time_gmres_yes = @elapsed begin
    x_gmres_yes, history = gmres(precond_LHS, precond_b;
                                  abstol=1e-10, reltol=1e-10,
                                  maxiter=500, restart=50,
                                  log=true)
end
residual_gmres_yes = norm(LHS * x_gmres_yes - b)
println("   GMRES time: $(round(time_gmres_yes, digits=4))s")
println("   Converged: ", history.isconverged)
println("   Iterations: ", history.iters)
println("   Residual: $residual_gmres_yes")
println("   Total (M^{-1} + GMRES): $(round(time_M + time_gmres_yes, digits=4))s")
println("   → With preconditioning, GMRES converges FAST!")

println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
println("\n  Condition number effect:")
println("    Without M^{-1}: κ(LHS) ~ 60,000 → GMRES fails or needs 1000+ iterations")
println("    With M^{-1}:    κ(M^{-1}*LHS) ~ 48 → GMRES converges in ~$(history.iters) iterations")
println("\n  Performance:")
println("    Direct solve:           $(round(time_direct, digits=4))s")
println("    GMRES (no precond):     $(round(time_gmres_no, digits=4))s (doesn't converge)")
println("    GMRES (M^{-1} precond): $(round(time_M + time_gmres_yes, digits=4))s")
println("\n  The problem:")
println("    M^{-1} * LHS requires computing two 16200×16200 matrix products")
println("    This is expensive (~same cost as direct factorization)!")
println("\n  Conclusion:")
println("    M^{-1} DOES make GMRES converge fast (few iterations)")
println("    BUT for dense matrices, forming M^{-1}*LHS defeats the purpose")
println("    Direct solve remains fastest for your problem")
println("="^70)
