# Compare M_inverse (full matrix) vs M_inverse_operator (function)

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra

println("="^70)
println("  M_INVERSE: MATRIX vs OPERATOR COMPARISON")
println("="^70)

# Medium-sized system for testing
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
nx = 20
ny = 20
xmax = 15.0
ymax = 15.0
alpha = 0.5

println("\nSystem configuration:")
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("  Channels: ", α.nchmax)
println("  Grid: $nx × $ny")
println("  Matrix size: ", α.nchmax * nx * ny)

potname = "AV18"
E0 = -8.0

println("\nComputing T and V matrices...")
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)

println("\n" * "="^70)
println("METHOD 1: M_inverse (returns full matrix)")
println("="^70)
print("  Building full M_inv matrix... ")
time_matrix = @elapsed M_inv_matrix = M_inverse(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
println("$(round(time_matrix, digits=4))s")
println("  Memory: ", sizeof(M_inv_matrix) / 1024^2, " MB")

println("\n" * "="^70)
println("METHOD 2: M_inverse_operator (returns function)")
println("="^70)
print("  Building M_inv operator... ")
time_operator = @elapsed M_inv_op = M_inverse_operator(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
println("$(round(time_operator, digits=4))s")

println("\n" * "="^70)
println("VERIFICATION: Check that both give same results")
println("="^70)

# Test on a random vector
n = α.nchmax * nx * ny
test_vec = randn(n)

# Apply both methods
result_matrix = M_inv_matrix * test_vec
result_operator = M_inv_op(test_vec)

# Compare
diff = norm(result_matrix - result_operator)
rel_diff = diff / norm(result_matrix)

println("  Difference norm: ", diff)
println("  Relative difference: ", rel_diff)

if rel_diff < 1e-12
    println("  ✓ Results match perfectly!")
else
    println("  ⚠ Results differ (may indicate implementation issue)")
end

println("\n" * "="^70)
println("PERFORMANCE COMPARISON")
println("="^70)
println("  Matrix construction time: $(round(time_matrix, digits=4))s")
println("  Operator construction time: $(round(time_operator, digits=4))s")
println("  Speedup: $(round(time_matrix/time_operator, digits=2))×")

# Test application speed
println("\nTesting application speed (applying M^{-1} to vector):")
time_matrix_apply = @elapsed for i in 1:10
    M_inv_matrix * test_vec
end
time_operator_apply = @elapsed for i in 1:10
    M_inv_op(test_vec)
end

println("  Matrix application (10×): $(round(time_matrix_apply, digits=4))s")
println("  Operator application (10×): $(round(time_operator_apply, digits=4))s")
println("  Application speedup: $(round(time_matrix_apply/time_operator_apply, digits=2))×")

println("\n" * "="^70)
println("RECOMMENDATION:")
println("="^70)
println("  Use M_inverse_operator for:")
println("    - GMRES preconditioning (memory efficient)")
println("    - Iterative methods (only need matrix-vector products)")
println()
println("  Use M_inverse (full matrix) for:")
println("    - Testing and validation")
println("    - When you need M_inv * (large matrix)")
println("="^70)
