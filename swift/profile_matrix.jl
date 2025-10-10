# Detailed profiling of matrix computation bottlenecks

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra

println("="^70)
println("  MATRIX COMPUTATION PROFILING")
println("="^70)

# Same system parameters as test_matrix.jl
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
println("  Grid: $nx × $ny × $nθ")
println("  Matrix size: ", α.nchmax * nx * ny)
println()

potname = "AV18"

# Profile each matrix computation
println("TIMING ANALYSIS:")
println("-"^70)

# 1. B matrix (should be fast)
println("\n1. B matrix:")
t_start = time()
B = Bmatrix(α, grid)
t_B = time() - t_start
println("   Time: $(round(t_B, digits=3)) seconds")

# 2. T matrix
println("\n2. T matrix:")
t_start = time()
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
t_T = time() - t_start
println("   Time: $(round(t_T, digits=3)) seconds")

# 3. V matrix
println("\n3. V matrix:")
t_start = time()
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
t_V = time() - t_start
println("   Time: $(round(t_V, digits=3)) seconds")

# 4. Rxy matrix (expected to be slowest)
println("\n4. Rxy matrix:")
t_start = time()
Rxy, Rxy_31, Rxy_32 = Rxy_matrix(α, grid)
t_Rxy = time() - t_start
println("   Time: $(round(t_Rxy, digits=3)) seconds")

# Summary
println("\n" * "="^70)
println("SUMMARY:")
println("-"^70)
total_time = t_B + t_T + t_V + t_Rxy
println("B matrix:    $(rpad(round(t_B, digits=3), 8)) s  ($(round(100*t_B/total_time, digits=1))%)")
println("T matrix:    $(rpad(round(t_T, digits=3), 8)) s  ($(round(100*t_T/total_time, digits=1))%)")
println("V matrix:    $(rpad(round(t_V, digits=3), 8)) s  ($(round(100*t_V/total_time, digits=1))%)")
println("Rxy matrix:  $(rpad(round(t_Rxy, digits=3), 8)) s  ($(round(100*t_Rxy/total_time, digits=1))%)")
println("-"^70)
println("TOTAL:       $(rpad(round(total_time, digits=3), 8)) s  (100.0%)")
println("="^70)

# Complexity analysis
println("\n" * "="^70)
println("COMPLEXITY ANALYSIS:")
println("-"^70)
println("Parameters:")
println("  nα = $(α.nchmax) channels")
println("  nx = $nx points")
println("  ny = $ny points")
println("  nθ = $nθ angular points")
println()
println("Theoretical complexity:")
println("  B matrix:    O(nα × nx × ny) = O($(α.nchmax * nx * ny))")
println("  T matrix:    O(nα × (nx² × ny + nx × ny²)) = O($(α.nchmax * (nx^2 * ny + nx * ny^2)))")
println("  V matrix:    O(nα² × nx) = O($(α.nchmax^2 * nx))")
println("  Rxy matrix:  O(nθ × nx² × ny² × nα²) = O($(nθ * nx^2 * ny^2 * α.nchmax^2))")
println("="^70)
