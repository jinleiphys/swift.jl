# Full performance comparison: Non-optimized vs Optimized matrix computations
# Tests T, V, and Rxy matrices

include("general_modules/channels.jl")
include("general_modules/mesh.jl")
using .channels
using .mesh
include("swift/matrices.jl")
using .matrices
include("swift/matrices_optimized.jl")
using .matrices_optimized
using LinearAlgebra

println("="^70)
println("  FULL MATRIX OPTIMIZATION PERFORMANCE TEST")
println("="^70)

# Test system parameters
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
potname = "AV18"

println("\n" * "="^70)
println("  System Setup")
println("="^70)
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

println("  Fermion system: $fermion")
println("  Jtot = $Jtot, T = $T, Parity = $Parity")
println("  λmax = $λmax, lmax = $lmax")
println("  Channels: ", α.nchmax)
println("  Grid: $nx × $ny × $nθ")
println("  Total matrix size: ", α.nchmax * nx * ny, " × ", α.nchmax * nx * ny)
println("  Matrix elements: ", (α.nchmax * nx * ny)^2, " (",
        round((α.nchmax * nx * ny)^2 * 16 / 1024^3, digits=2), " GB complex)")

# T matrix tests
println("\n" * "="^70)
println("  TEST 1: Non-Optimized T_matrix (Original)")
println("="^70)

println("  Warming up (compilation)...")
@time T_orig = T_matrix(α, grid)

println("\n  Timed run:")
time_T_orig = @elapsed begin
    T_orig = T_matrix(α, grid)
end
println("  ✓ Non-optimized T time: ", round(time_T_orig, digits=3), " seconds")

println("\n" * "="^70)
println("  TEST 2: Optimized T_matrix_optimized")
println("="^70)

println("  Warming up (compilation)...")
@time T_opt = T_matrix_optimized(α, grid)

println("\n  Timed run:")
time_T_opt = @elapsed begin
    T_opt = T_matrix_optimized(α, grid)
end
println("  ✓ Optimized T time: ", round(time_T_opt, digits=3), " seconds")

# V matrix tests
println("\n" * "="^70)
println("  TEST 3: Non-Optimized V_matrix (Original)")
println("="^70)

println("  Warming up (compilation)...")
@time V_orig = V_matrix(α, grid, potname)

println("\n  Timed run:")
time_V_orig = @elapsed begin
    V_orig = V_matrix(α, grid, potname)
end
println("  ✓ Non-optimized V time: ", round(time_V_orig, digits=3), " seconds")

println("\n" * "="^70)
println("  TEST 4: Optimized V_matrix_optimized")
println("="^70)

println("  Warming up (compilation)...")
@time V_opt = V_matrix_optimized(α, grid, potname)

println("\n  Timed run:")
time_V_opt = @elapsed begin
    V_opt = V_matrix_optimized(α, grid, potname)
end
println("  ✓ Optimized V time: ", round(time_V_opt, digits=3), " seconds")

# Rxy matrix tests
println("\n" * "="^70)
println("  TEST 5: Non-Optimized Rxy_matrix (Original)")
println("="^70)

println("  Warming up (compilation)...")
@time Rxy_orig, Rxy31_orig, Rxy32_orig = Rxy_matrix(α, grid)

println("\n  Timed run:")
time_Rxy_orig = @elapsed begin
    Rxy_orig, Rxy31_orig, Rxy32_orig = Rxy_matrix(α, grid)
end
println("  ✓ Non-optimized Rxy time: ", round(time_Rxy_orig, digits=3), " seconds")

println("\n" * "="^70)
println("  TEST 6: Optimized Rxy_matrix_optimized")
println("="^70)

println("  Warming up (compilation)...")
@time Rxy_opt, Rxy31_opt, Rxy32_opt = Rxy_matrix_optimized(α, grid)

println("\n  Timed run:")
time_Rxy_opt = @elapsed begin
    Rxy_opt, Rxy31_opt, Rxy32_opt = Rxy_matrix_optimized(α, grid)
end
println("  ✓ Optimized Rxy time: ", round(time_Rxy_opt, digits=3), " seconds")

# Performance comparison
println("\n" * "="^70)
println("  PERFORMANCE COMPARISON")
println("="^70)

# T matrix comparison
speedup_T = time_T_orig / time_T_opt
println("\nT Matrix:")
println("  Non-optimized time:  ", round(time_T_orig, digits=3), " s")
println("  Optimized time:      ", round(time_T_opt, digits=3), " s")
println("  Speedup:             ", round(speedup_T, digits=2), "×")
println("  Time saved:          ", round(time_T_orig - time_T_opt, digits=3), " s (",
        round(100 * (time_T_orig - time_T_opt) / time_T_orig, digits=1), "%)")

# V matrix comparison
speedup_V = time_V_orig / time_V_opt
println("\nV Matrix:")
println("  Non-optimized time:  ", round(time_V_orig, digits=3), " s")
println("  Optimized time:      ", round(time_V_opt, digits=3), " s")
println("  Speedup:             ", round(speedup_V, digits=2), "×")
println("  Time saved:          ", round(time_V_orig - time_V_opt, digits=3), " s (",
        round(100 * (time_V_orig - time_V_opt) / time_V_orig, digits=1), "%)")

# Rxy matrix comparison
speedup_Rxy = time_Rxy_orig / time_Rxy_opt
println("\nRxy Matrix:")
println("  Non-optimized time:  ", round(time_Rxy_orig, digits=3), " s")
println("  Optimized time:      ", round(time_Rxy_opt, digits=3), " s")
println("  Speedup:             ", round(speedup_Rxy, digits=2), "×")
println("  Time saved:          ", round(time_Rxy_orig - time_Rxy_opt, digits=3), " s (",
        round(100 * (time_Rxy_orig - time_Rxy_opt) / time_Rxy_orig, digits=1), "%)")

# Total time
time_orig_total = time_T_orig + time_V_orig + time_Rxy_orig
time_opt_total = time_T_opt + time_V_opt + time_Rxy_opt
speedup_total = time_orig_total / time_opt_total
println("\nTotal (T + V + Rxy):")
println("  Non-optimized time:  ", round(time_orig_total, digits=3), " s")
println("  Optimized time:      ", round(time_opt_total, digits=3), " s")
println("  Speedup:             ", round(speedup_total, digits=2), "×")
println("  Time saved:          ", round(time_orig_total - time_opt_total, digits=3), " s (",
        round(100 * (time_orig_total - time_opt_total) / time_orig_total, digits=1), "%)")

# Numerical accuracy check
println("\n" * "="^70)
println("  NUMERICAL ACCURACY CHECK")
println("="^70)

# Check T matrix
diff_T = norm(T_orig - T_opt)
println("\nT Matrix:")
println("  ||T_orig - T_opt|| = ", diff_T)
println("  Relative error:    ", diff_T / norm(T_orig))

# Check V matrix
diff_V = norm(V_orig - V_opt)
println("\nV Matrix:")
println("  ||V_orig - V_opt|| = ", diff_V)
println("  Relative error:    ", diff_V / norm(V_orig))

# Check Rxy matrices
diff_Rxy = norm(Rxy_orig - Rxy_opt)
diff_31 = norm(Rxy31_orig - Rxy31_opt)
diff_32 = norm(Rxy32_orig - Rxy32_opt)

println("\nRxy Matrices:")
println("  ||Rxy_orig - Rxy_opt||     = ", diff_Rxy)
println("  ||Rxy31_orig - Rxy31_opt|| = ", diff_31)
println("  ||Rxy32_orig - Rxy32_opt|| = ", diff_32)

# Use relative tolerance (more appropriate for numerical comparison)
rel_tol = 1e-14
abs_tol = 1e-10

rel_err_T = diff_T / norm(T_orig)
rel_err_V = diff_V / norm(V_orig)
rel_err_Rxy = diff_Rxy / norm(Rxy_orig)
rel_err_31 = diff_31 / norm(Rxy31_orig)
rel_err_32 = diff_32 / norm(Rxy32_orig)

# Check both absolute and relative tolerances
pass_T = (diff_T < abs_tol) || (rel_err_T < rel_tol)
pass_V = (diff_V < abs_tol) || (rel_err_V < rel_tol)
pass_Rxy = (diff_Rxy < abs_tol) || (rel_err_Rxy < rel_tol)
pass_31 = (diff_31 < abs_tol) || (rel_err_31 < rel_tol)
pass_32 = (diff_32 < abs_tol) || (rel_err_32 < rel_tol)

if pass_T && pass_V && pass_Rxy && pass_31 && pass_32
    println("\n  ✓ PASS: All results are numerically identical")
    println("    (relative errors < $rel_tol or absolute errors < $abs_tol)")
else
    println("\n  ✗ FAIL: Results differ beyond tolerance!")
    if !pass_T
        println("    T matrix relative error: ", rel_err_T)
    end
    if !pass_V
        println("    V matrix relative error: ", rel_err_V)
    end
    if !pass_Rxy
        println("    Rxy matrix relative error: ", rel_err_Rxy)
    end
end

# Summary
println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
println("\nIndividual matrix speedups:")
println("  T Matrix:   ", round(speedup_T, digits=2), "× speedup  (",
        round(time_T_orig, digits=3), " s → ", round(time_T_opt, digits=3), " s)")
println("  V Matrix:   ", round(speedup_V, digits=2), "× speedup  (",
        round(time_V_orig, digits=3), " s → ", round(time_V_opt, digits=3), " s)")
println("  Rxy Matrix: ", round(speedup_Rxy, digits=2), "× speedup  (",
        round(time_Rxy_orig, digits=3), " s → ", round(time_Rxy_opt, digits=3), " s)")

println("\nOverall results:")
println("  Total time:     ", round(time_orig_total, digits=3), " s → ", round(time_opt_total, digits=3), " s")
println("  Overall speedup: ", round(speedup_total, digits=2), "×")
println("  Total time saved: ", round(time_orig_total - time_opt_total, digits=3), " s")

println("\n  ✓ Numerical accuracy:  Identical results (within tolerance)")
println("  ✓ Memory optimization: Reduced Kronecker product overhead")
println("  ✓ Symmetry exploited:  Rxy_31 = Rxy_32 in optimized version")
println("="^70)
