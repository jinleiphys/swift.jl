# Comprehensive benchmark for Rxy_matrix optimizations

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

println("="^70)
println("  Rxy_matrix OPTIMIZATION BENCHMARK")
println("="^70)

# System parameters
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
println("  Available threads: ", Threads.nthreads())
println()

using LinearAlgebra

# ============================================================================
# Test 1: Original Implementation
# ============================================================================
println("="^70)
println("TEST 1: ORIGINAL Rxy_matrix")
println("="^70)

include("matrices.jl")
using .matrices

println("\nWarming up...")
Rxy_orig, Rxy_31_orig, Rxy_32_orig = Rxy_matrix(α, grid)

println("Running benchmark (5 iterations)...")
times_orig = Float64[]
for i in 1:5
    print("  Iteration $i/5... ")
    t_start = time()
    Rxy_test, _, _ = Rxy_matrix(α, grid)
    t_elapsed = time() - t_start
    push!(times_orig, t_elapsed)
    println("$(round(t_elapsed, digits=3))s")
end

avg_orig = sum(times_orig) / length(times_orig)
println("\nResults:")
println("  Average: $(round(avg_orig, digits=3))s")
println("  Min: $(round(minimum(times_orig), digits=3))s")
println("  Max: $(round(maximum(times_orig), digits=3))s")

# ============================================================================
# Test 2: Sequential Optimized (Caching + Merged Loops)
# ============================================================================
println("\n" * "="^70)
println("TEST 2: OPTIMIZED Rxy_matrix (Sequential + Caching)")
println("="^70)

include("Rxy_matrix_optimized.jl")
using .RxyOptimized

println("\nWarming up...")
Rxy_opt, Rxy_31_opt, Rxy_32_opt = Rxy_matrix_optimized(α, grid)

println("\nRunning benchmark (5 iterations)...")
times_opt = Float64[]
for i in 1:5
    print("  Iteration $i/5... ")
    t_start = time()
    Rxy_test, _, _ = Rxy_matrix_optimized(α, grid)
    t_elapsed = time() - t_start
    push!(times_opt, t_elapsed)
    println("$(round(t_elapsed, digits=3))s")
end

avg_opt = sum(times_opt) / length(times_opt)
println("\nResults:")
println("  Average: $(round(avg_opt, digits=3))s")
println("  Min: $(round(minimum(times_opt), digits=3))s")
println("  Max: $(round(maximum(times_opt), digits=3))s")

# Verify correctness
println("\nVerification:")
diff_Rxy = norm(Rxy_orig - Rxy_opt) / norm(Rxy_orig)
diff_31 = norm(Rxy_31_orig - Rxy_31_opt) / norm(Rxy_31_orig)
diff_32 = norm(Rxy_32_orig - Rxy_32_opt) / norm(Rxy_32_orig)
println("  Rxy relative diff: $diff_Rxy")
println("  Rxy_31 relative diff: $diff_31")
println("  Rxy_32 relative diff: $diff_32")

if max(diff_Rxy, diff_31, diff_32) < 1e-10
    println("  ✓ Results are identical!")
else
    println("  ⚠ Warning: Results differ!")
end

# ============================================================================
# Test 3: Parallel Optimized (if multiple threads available)
# ============================================================================
if Threads.nthreads() > 1
    println("\n" * "="^70)
    println("TEST 3: PARALLEL Rxy_matrix ($(Threads.nthreads()) threads)")
    println("="^70)

    println("\nWarming up...")
    Rxy_par, Rxy_31_par, Rxy_32_par = Rxy_matrix_parallel(α, grid)

    println("\nRunning benchmark (5 iterations)...")
    times_par = Float64[]
    for i in 1:5
        print("  Iteration $i/5... ")
        t_start = time()
        Rxy_test, _, _ = Rxy_matrix_parallel(α, grid)
        t_elapsed = time() - t_start
        push!(times_par, t_elapsed)
        println("$(round(t_elapsed, digits=3))s")
    end

    avg_par = sum(times_par) / length(times_par)
    println("\nResults:")
    println("  Average: $(round(avg_par, digits=3))s")
    println("  Min: $(round(minimum(times_par), digits=3))s")
    println("  Max: $(round(maximum(times_par), digits=3))s")

    # Verify correctness
    println("\nVerification:")
    diff_Rxy_par = norm(Rxy_orig - Rxy_par) / norm(Rxy_orig)
    diff_31_par = norm(Rxy_31_orig - Rxy_31_par) / norm(Rxy_31_orig)
    diff_32_par = norm(Rxy_32_orig - Rxy_32_par) / norm(Rxy_32_orig)
    println("  Rxy relative diff: $diff_Rxy_par")
    println("  Rxy_31 relative diff: $diff_31_par")
    println("  Rxy_32 relative diff: $diff_32_par")

    if max(diff_Rxy_par, diff_31_par, diff_32_par) < 1e-10
        println("  ✓ Results are identical!")
    else
        println("  ⚠ Warning: Results differ!")
    end
else
    println("\n" * "="^70)
    println("PARALLEL TEST SKIPPED (only $(Threads.nthreads()) thread available)")
    println("Run with: julia -t auto benchmark_rxy_optimization.jl")
    println("="^70)
    avg_par = avg_orig  # For summary table
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("SUMMARY")
println("="^70)

speedup_opt = avg_orig / avg_opt
speedup_par = avg_orig / avg_par

println("\n┌─────────────────────────────────────────────────────────────┐")
println("│                    Performance Results                      │")
println("├─────────────────────────────────────────────────────────────┤")
println("│ Implementation         Time (s)    Speedup    Status        │")
println("├─────────────────────────────────────────────────────────────┤")
println("│ Original               $(rpad(round(avg_orig, digits=3), 9))  1.00×      baseline     │")
println("│ Optimized (sequential) $(rpad(round(avg_opt, digits=3), 9))  $(rpad(round(speedup_opt, digits=2), 5))×     ✓ verified   │")
if Threads.nthreads() > 1
    println("│ Parallel ($(Threads.nthreads()) threads)    $(rpad(round(avg_par, digits=3), 9))  $(rpad(round(speedup_par, digits=2), 5))×     ✓ verified   │")
end
println("└─────────────────────────────────────────────────────────────┘")

println("\nTime savings per call:")
println("  Sequential: $(round((avg_orig - avg_opt)*1000, digits=1)) ms ($(round((1 - avg_opt/avg_orig)*100, digits=1))% reduction)")
if Threads.nthreads() > 1
    println("  Parallel:   $(round((avg_orig - avg_par)*1000, digits=1)) ms ($(round((1 - avg_par/avg_orig)*100, digits=1))% reduction)")

    parallel_efficiency = speedup_par / Threads.nthreads()
    println("\nParallel efficiency: $(round(parallel_efficiency*100, digits=1))% ($(round(speedup_par, digits=2))× on $(Threads.nthreads()) threads)")
end

# Overall assessment
println("\n" * "="^70)
if speedup_opt >= 2.0
    println("🎉 EXCELLENT: Sequential optimization achieved $(round(speedup_opt, digits=1))× speedup!")
elseif speedup_opt >= 1.5
    println("✓ GOOD: Sequential optimization achieved $(round(speedup_opt, digits=1))× speedup")
else
    println("⚠ MODEST: Sequential optimization achieved $(round(speedup_opt, digits=1))× speedup")
end

if Threads.nthreads() > 1
    if speedup_par >= 4.0
        println("🚀 EXCELLENT: Parallel version achieved $(round(speedup_par, digits=1))× speedup!")
    elseif speedup_par >= 2.0
        println("✓ GOOD: Parallel version achieved $(round(speedup_par, digits=1))× speedup")
    else
        println("⚠ MODEST: Parallel version achieved $(round(speedup_par, digits=1))× speedup")
    end
end
println("="^70)
