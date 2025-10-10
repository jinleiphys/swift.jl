# Quick benchmark for refined Rxy optimization (v2)

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

println("="^70)
println("  Rxy_matrix V2 OPTIMIZATION BENCHMARK")
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
println()

using LinearAlgebra

# Original implementation
println("="^70)
println("ORIGINAL Rxy_matrix")
println("="^70)

include("matrices.jl")
using .matrices

println("\nWarming up...")
@time Rxy_orig, Rxy_31_orig, Rxy_32_orig = Rxy_matrix(α, grid)

println("\nBenchmark (10 iterations)...")
times_orig = Float64[]
for i in 1:10
    t = @elapsed Rxy_matrix(α, grid)
    push!(times_orig, t)
end
avg_orig = sum(times_orig) / length(times_orig)
println("Average: $(round(avg_orig*1000, digits=1)) ms")
println("Min: $(round(minimum(times_orig)*1000, digits=1)) ms")
println("Max: $(round(maximum(times_orig)*1000, digits=1)) ms")

# Optimized v2
println("\n" * "="^70)
println("OPTIMIZED V2 Rxy_matrix (Algorithmic improvements)")
println("="^70)

include("Rxy_matrix_optimized_v2.jl")
using .RxyOptimizedV2

println("\nWarming up...")
@time Rxy_opt, Rxy_31_opt, Rxy_32_opt = Rxy_matrix_optimized_v2(α, grid)

println("\nBenchmark (10 iterations)...")
times_opt = Float64[]
for i in 1:10
    t = @elapsed Rxy_matrix_optimized_v2(α, grid)
    push!(times_opt, t)
end
avg_opt = sum(times_opt) / length(times_opt)
println("Average: $(round(avg_opt*1000, digits=1)) ms")
println("Min: $(round(minimum(times_opt)*1000, digits=1)) ms")
println("Max: $(round(maximum(times_opt)*1000, digits=1)) ms")

# Verification
println("\n" * "="^70)
println("VERIFICATION")
println("="^70)
diff_Rxy = norm(Rxy_orig - Rxy_opt) / norm(Rxy_orig)
diff_31 = norm(Rxy_31_orig - Rxy_31_opt) / norm(Rxy_31_orig)
diff_32 = norm(Rxy_32_orig - Rxy_32_opt) / norm(Rxy_32_orig)
println("Rxy relative diff:    $diff_Rxy")
println("Rxy_31 relative diff: $diff_31")
println("Rxy_32 relative diff: $diff_32")

if max(diff_Rxy, diff_31, diff_32) < 1e-10
    println("\n✓ Results are IDENTICAL (within machine precision)")
else
    println("\n⚠ WARNING: Results differ by more than tolerance!")
end

# Summary
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
speedup = avg_orig / avg_opt
time_saved = (avg_orig - avg_opt) * 1000

println("\nOriginal:    $(round(avg_orig*1000, digits=1)) ms")
println("Optimized:   $(round(avg_opt*1000, digits=1)) ms")
println("─"^70)
println("Speedup:     $(round(speedup, digits=2))×")
println("Time saved:  $(round(time_saved, digits=1)) ms per call")
println("Reduction:   $(round((1 - avg_opt/avg_orig)*100, digits=1))%")
println("="^70)

if speedup >= 1.5
    println("\n🎉 SUCCESS: Achieved $(round(speedup, digits=1))× speedup!")
elseif speedup >= 1.2
    println("\n✓ GOOD: Achieved $(round(speedup, digits=1))× speedup")
elseif speedup >= 1.0
    println("\n⚠ MODEST: Only $(round(speedup, digits=2))× speedup")
else
    println("\n❌ SLOWER: Performance degraded by $(round(1/speedup, digits=2))×")
end
