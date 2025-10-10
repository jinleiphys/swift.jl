# Benchmark script to compare original vs optimized matrix implementations

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

println("="^70)
println("  OPTIMIZATION BENCHMARK")
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
println("  Grid: $nx × $ny")
println("  Matrix size: ", α.nchmax * nx * ny)
println()

# Benchmark T_matrix
println("="^70)
println("BENCHMARKING T_matrix")
println("="^70)

include("matrices.jl")
using .matrices

println("\n1. Original T_matrix:")
println("   Warming up...")
@time T_orig, Tx_ch_orig, Ty_ch_orig, Nx_orig, Ny_orig = T_matrix(α, grid, return_components=true)

println("   Running benchmark (10 iterations)...")
times_orig = Float64[]
for i in 1:10
    t_start = time()
    T_orig, _, _, _, _ = T_matrix(α, grid, return_components=true)
    push!(times_orig, time() - t_start)
end
avg_time_orig = sum(times_orig) / length(times_orig)
println("   Average time: $(round(avg_time_orig*1000, digits=2)) ms")
println("   Min time: $(round(minimum(times_orig)*1000, digits=2)) ms")
println("   Max time: $(round(maximum(times_orig)*1000, digits=2)) ms")

include("matrices_optimized.jl")
using .matrices_optimized

println("\n2. Optimized T_matrix:")
println("   Warming up...")
@time T_opt, Tx_ch_opt, Ty_ch_opt, Nx_opt, Ny_opt = T_matrix_optimized(α, grid, return_components=true)

println("   Running benchmark (10 iterations)...")
times_opt = Float64[]
for i in 1:10
    t_start = time()
    T_opt, _, _, _, _ = T_matrix_optimized(α, grid, return_components=true)
    push!(times_opt, time() - t_start)
end
avg_time_opt = sum(times_opt) / length(times_opt)
println("   Average time: $(round(avg_time_opt*1000, digits=2)) ms")
println("   Min time: $(round(minimum(times_opt)*1000, digits=2)) ms")
println("   Max time: $(round(maximum(times_opt)*1000, digits=2)) ms")

# Verify results are identical
println("\n3. Verification:")
using LinearAlgebra
diff = norm(T_orig - T_opt) / norm(T_orig)
println("   Relative difference: $(diff)")
if diff < 1e-10
    println("   ✓ Results are identical!")
else
    println("   ⚠ Warning: Results differ!")
end

# Summary
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
speedup = avg_time_orig / avg_time_opt
println("Original:  $(rpad(round(avg_time_orig*1000, digits=2), 8)) ms")
println("Optimized: $(rpad(round(avg_time_opt*1000, digits=2), 8)) ms")
println("-"^70)
println("Speedup:   $(round(speedup, digits=2))×")
println("Time saved: $(round((avg_time_orig - avg_time_opt)*1000, digits=2)) ms per call")
println("="^70)

if speedup > 1.3
    println("\n🎉 Optimization successful! Achieved $(round(speedup, digits=2))× speedup")
elseif speedup > 1.0
    println("\n✓ Modest improvement: $(round(speedup, digits=2))× speedup")
else
    println("\n⚠ No improvement observed - optimization may need adjustment")
end
