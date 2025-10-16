using LinearAlgebra
# Simulate the actual problem size
n = 16200  # 18 channels × 30 × 30
println("Testing matrix multiplication for n=$n...")

A = rand(n, n)
B = rand(n, n)

println("Matrix size: $n × $n = $(n^2) elements")
println("Memory per matrix: $(round(n^2 * 8 / 1e9, digits=2)) GB")
println()

println("Timing V * Rxy multiplication...")
@time C = A * B

println()
println("If this took ~255s, then V*Rxy is the bottleneck")
