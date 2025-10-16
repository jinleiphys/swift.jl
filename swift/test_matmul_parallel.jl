using LinearAlgebra

println("Testing with different thread counts...")
n = 16200

println("\nTest 1: Default BLAS threads ($(BLAS.get_num_threads()))")
A = rand(n, n)
B = rand(n, n)
@time C = A * B

println("\nTest 2: Maximum BLAS threads ($(Sys.CPU_THREADS))")
BLAS.set_num_threads(Sys.CPU_THREADS)
A = rand(n, n)
B = rand(n, n)
@time C = A * B

println("\nSpeedup: $(round(118.7 / ((@elapsed C = A * B)), digits=2))x")
