using LinearAlgebra, SparseArrays

println("="^70)
println("Testing Sparse vs Dense Matrix Multiplication")
println("="^70)
println()

# Create test matrices matching actual sparsity
n = 16200
println("Matrix size: $n × $n")
println()

# Create V matrix (99.7% sparse like the real one)
println("Creating V matrix (99.7% sparse, 702000 nonzeros)...")
V_dense = zeros(n, n)
n_nonzero = 702000
indices = rand(1:n*n, n_nonzero)
for idx in indices
    i = (idx-1) ÷ n + 1
    j = (idx-1) % n + 1
    V_dense[i, j] = randn()
end
println("  Dense V memory: $(round(sizeof(V_dense) / 1e9, digits=2)) GB")

# Convert to sparse
println("Converting to sparse format...")
V_sparse = sparse(V_dense)
println("  Sparse V memory: $(round((length(V_sparse.nzval) * 8 + length(V_sparse.rowval) * 8 + length(V_sparse.colptr) * 8) / 1e9, digits=2)) GB")
println("  Memory reduction: $(round(100 * (1 - (length(V_sparse.nzval) * 16) / sizeof(V_dense)), digits=1))%")
println()

# Create Rxy matrix (85% dense like the real one)
println("Creating Rxy matrix (85% dense)...")
Rxy = randn(ComplexF64, n, n)
for i in 1:n, j in 1:n
    if rand() < 0.148  # 14.8% zeros
        Rxy[i,j] = 0
    end
end
println("  Rxy memory: $(round(sizeof(Rxy) / 1e9, digits=2)) GB")
println()

println("="^70)
println("BENCHMARK: Dense V * Rxy")
println("="^70)
dense_time = @elapsed VRxy_dense = V_dense * Rxy
println("Time: $(round(dense_time, digits=2))s")
println()

println("="^70)
println("BENCHMARK: Sparse V * Rxy")
println("="^70)
sparse_time = @elapsed VRxy_sparse = V_sparse * Rxy
println("Time: $(round(sparse_time, digits=2))s")
println()

println("="^70)
println("RESULTS")
println("="^70)
println("Dense time:   $(round(dense_time, digits=2))s")
println("Sparse time:  $(round(sparse_time, digits=2))s")
println("Speedup:      $(round(dense_time / sparse_time, digits=2))x")
println()

if dense_time / sparse_time > 2
    println("✓✓ HUGE SPEEDUP! Sparse is $(round(dense_time / sparse_time, digits=1))x faster!")
elseif dense_time / sparse_time > 1.2
    println("✓ Good speedup! Sparse is $(round(dense_time / sparse_time, digits=1))x faster")
else
    println("✗ Minimal speedup")
end

println()
println("Estimated RHS cache time:")
println("  Current (dense):  ~255 seconds")
println("  With sparse:      ~$(round(255 / (dense_time / sparse_time), digits=1)) seconds")
