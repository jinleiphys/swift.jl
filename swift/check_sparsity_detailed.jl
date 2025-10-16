using LinearAlgebra
using SparseArrays
using Printf

# PERFORMANCE: Use all available CPU cores for BLAS operations
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS threads set to: $(BLAS.get_num_threads()) (CPU cores: $(Sys.CPU_THREADS))")
println()

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

include("matrices.jl")
using .matrices

include("twobody.jl")
using .twobodybound

# Same parameters as swift_3H.jl
fermion=true; Jtot = 0.5; T = 0.5; Parity=1
lmax=8; lmin=0; λmax=20; λmin=0; s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5; MT=-0.5
j2bmax=2.0
nθ=12; nx=30; ny=30; xmax=20; ymax=20; alpha=1

println("="^70)
println("    RHS MATRIX SPARSITY ANALYSIS")
println("="^70)
println()

α = α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)
grid = initialmesh(nθ,nx,ny,Float64(xmax),Float64(ymax),Float64(alpha))

potname = "AV18"
e2b, ψ = bound2b(grid, potname)

println("Computing matrices...")
println("-"^70)

# Compute all matrices with components
print("  Computing T matrix... ")
T_time = @elapsed begin
    T, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
end
println("$(round(T_time, digits=2))s")

print("  Computing V matrix... ")
V_time = @elapsed begin
    V, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
end
println("$(round(V_time, digits=2))s")

print("  Computing Rxy matrix... ")
Rxy_time = @elapsed begin
    Rxy, Rxy_31, Rxy_32 = Rxy_matrix(α, grid)
end
println("$(round(Rxy_time, digits=2))s")

println()
println("="^70)
println("    SPARSITY ANALYSIS")
println("="^70)

nα = α.nchmax
n_total = nα * nx * ny

@printf("Matrix dimensions: %d × %d (nα=%d, nx=%d, ny=%d)\n", n_total, n_total, nα, nx, ny)
@printf("Total matrix elements: %d\n", n_total^2)
println()

# Function to analyze sparsity
function analyze_sparsity(M::AbstractMatrix, name::String)
    n = size(M, 1)
    total_elements = n^2

    # Count nonzeros with different thresholds
    tol_1e8 = count(abs.(M) .> 1e-8)
    tol_1e10 = count(abs.(M) .> 1e-10)
    tol_1e12 = count(abs.(M) .> 1e-12)
    tol_eps = count(abs.(M) .> eps(Float64))

    sparsity_1e8 = 100.0 * (1 - tol_1e8 / total_elements)
    sparsity_1e10 = 100.0 * (1 - tol_1e10 / total_elements)
    sparsity_1e12 = 100.0 * (1 - tol_1e12 / total_elements)
    sparsity_eps = 100.0 * (1 - tol_eps / total_elements)

    println("$name:")
    @printf("  Nonzeros (|x| > 1e-8):  %10d / %10d (%5.2f%% sparse)\n", tol_1e8, total_elements, sparsity_1e8)
    @printf("  Nonzeros (|x| > 1e-10): %10d / %10d (%5.2f%% sparse)\n", tol_1e10, total_elements, sparsity_1e10)
    @printf("  Nonzeros (|x| > 1e-12): %10d / %10d (%5.2f%% sparse)\n", tol_1e12, total_elements, sparsity_1e12)
    @printf("  Nonzeros (|x| > eps):   %10d / %10d (%5.2f%% sparse)\n", tol_eps, total_elements, sparsity_eps)

    # Check for block structure
    println("  Block structure analysis:")
    block_size = nx * ny
    for iα in 1:min(nα, 3)  # Analyze first few blocks
        for jα in 1:min(nα, 3)
            idx_i = ((iα-1)*block_size + 1):(iα*block_size)
            idx_j = ((jα-1)*block_size + 1):(jα*block_size)
            block = M[idx_i, idx_j]
            nnz_block = count(abs.(block) .> 1e-10)
            sparsity_block = 100.0 * (1 - nnz_block / length(block))
            @printf("    Block[%d,%d]: %5.1f%% sparse (%d nonzeros)\n", iα, jα, sparsity_block, nnz_block)
        end
    end
    println()
end

# Analyze V matrix
println("1. POTENTIAL MATRIX V:")
analyze_sparsity(V, "V matrix")

# Analyze Rxy matrix
println("2. REARRANGEMENT MATRIX Rxy:")
analyze_sparsity(Rxy, "Rxy matrix")

# Compute V*Rxy
print("Computing V * Rxy... ")
VRxy_time = @elapsed begin
    VRxy = V * Rxy
end
println("$(round(VRxy_time, digits=2))s")
println()

println("3. PRODUCT MATRIX V*Rxy:")
analyze_sparsity(VRxy, "V*Rxy matrix")

# Build V_diag (diagonal block part)
print("Building V_diag (diagonal blocks)... ")
Vdiag_time = @elapsed begin
    V_diag = zeros(n_total, n_total)
    for iα in 1:nα
        idx_start = (iα-1) * nx * ny + 1
        idx_end = iα * nx * ny
        V_diag_block = kron(V_x_diag_ch[iα], Matrix{Float64}(I, ny, ny))
        V_diag[idx_start:idx_end, idx_start:idx_end] = V_diag_block
    end
end
println("$(round(Vdiag_time, digits=2))s")
println()

println("4. DIAGONAL BLOCK MATRIX V_diag:")
analyze_sparsity(V_diag, "V_diag matrix")

# Compute V_off_diag = V - V_diag
V_off_diag = V - V_diag

println("5. OFF-DIAGONAL MATRIX V - V_diag:")
analyze_sparsity(V_off_diag, "V - V_diag matrix")

# Build RHS = V_off_diag + VRxy
RHS_matrix = V_off_diag + VRxy

println("6. FULL RHS MATRIX (V - V_diag) + V*Rxy:")
analyze_sparsity(RHS_matrix, "RHS matrix")

println()
println("="^70)
println("    SPARSE MATRIX PERFORMANCE TEST")
println("="^70)
println()

# Test vector
x = randn(ComplexF64, n_total)

# Benchmark dense matrix-vector multiplication
println("Dense matrix-vector multiplication:")
dense_times = Float64[]
for i in 1:5
    t = @elapsed y = RHS_matrix * x
    push!(dense_times, t)
    if i == 1
        @printf("  Trial %d: %.4f s (compilation)\n", i, t)
    else
        @printf("  Trial %d: %.4f s\n", i, t)
    end
end
avg_dense_time = mean(dense_times[2:end])
@printf("  Average (excluding first): %.4f s\n", avg_dense_time)
println()

# Convert to sparse with different thresholds
thresholds = [1e-12, 1e-10, 1e-8, 1e-6]

println("Sparse matrix-vector multiplication:")
for tol in thresholds
    # Create sparse matrix
    print("  Creating sparse matrix (tol=$tol)... ")
    sparse_time = @elapsed begin
        RHS_sparse = sparse(RHS_matrix .* (abs.(RHS_matrix) .> tol))
    end
    nnz_sparse = nnz(RHS_sparse)
    sparsity = 100.0 * (1 - nnz_sparse / length(RHS_matrix))
    println("$(round(sparse_time, digits=2))s")
    @printf("    Nonzeros: %d (%.2f%% sparse)\n", nnz_sparse, sparsity)

    # Benchmark sparse multiplication
    sparse_times = Float64[]
    for i in 1:5
        t = @elapsed y = RHS_sparse * x
        push!(sparse_times, t)
    end
    avg_sparse_time = mean(sparse_times[2:end])
    speedup = avg_dense_time / avg_sparse_time
    @printf("    Average time: %.4f s (%.2fx %s)\n", avg_sparse_time, abs(speedup), speedup > 1 ? "speedup" : "slowdown")
    println()
end

println()
println("="^70)
println("    MEMORY USAGE COMPARISON")
println("="^70)
println()

dense_memory_mb = sizeof(RHS_matrix) / 1024^2
@printf("Dense matrix memory:  %.2f MB\n", dense_memory_mb)

for tol in [1e-10, 1e-8]
    RHS_sparse = sparse(RHS_matrix .* (abs.(RHS_matrix) .> tol))
    sparse_memory_mb = (sizeof(RHS_sparse.nzval) + sizeof(RHS_sparse.rowval) + sizeof(RHS_sparse.colptr)) / 1024^2
    savings = 100.0 * (1 - sparse_memory_mb / dense_memory_mb)
    @printf("Sparse matrix (tol=%.0e): %.2f MB (%.1f%% savings)\n", tol, sparse_memory_mb, savings)
end

println()
println("="^70)
println("    RECOMMENDATIONS")
println("="^70)
println()

# Determine if sparse is beneficial
RHS_sparse_1e10 = sparse(RHS_matrix .* (abs.(RHS_matrix) .> 1e-10))
sparsity_pct = 100.0 * (1 - nnz(RHS_sparse_1e10) / length(RHS_matrix))

if sparsity_pct > 90
    println("✓ HIGHLY SPARSE (>90%): Sparse matrix strongly recommended!")
    println("  Expected speedup: 5-10× for matrix-vector products")
elseif sparsity_pct > 70
    println("✓ MODERATELY SPARSE (70-90%): Sparse matrix recommended")
    println("  Expected speedup: 2-5× for matrix-vector products")
elseif sparsity_pct > 50
    println("⚠ SOMEWHAT SPARSE (50-70%): Sparse may help slightly")
    println("  Expected speedup: 1.2-2× for matrix-vector products")
else
    println("✗ NOT SPARSE ENOUGH (<50%): Keep dense representation")
    println("  Sparse would likely be slower due to overhead")
end

println()
