println("Checking sparsity of V and Rxy matrices...")
println()

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

# Use the same parameters as swift_3H.jl
fermion=true; Jtot = 0.5; T = 0.5; Parity=1
lmax=8; lmin=0; λmax=20; λmin=0; s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5; MT=-0.5
j2bmax=2.0
nθ=12; nx=30; ny=30; xmax=20; ymax=20; alpha=1

println("Generating channels and mesh...")
α = α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)
grid = initialmesh(nθ,nx,ny,Float64(xmax),Float64(ymax),Float64(alpha))

println("  Channels: $(α.nchmax)")
println("  Grid: $(grid.nx)×$(grid.ny)")
println("  Total matrix size: $(α.nchmax * grid.nx * grid.ny)")
println()

include("twobody.jl")
using .twobodybound
potname="AV18"
println("Computing 2-body bound state...")
e2b, ψ = bound2b(grid, potname)
println()

include("matrices.jl")
using .matrices

println("="^70)
println("Building V matrix...")
println("="^70)
V_time = @elapsed V = V_matrix(α, grid, potname)
n = size(V, 1)
println("V matrix: $(size(V))")
println("Build time: $(round(V_time, digits=2))s")

# Check sparsity of V
n_total = n * n
n_nonzero_V = count(!iszero, V)
sparsity_V = 100 * (1 - n_nonzero_V / n_total)

println()
println("V Matrix Sparsity Analysis:")
println("  Total elements:    $(n_total)")
println("  Nonzero elements:  $(n_nonzero_V)")
println("  Zero elements:     $(n_total - n_nonzero_V)")
println("  Sparsity:          $(round(sparsity_V, digits=2))%")
println()

if sparsity_V > 50
    println("✓ V is SPARSE! ($(round(sparsity_V, digits=1))% zeros)")
else
    println("✗ V is DENSE (only $(round(sparsity_V, digits=1))% zeros)")
end

println()
println("="^70)
println("Building Rxy matrix...")
println("="^70)
Rxy_time = @elapsed Rxy_result = Rxy_matrix(α, grid)

# Rxy_matrix returns a tuple, extract the first element
if isa(Rxy_result, Tuple)
    Rxy = Rxy_result[1]
    println("Rxy returned as tuple, using first element")
else
    Rxy = Rxy_result
end

println("Rxy matrix: $(size(Rxy))")
println("Build time: $(round(Rxy_time, digits=2))s")

# Check sparsity of Rxy
n_nonzero_Rxy = count(x -> abs(x) > 1e-12, Rxy)  # Use tolerance for complex numbers
sparsity_Rxy = 100 * (1 - n_nonzero_Rxy / n_total)

println()
println("Rxy Matrix Sparsity Analysis:")
println("  Total elements:    $(n_total)")
println("  Nonzero elements:  $(n_nonzero_Rxy)")
println("  Zero elements:     $(n_total - n_nonzero_Rxy)")
println("  Sparsity:          $(round(sparsity_Rxy, digits=2))%")
println()

if sparsity_Rxy > 50
    println("✓ Rxy is SPARSE! ($(round(sparsity_Rxy, digits=1))% zeros)")
else
    println("✗ Rxy is DENSE (only $(round(sparsity_Rxy, digits=1))% zeros)")
end

println()
println("="^70)
println("CONCLUSION")
println("="^70)

if sparsity_V > 50 && sparsity_Rxy > 50
    println("✓✓ BOTH matrices are sparse!")
    println("   → Can use SparseArrays for massive speedup")
    println("   → Expected speedup: 10-100x on V*Rxy multiplication")
elseif sparsity_V > 50 || sparsity_Rxy > 50
    println("✓ One matrix is sparse")
    println("   → Can use sparse matrix operations for moderate speedup")
    println("   → Expected speedup: 2-10x")
else
    println("✗ Both matrices are dense")
    println("   → No sparsity optimization possible")
    println("   → Must reduce matrix size or use GPU acceleration")
end

println()
println("Current dense multiplication: ~250 seconds")
if sparsity_V > 50 && sparsity_Rxy > 50
    potential_time = 250 * (1 - sparsity_V/100) * (1 - sparsity_Rxy/100) / 0.01
    println("Potential sparse time: ~$(round(potential_time, digits=1)) seconds")
    println("Potential speedup: $(round(250/potential_time, digits=1))x")
end
