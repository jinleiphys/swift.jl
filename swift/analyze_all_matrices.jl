println("="^70)
println("COMPREHENSIVE MATRIX SPARSITY ANALYSIS")
println("="^70)
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
n = α.nchmax * grid.nx * grid.ny

println("  Matrix size: $n × $n")
println()

include("twobody.jl")
using .twobodybound
potname="AV18"
e2b, ψ = bound2b(grid, potname)

include("matrices.jl")
using .matrices

function analyze_matrix(name, matrix)
    n_total = length(matrix)
    n_nonzero = count(x -> abs(x) > 1e-12, matrix)
    sparsity = 100 * (1 - n_nonzero / n_total)
    mem_dense = sizeof(matrix) / 1e9

    println("$name Matrix:")
    println("  Size: $(size(matrix))")
    println("  Nonzero elements: $(n_nonzero) / $(n_total)")
    println("  Sparsity: $(round(sparsity, digits=2))%")
    println("  Dense memory: $(round(mem_dense, digits=2)) GB")

    if sparsity > 90
        println("  ✓✓ HIGHLY SPARSE - excellent candidate for sparse format")
    elseif sparsity > 50
        println("  ✓ MODERATELY SPARSE - good candidate for sparse format")
    else
        println("  ✗ DENSE - keep as dense matrix")
    end
    println()

    return sparsity
end

println("="^70)
println("Building and analyzing all matrices...")
println("="^70)
println()

# 1. T matrix (Kinetic energy)
println("[1/5] T Matrix (Kinetic Energy)")
T, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
sparsity_T = analyze_matrix("T", T)

# 2. V matrix (Potential)
println("[2/5] V Matrix (Potential)")
V = V_matrix(α, grid, potname)
sparsity_V = analyze_matrix("V", V)

# 3. B matrix (Norm)
println("[3/5] B Matrix (Norm)")
B = Bmatrix(α, grid)
sparsity_B = analyze_matrix("B", B)

# 4. Rxy matrix (Rearrangement)
println("[4/5] Rxy Matrix (Rearrangement)")
Rxy_result = Rxy_matrix(α, grid)
Rxy = isa(Rxy_result, Tuple) ? Rxy_result[1] : Rxy_result
sparsity_Rxy = analyze_matrix("Rxy", Rxy)

# 5. LHS matrix = E*B - T - V (at typical energy)
println("[5/5] LHS Matrix (E*B - T - V at E = -7.5 MeV)")
E_test = -7.5
LHS = E_test * B - T - V
sparsity_LHS = analyze_matrix("LHS", LHS)

println("="^70)
println("OPTIMIZATION RECOMMENDATIONS")
println("="^70)
println()

recommendations = []

if sparsity_V > 90
    push!(recommendations, "✓ V matrix: ALREADY OPTIMIZED (using sparse in RHS cache)")
end

if sparsity_T > 90
    push!(recommendations, "✓ T matrix ($(round(sparsity_T, digits=1))% sparse): Convert to sparse format")
    push!(recommendations, "  → Affects: LHS = E*B - T - V construction")
end

if sparsity_B > 90
    push!(recommendations, "✓ B matrix ($(round(sparsity_B, digits=1))% sparse): Convert to sparse format")
    push!(recommendations, "  → Affects: LHS = E*B - T - V construction")
end

if sparsity_Rxy > 50
    push!(recommendations, "? Rxy matrix ($(round(sparsity_Rxy, digits=1))% sparse): Moderate benefit")
    push!(recommendations, "  → Affects: V*Rxy multiplication (already done)")
end

if sparsity_LHS > 90
    push!(recommendations, "✓ LHS matrix ($(round(sparsity_LHS, digits=1))% sparse): Use sparse for GMRES/solves")
    push!(recommendations, "  → Affects: Matrix inversions and eigenvalue computations")
end

if isempty(recommendations)
    println("✗ No further optimization opportunities found")
else
    for (i, rec) in enumerate(recommendations)
        println("$i. $rec")
    end
end

println()
println("="^70)
println("PRIORITY OPTIMIZATIONS")
println("="^70)

priorities = []
if sparsity_T > 90
    push!(priorities, ("T matrix", sparsity_T, "High - used in every energy evaluation"))
end
if sparsity_B > 90
    push!(priorities, ("B matrix", sparsity_B, "High - used in every energy evaluation"))
end
if sparsity_LHS > 90
    push!(priorities, ("LHS operations", sparsity_LHS, "Medium - used in solves but may be complex"))
end

if !isempty(priorities)
    sort!(priorities, by=x->x[2], rev=true)
    for (i, (name, sp, reason)) in enumerate(priorities)
        println("$i. $name ($(round(sp, digits=1))% sparse)")
        println("   Priority: $reason")
        println()
    end
else
    println("No high-priority optimizations remaining.")
    println("V matrix optimization (RHS cache) was the main bottleneck.")
end
