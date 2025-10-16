#!/usr/bin/env julia
# Minimal test for RHS cache only

println("Loading modules...")
include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

# Parameters
fermion=true; Jtot = 0.5; T = 0.5; Parity=1
lmax=8; lmin=0; λmax=20; λmin=0; s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5; MT=-0.5
j2bmax=2.0
nθ=12; nx=30; ny=30; xmax=20; ymax=20; alpha=1

α= α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)
grid= initialmesh(nθ,nx,ny,Float64(xmax),Float64(ymax),Float64(alpha))

n_total = α.nchmax * grid.nx * grid.ny
println("\nMatrix size: $(n_total) × $(n_total)")
println("Memory per matrix: $(round(n_total^2 * 8 / 1e9, digits=2)) GB\n")

include("twobody.jl")
using .twobodybound
potname="AV18"
e2b, ψ = bound2b(grid, potname)

# Build matrices
println("Building T, V, Rxy matrices...")
include("matrices.jl")
using .matrices

matrix_time = @elapsed begin
    V = V_matrix(α, grid, potname, e2b)
    println("  V matrix: $(size(V))")

    Rxy_31 = Rxy_matrix(α, grid)
    println("  Rxy matrix: $(size(Rxy_31))")

    # Build V_x_diag_ch (needed for RHS cache)
    V_x_diag_ch = [zeros(grid.nx, grid.nx) for _ in 1:α.nchmax]
    for iα in 1:α.nchmax
        for ix in 1:grid.nx
            for jx in 1:grid.nx
                i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + 1
                j = (iα-1)*grid.nx*grid.ny + (jx-1)*grid.ny + 1
                V_x_diag_ch[iα][ix, jx] = V[i, j]
            end
        end
    end
end
println("Matrix construction: $(round(matrix_time, digits=2))s\n")

# Now test RHS cache
include("MalflietTjon.jl")
using .MalflietTjon

println("="^70)
println("TESTING RHS CACHE PRECOMPUTATION")
println("="^70)

println("\nTest 1: Measure V*Rxy alone...")
vrxy_time = @elapsed begin
    VRxy = V * Rxy_31
end
println("  V * Rxy: $(round(vrxy_time, digits=2))s")
println("  Result size: $(size(VRxy))")

println("\nTest 2: Full RHS cache with optimization...")
rhs_time = @elapsed begin
    RHS_cache = precompute_RHS_cache(V, V_x_diag_ch, Rxy_31, α, grid; V_UIX=nothing)
end
println("  RHS cache total: $(round(rhs_time, digits=2))s")

println("\n" * "="^70)
println("RESULTS:")
println("  V * Rxy alone:        $(round(vrxy_time, digits=2))s")
println("  Full RHS cache:       $(round(rhs_time, digits=2))s")
println("  Additional overhead:  $(round(rhs_time - vrxy_time, digits=2))s")
println("="^70)
