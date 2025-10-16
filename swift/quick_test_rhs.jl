println("Quick test of optimized RHS cache...")
include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels, .mesh

# Small test
fermion=true; Jtot=0.5; T=0.5; Parity=1
lmax=2; lmin=0; λmax=2; λmin=0; s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5; MT=-0.5
j2bmax=1.0; nθ=4; nx=10; ny=10; xmax=10; ymax=10; alpha=1

α = α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)
grid = initialmesh(nθ,nx,ny,Float64(xmax),Float64(ymax),Float64(alpha))

include("twobody.jl")
using .twobodybound
e2b, _ = bound2b(grid, "AV18")

include("matrices.jl")
using .matrices
V = V_matrix(α, grid, "AV18")
Rxy = Rxy_matrix(α, grid)

# Extract V_x_diag_ch
V_x_diag_ch = [zeros(grid.nx, grid.nx) for _ in 1:α.nchmax]
for iα in 1:α.nchmax
    for ix in 1:grid.nx, jx in 1:grid.nx
        i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + 1
        j = (iα-1)*grid.nx*grid.ny + (jx-1)*grid.ny + 1
        V_x_diag_ch[iα][ix, jx] = V[i, j]
    end
end

include("MalflietTjon.jl")
using .MalflietTjon

println("Testing with $(α.nchmax) channels, $(grid.nx)×$(grid.ny) grid...")
@time RHS = precompute_RHS_cache(V, V_x_diag_ch, Rxy, α, grid; V_UIX=nothing)
println("Success!")
