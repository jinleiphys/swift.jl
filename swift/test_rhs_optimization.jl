#!/usr/bin/env julia

# Quick test to measure RHS cache optimization performance

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

# Use same parameters as swift_3H.jl
fermion=true; Jtot = 0.5; T = 0.5; Parity=1
lmax=8; lmin=0; λmax=20; λmin=0; s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5; MT=-0.5
j2bmax=2.0
nθ=12; nx=30; ny=30; xmax=20; ymax=20; alpha=1

println("Generating channels...")
α= α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)
println("  Number of channels: ", α.nchmax)

println("\nGenerating mesh...")
grid= initialmesh(nθ,nx,ny,Float64(xmax),Float64(ymax),Float64(alpha))
println("  Grid dimensions: nx=$(grid.nx), ny=$(grid.ny)")
println("  Total matrix size: $(α.nchmax * grid.nx * grid.ny)")

include("twobody.jl")
using .twobodybound

println("\nComputing 2-body bound state...")
potname="AV18"
e2b, ψ = bound2b(grid, potname)
println("  Deuteron energy: $(e2b) MeV")

# Now just run a single iteration of Malfiet-Tjon to measure RHS cache time
println("\n" * "="^70)
println("Running Malfiet-Tjon solver (measures RHS cache)...")
println("="^70)

include("MalflietTjon.jl")
using .MalflietTjon

# Just one iteration with verbose timing
E0_guess = -7.5
E1_guess = -6.5

total_time = @elapsed begin
    result_opt, ψtot_opt, ψ3_opt = malfiet_tjon_solve_optimized(α, grid, potname, e2b,
                               E0=E0_guess,
                               E1=E1_guess,
                               tolerance=1e-6,
                               max_iterations=5,  # Only 5 iterations for quick test
                               verbose=true,
                               include_uix=false)
end

println("\n" * "="^70)
println("Total time: $(round(total_time, digits=2)) seconds")
println("="^70)
