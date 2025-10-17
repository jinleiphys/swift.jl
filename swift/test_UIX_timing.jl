using LinearAlgebra

# PERFORMANCE: Use all available CPU cores for BLAS operations
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS threads set to: $(BLAS.get_num_threads()) (CPU cores: $(Sys.CPU_THREADS))")
println()

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

# SMALL MODEL SPACE for fast testing
fermion=true; Jtot = 0.5; T = 0.5; Parity=1
lmax=2; lmin=0; λmax=4; λmin=0   # REDUCED from lmax=8, λmax=20
s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5; MT=-0.5
j2bmax=1.0  # REDUCED from 2.0
nθ=8; nx=10; ny=10; xmax=12; ymax=12; alpha=1  # REDUCED from 20x20

α= α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)
grid= initialmesh(nθ,nx,ny,Float64(xmax),Float64(ymax),Float64(alpha));

include("twobody.jl")
using .twobodybound

potname="AV18"
e2b, ψ =bound2b(grid, potname);

include("MalflietTjon.jl")
using .MalflietTjon

# Test UIX timing with small model space
println("\n" * "="^70)
println("    UIX TIMING TEST (SMALL MODEL SPACE)")
println("="^70)
println("Model space: lmax=$lmax, λmax=$λmax, j2bmax=$j2bmax, grid=$(nx)×$(ny)")
println("="^70)

E0_guess = -7.5
E1_guess = -6.5

println("\nTesting OPTIMIZED malfiet_tjon_solve_optimized...")
println("-"^70)
time_optimized = @elapsed begin
    result_opt, ψtot_opt, ψ3_opt = malfiet_tjon_solve_optimized(α, grid, potname, e2b,
                               E0=E0_guess,
                               E1=E1_guess,
                               tolerance=1e-5,  # Looser tolerance for speed
                               max_iterations=10,
                               verbose=true,
                               include_uix=true)
end

println("\n" * "="^70)
println("    TOTAL TIME: $(round(time_optimized, digits=2)) seconds")
println("="^70)
