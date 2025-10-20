using LinearAlgebra
using Printf

# PERFORMANCE: Use all available CPU cores for BLAS operations
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS threads set to: $(BLAS.get_num_threads()) (CPU cores: $(Sys.CPU_THREADS))")
println()

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

# ============================================================================
# THREE-BODY SYSTEM: ³He (Helium-3)
# ============================================================================
# ³He consists of 2 protons + 1 neutron
# Quantum numbers: J^π = (1/2)⁺, T = 1/2, MT = +1/2
# Mirror nucleus of ³H (tritium) with MT = -1/2
#
# Key difference from ³H: Coulomb repulsion between protons
# - Reduces binding energy: ³He (~7.72 MeV) vs ³H (~8.48 MeV)
# - Coulomb potential: Point Coulomb VCOUL_point added in matrices.jl for MT > 0
# - For AV18: Coulomb is subtracted in nuclear_potentials.jl and re-added consistently
# - Net physics: Same as Fortran AV18, but Coulomb handling is explicit
# ============================================================================

fermion=true; Jtot = 0.5; T = 0.5; Parity=1
lmax=8; lmin=0; λmax=20; λmin=0; s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5; MT=+0.5 # +0.5 for 3He (2p + 1n)
j2bmax=5.0  # Maximum J12 (two-body angular momentum)
nθ=12; nx=20; ny=20; xmax=16; ymax=16; alpha=1

α= α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)  # the last variable define the parity of the pair system, if not defined then consider both parities.
grid= initialmesh(nθ,nx,ny,Float64(xmax),Float64(ymax),Float64(alpha));

include("twobody.jl")
using .twobodybound

potname="AV18"
e2b, ψ =bound2b(grid, potname);


include("MalflietTjon.jl")
using .MalflietTjon

# Solve using Malfiet-Tjon method with initial energy guesses
println("="^70)
println("    MALFIET-TJON METHOD FOR ³He GROUND STATE")
println("="^70)
println("System: ³He (2 protons + 1 neutron)")
println("Coulomb interaction: ENABLED (MT > 0)")
println()

# Use better initial energy guesses based on expected ³He binding energy
# ³He is less bound than ³H due to Coulomb repulsion
# Experimental: B.E.(³He) ≈ 7.72 MeV vs B.E.(³H) ≈ 8.48 MeV
# Energy = -B.E., so use guesses around -7.0 to -8.0 MeV

    E0_guess = -7.0
    E1_guess = -6.0

println("\n" * "="^70)
println("    PRE-COMPILATION WARM-UP")
println("="^70)
println("Pre-compiling functions with minimal test case...")

# Use TINY grid for fast pre-compilation (only compiles functions, doesn't do real work)
fermion_test=true; Jtot_test = 0.5; T_test = 0.5; Parity_test=1
lmax_test=2; lmin_test=0; λmax_test=2; λmin_test=0
j2bmax_test=1.0
nθ_test=4; nx_test=5; ny_test=5; xmax_test=10; ymax_test=10; alpha_test=1

α_test = α3b(fermion_test,Jtot_test,T_test,Parity_test,lmax_test,lmin_test,λmax_test,λmin_test,s1,s2,s3,t1,t2,t3,MT,j2bmax_test)
grid_test = initialmesh(nθ_test,nx_test,ny_test,Float64(xmax_test),Float64(ymax_test),Float64(alpha_test))
e2b_test, _ = bound2b(grid_test, potname)

# Quick pre-compile run (5×5 grid = 125×125 matrices, very fast!)
_ = malfiet_tjon_solve_optimized(α_test, grid_test, potname, e2b_test,
                              E0=E0_guess,
                              E1=E1_guess,
                              tolerance=1e-3,  # Loose tolerance
                              max_iterations=1,  # Just 1 iteration
                              verbose=false,
                              include_uix=true)

println("✓ Pre-compilation complete (used 5×5 test grid)")
println()

println("="^70)
println("    ACTUAL PERFORMANCE MEASUREMENT")
println("="^70)
println()

# Test: Optimized version with accurate timing
println("Testing OPTIMIZED malfiet_tjon_solve_optimized...")
println("-"^70)
time_optimized = @elapsed begin
    result_opt, ψtot_opt, ψ3_opt = malfiet_tjon_solve_optimized(α, grid, potname, e2b,
                              E0=E0_guess,
                              E1=E1_guess,
                              tolerance=1e-6,
                              max_iterations=30,
                              verbose=true,
                              include_uix=false)  # Include UIX three-body force
end
print_convergence_summary(result_opt)

println("\n" * "="^70)
println("    ³He BINDING ENERGY RESULTS")
println("="^70)
@printf("Calculated binding energy: %.4f MeV\n", -result_opt.energy)
@printf("Experimental value:        ~7.718 MeV\n")
@printf("Difference:                %.4f MeV\n", -result_opt.energy - 7.718)
println()
println("Note: Coulomb interaction between protons is included.")
println("      Expected to be less bound than ³H by ~0.76 MeV.")
println("="^70)

println("\n\n")
