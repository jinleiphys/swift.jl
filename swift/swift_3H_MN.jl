#!/usr/bin/env julia
# ³H (Tritium) bound state with Minnesota potential (u=1.0 Serber).
# Exact benchmark for NCM VMC comparison.

using LinearAlgebra
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS threads: $(BLAS.get_num_threads())")

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

# ³H quantum numbers
fermion = true; Jtot = 0.5; T = 0.5; Parity = 1; MT = -0.5
s1 = 0.5; s2 = 0.5; s3 = 0.5; t1 = 0.5; t2 = 0.5; t3 = 0.5
lmax = 6; lmin = 0; λmax = 12; λmin = 0
j2bmax = 1.0
nθ = 12; nx = 20; ny = 20; xmax = 16.0; ymax = 16.0; alpha = 1

potname = "MN"

# Generate channels
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin,
        s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, xmax, ymax, Float64(alpha))

# 2-body first
include("twobody.jl")
using .twobodybound

println("\n" * "="^60)
println("  Deuteron with MN potential")
println("="^60)
e2b, ψ = bound2b(grid, potname)

# 3-body with Malfiet-Tjon method
include("MalflietTjon.jl")
using .MalflietTjon

println("\n" * "="^60)
println("  ³H with MN potential (Malfiet-Tjon solver)")
println("="^60)

# Pre-compile with tiny grid
α_test = α3b(fermion, Jtot, T, Parity, 2, 0, 2, 0,
             s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid_test = initialmesh(4, 5, 5, 10.0, 10.0, 1.0)
e2b_test, _ = bound2b(grid_test, potname)
_ = malfiet_tjon_solve_optimized(α_test, grid_test, potname, e2b_test,
    E0=-7.0, E1=-6.0, tolerance=1e-3, max_iterations=1, verbose=false, include_uix=false)
println("Pre-compilation done.")

# Actual calculation
println("\nSolving ³H ground state...")
time_calc = @elapsed begin
    result, ψtot, ψ3 = malfiet_tjon_solve_optimized(α, grid, potname, e2b,
        E0=-7.5, E1=-6.5, tolerance=1e-6, max_iterations=30, verbose=true, include_uix=false)
end
print_convergence_summary(result)
println("Time: $(round(time_calc, digits=1))s")

println("\n" * "="^60)
println("  NCM VMC result: E(³H) = -7.889 ± 0.047 MeV")
println("="^60)
