#!/usr/bin/env julia

"""
Test n_gauss convergence for a fixed θ
For a given θ, results should converge as n_gauss increases
"""

using LinearAlgebra
using Printf

BLAS.set_num_threads(Sys.CPU_THREADS)

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

include("twobody.jl")
using .twobodybound

include("MalflietTjon.jl")
using .MalflietTjon

println("="^80)
println("Testing n_gauss convergence for fixed θ=5°")
println("="^80)

# Small system for faster testing
fermion = true; Jtot = 0.5; T = 0.5; Parity = 1
lmax = 4; lmin = 0; λmax = 8; λmin = 0
s1 = 0.5; s2 = 0.5; s3 = 0.5
t1 = 0.5; t2 = 0.5; t3 = 0.5
MT = -0.5
j2bmax = 1.0
nθ = 8; nx = 12; ny = 12; xmax = 14; ymax = 14; alpha = 1

α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

println("System: nx=$(grid.nx), ny=$(grid.ny)")
println("Channels: $(α.nchmax)")

potname = "AV18"
e2b, ψ = bound2b(grid, potname)

println("\nTesting θ=5° with different n_gauss values:")
println("="^80)

θ_test = 5.0
n_gauss_values = [12, 24, 36, 48, 60]  # From 1x to 5x grid size

results = []

for n_gauss in n_gauss_values
    println("\nn_gauss = $n_gauss ($(round(n_gauss/grid.nx, digits=1))× grid.nx)")
    println("-"^80)
    
    result, _, _ = malfiet_tjon_solve_optimized(
        α, grid, potname, e2b,
        E0=-7.5, E1=-6.5,
        tolerance=1e-6, max_iterations=20,
        verbose=false, include_uix=false,
        θ_deg=θ_test, n_gauss=n_gauss
    )
    
    E = result.energy
    push!(results, (n_gauss, E))
    
    @printf("  Energy: %.6f MeV\n", real(E))
    if abs(imag(E)) > 1e-10
        @printf("  (Imaginary part: %.6e MeV)\n", imag(E))
    end
end

println("\n" * "="^80)
println("CONVERGENCE ANALYSIS")
println("="^80)
@printf("%-12s | %-18s | %-18s\n", "n_gauss", "Energy (MeV)", "Δ from previous")
println("-"^80)

for i in 1:length(results)
    n_gauss, E = results[i]
    E_real = real(E)
    
    if i == 1
        @printf("%-12d | %18.6f | %18s\n", n_gauss, E_real, "-")
    else
        E_prev = real(results[i-1][2])
        ΔE = abs(E_real - E_prev)
        @printf("%-12d | %18.6f | %18.6e\n", n_gauss, E_real, ΔE)
    end
end

println("="^80)

# Check convergence
E_final = real(results[end][2])
E_second_last = real(results[end-1][2])
convergence = abs(E_final - E_second_last)

if convergence < 1e-4
    println("\n✓ CONVERGED: Results stable to < 0.1 keV")
else
    println("\n✗ NOT CONVERGED: Still changing by $(round(convergence*1000, digits=2)) keV")
    println("  This indicates a problem with the implementation!")
end

