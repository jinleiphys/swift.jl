#!/usr/bin/env julia

"""
Test that complex scaling gives the same energy as standard calculation
For a bound state, the energy should be independent of θ
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
println("Testing Complex Scaling: Energy should be independent of θ")
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

println("System: nx=$(grid.nx), ny=$(grid.ny), channels=$(α.nchmax)")
println()

potname = "AV18"
e2b, ψ = bound2b(grid, potname)

# Test different θ values
θ_values = [0.0, 2.0, 5.0, 8.0]
n_gauss = 60  # Use converged value

println("Testing with n_gauss=$(n_gauss) (converged)")
println("="^80)
@printf("%-10s | %-18s | %-18s\n", "θ (deg)", "Energy (MeV)", "Δ from θ=0")
println("-"^80)

energies = []

for θ_deg in θ_values
    result, _, _ = malfiet_tjon_solve_optimized(
        α, grid, potname, e2b,
        E0=-7.5, E1=-6.5,
        tolerance=1e-6, max_iterations=20,
        verbose=false, include_uix=false,
        θ_deg=θ_deg, n_gauss=n_gauss
    )

    E = result.energy
    push!(energies, E)

    if θ_deg == 0.0
        @printf("%-10.1f | %18.6f | %18s\n", θ_deg, real(E), "-")
    else
        E_ref = energies[1]
        ΔE = abs(real(E) - real(E_ref))
        @printf("%-10.1f | %18.6f | %18.6e MeV\n", θ_deg, real(E), ΔE)
    end
end

println("="^80)

# Check if energies are consistent
E_ref = real(energies[1])
max_deviation = maximum(abs(real(E) - E_ref) for E in energies)

if max_deviation < 0.01  # Within 10 keV
    println("\n✓ SUCCESS: Complex scaling working correctly!")
    @printf("  Max energy deviation: %.6f MeV (< 0.01 MeV)\n", max_deviation)
else
    println("\n✗ PROBLEM: Energies depend on θ!")
    @printf("  Max energy deviation: %.6f MeV\n", max_deviation)
    println("  This indicates an issue with complex scaling implementation.")
end
