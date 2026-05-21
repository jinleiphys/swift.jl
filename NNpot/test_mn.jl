#!/usr/bin/env julia
"""Test Minnesota potential: deuteron benchmark.

Expected: E_d ≈ -2.224 MeV for ³S₁ channel with u=1.0 (Serber).
"""

# Include modules
include("nuclear_potentials.jl")
include("../general_modules/mesh.jl")
include("../swift/twobody.jl")

using .NuclearPotentials
using .mesh
using .twobodybound

println("=" ^ 60)
println("  Minnesota Potential: Deuteron Benchmark")
println("=" ^ 60)

# --- Quick sanity check: potential values ---
println("\n--- Potential values check ---")
println("³S₁ (l=0, s=1): V = V_R + V_T")
println("  r (fm)     V_code       V_R+V_T      diff")
for r in [0.5, 1.0, 1.5, 2.0, 3.0]
    V_R = 200.0 * exp(-1.487 * r^2)
    V_T = -178.0 * exp(-0.639 * r^2)
    V_expected = V_R + V_T
    V = potential_matrix("MN", r, [0, 2], 1, 1, 0, 0)
    diff = V[1,1] - V_expected
    println("  $(r)       $(round(V[1,1], digits=6))   $(round(V_expected, digits=6))   $(diff)")
    @assert abs(diff) < 1e-10 "MN potential mismatch at r=$r!"
end

# Check S-D coupling is zero
r = 1.0
V = potential_matrix("MN", r, [0, 2], 1, 1, 0, 0)
@assert V[1,2] == 0.0 && V[2,1] == 0.0 "MN should have zero S-D coupling!"
println("\nS-D coupling: V[1,2] = $(V[1,2]), V[2,1] = $(V[2,1]) (both zero) ✓")
println("All quick tests PASSED.")

# --- Deuteron bound state ---
println("\n" * "=" ^ 60)
println("  Deuteron Bound State (MN, u=1.0 Serber)")
println("=" ^ 60)

# Convergence study: vary nx with fixed xmax
println("\n--- Convergence vs mesh size (xmax=20.0 fm) ---")
for nx in [20, 30, 40, 50]
    grid = initialmesh(4, nx, 4, 20.0, 10.0, 0.0)
    energies, _ = bound2b(grid, "MN")
    if length(energies) > 0
        println("\n>>> nx=$nx: E_d = $(round(real(energies[1]), digits=6)) MeV")
    else
        println("\n>>> nx=$nx: No bound state found!")
    end
end

# Also try with xmax=30 for safety
println("\n--- Convergence vs xmax (nx=40) ---")
for xmax in [15.0, 20.0, 25.0, 30.0]
    grid = initialmesh(4, 40, 4, xmax, 10.0, 0.0)
    energies, _ = bound2b(grid, "MN")
    if length(energies) > 0
        println("\n>>> xmax=$xmax: E_d = $(round(real(energies[1]), digits=6)) MeV")
    else
        println("\n>>> xmax=$xmax: No bound state found!")
    end
end

println("\n" * "=" ^ 60)
println("Expected: E_d ≈ -2.224 MeV")
println("=" ^ 60)
