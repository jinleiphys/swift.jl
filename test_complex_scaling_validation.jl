#!/usr/bin/env julia

"""
Validation test for complex-scaled potential matrix implementation.

This script tests that V_matrix_optimized_scaled gives the same result as
V_matrix_optimized when θ=0 (no complex scaling).

The backward rotation method should be exact at θ=0, providing a strong
validation of the implementation correctness.
"""

println("\n" * "="^80)
println("Complex Scaling Validation Test")
println("Testing V_matrix_optimized_scaled implementation")
println("="^80)

# Required packages
using Printf

# Add the swift module to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "swift"))

# Load required modules
println("\nLoading modules...")
include("general_modules/channels.jl")
using .channels

include("general_modules/mesh.jl")
using .mesh

include("swift/matrices_optimized.jl")
using .matrices_optimized

println("✓ Modules loaded successfully\n")

# Set up a simple test case
println("Setting up test case for ³H (tritium)...")
println("-"^80)

# Nuclear system parameters for ³H (following swift_3H.jl)
fermion = true
Jtot = 0.5   # Total angular momentum
T = 0.5      # Total isospin
Parity = 1   # Parity (+1)
MT = -0.5    # MT projection (nnn system)
j2bmax = 2.0 # Two-body angular momentum cutoff

# Angular momentum ranges
lmax = 6     # Maximum l (orbital angular momentum)
lmin = 0     # Minimum l
λmax = 12    # Maximum λ (hypermomentum)
λmin = 0     # Minimum λ

# Particle properties (neutrons for ³H)
s1 = 0.5
s2 = 0.5
s3 = 0.5
t1 = 0.5
t2 = 0.5
t3 = 0.5

println("  Fermion: $(fermion)")
println("  J = $(Jtot), T = $(T), Parity = $(Parity), MT = $(MT)")
println("  Angular momentum: lmax=$(lmax), λmax=$(λmax)")
println("  Two-body cutoff: j2bmax = $(j2bmax)")

# Generate channel structure
println("\nGenerating three-body channels...")
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
println("  Number of channels: $(α.nchmax)")
println("  Two-body channels: $(α.α2b.nchmax)")

# Set up mesh
println("\nSetting up hyperspherical mesh...")
nθ = 12      # Number of angular grid points
nx = 20      # Number of x grid points
ny = 20      # Number of y grid points
xmax = 16.0  # Maximum x coordinate (fm)
ymax = 16.0  # Maximum y coordinate (fm)
alpha = 1.0  # Laguerre parameter

println("  Grid size: nx=$(nx), ny=$(ny), nθ=$(nθ)")
println("  Range: xmax=$(xmax) fm, ymax=$(ymax) fm")
println("  Laguerre parameter: α=$(alpha)")

grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("✓ Mesh initialized successfully")

# Choose nuclear potential
potname = "MT-V"  # Malfliet-Tjon potential (simplest for testing)
println("\nUsing potential: $(potname)")

# Run validation test
println("\n" * "="^80)
println("Running validation test...")
println("="^80)

# Test with different n_gauss values to check convergence
n_gauss_values = [30, 40, 50, 60]
tolerance = 1e-8

println("\nTesting convergence with different quadrature points:")
println("-"^80)

test_results = []
for n_gauss in n_gauss_values
    println("\n--- Testing with n_gauss = $(n_gauss) ---")
    passed, abs_err, rel_err = test_V_scaled_at_zero(α, grid, potname,
                                                      n_gauss=n_gauss,
                                                      tolerance=tolerance)
    push!(test_results, (n_gauss, passed, abs_err, rel_err))
end

# Summary
println("\n" * "="^80)
println("SUMMARY OF VALIDATION TESTS")
println("="^80)
println("Potential: $(potname)")
println("Mesh: $(nx)×$(ny) grid points")
println("Tolerance: $(tolerance)")
println("-"^80)
println("n_gauss  |  Status  |  Max Abs Error  |  Max Rel Error")
println("-"^80)

all_passed = true
for (n_gauss, passed, abs_err, rel_err) in test_results
    status = passed ? "PASS ✓" : "FAIL ✗"
    println(@sprintf("%7d  |  %7s  |  %14.6e  |  %14.6e",
                     n_gauss, status, abs_err, rel_err))
    global all_passed = all_passed && passed
end

println("="^80)

if all_passed
    println("\n✓ ALL VALIDATION TESTS PASSED")
    println("\nThe backward rotation implementation is correct!")
    println("You can now use V_matrix_optimized_scaled with confidence.")
else
    println("\n✗ SOME TESTS FAILED")
    println("\nConsider:")
    println("  1. Increasing n_gauss for better accuracy")
    println("  2. Checking the implementation of evaluate_laguerre_basis_at_point")
    println("  3. Verifying the interpolate_potential function")
end

println("\n" * "="^80)
println("Test completed")
println("="^80 * "\n")
