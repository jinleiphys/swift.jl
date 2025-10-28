#!/usr/bin/env julia

"""
Debug the Gauss-Laguerre quadrature to understand why results depend on n_gauss
"""

using LinearAlgebra
using FastGaussQuadrature
using Printf

# Set up a simple grid like in the tests
nx = 8
alpha = 1.0
hsx = 0.488  # Approximate value from test output

println("="^80)
println("Debugging Gauss-Laguerre Quadrature for Complex Scaling")
println("="^80)
println("Grid parameters: nx=$nx, alpha=$alpha, hsx=$hsx")
println()

# Test different n_gauss values
n_gauss_values = [8, 16, 24, 32]

for n_gauss in n_gauss_values
    println("n_gauss = $n_gauss:")
    println("-"^80)

    # Get Gauss-Laguerre quadrature points and weights
    xi_unscaled, dxi_unscaled = gausslaguerre(n_gauss, alpha)
    r_gauss = xi_unscaled .* hsx

    # Remove Laguerre weight to get plain dr integration weights
    w_gauss = dxi_unscaled .* hsx ./ (xi_unscaled.^alpha .* exp.(-xi_unscaled))

    # Print first few and last few quadrature points
    println("  Quadrature points (first 3):")
    for k in 1:min(3, n_gauss)
        @printf("    r[%2d] = %.6f fm, weight = %.6e\n", k, r_gauss[k], w_gauss[k])
    end
    if n_gauss > 6
        println("    ...")
    end
    println("  Quadrature points (last 3):")
    for k in max(1, n_gauss-2):n_gauss
        @printf("    r[%2d] = %.6f fm, weight = %.6e\n", k, r_gauss[k], w_gauss[k])
    end

    # Check total weight (should be ∞ for dr weight, but let's see what we get)
    total_weight = sum(w_gauss)
    println("  Total weight: $(total_weight)")

    # Test integrating a simple function: exp(-r/2)
    # Exact integral: ∫₀^∞ exp(-r/2) dr = 2
    test_func(r) = exp(-r/2.0)
    integral_approx = sum(w_gauss[k] * test_func(r_gauss[k]) for k in 1:n_gauss)
    exact_integral = 2.0
    error = abs(integral_approx - exact_integral)

    @printf("  Test integral ∫ e^(-r/2) dr: approx=%.8f, exact=%.8f, error=%.2e\n",
            integral_approx, exact_integral, error)
    println()
end

println("="^80)
println("ANALYSIS")
println("="^80)
println("If the test integral converges quickly, the quadrature is working correctly.")
println("If not, there may be an issue with how weights are computed.")
