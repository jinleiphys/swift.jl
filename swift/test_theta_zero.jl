#!/usr/bin/env julia

"""
Test if n_gauss affects results even at θ=0 (should match standard method)
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

include("matrices_optimized.jl")
using .matrices_optimized

println("="^80)
println("Testing if n_gauss affects V matrix at θ=0° (no rotation)")
println("="^80)

# Very small system
fermion = true; Jtot = 0.5; T = 0.5; Parity = 1
lmax = 2; lmin = 0; λmax = 4; λmin = 0
s1 = 0.5; s2 = 0.5; s3 = 0.5
t1 = 0.5; t2 = 0.5; t3 = 0.5
MT = -0.5
j2bmax = 1.0
nθ = 6; nx = 8; ny = 8; xmax = 12; ymax = 12; alpha = 1

α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

potname = "AV18"
e2b, ψ = bound2b(grid, potname)

# Get standard V matrix (no rotation)
V_standard = V_matrix_optimized(α, grid, potname)

println("Standard V matrix (no complex scaling):")
println("  V[1,1] = ", V_standard[1,1])
println("  V[2,2] = ", V_standard[2,2])
println()

# Now test complex-scaled version with θ=0 and different n_gauss
θ_deg = 0.0
n_gauss_values = [8, 16, 24, 32]

println("Complex-scaled V matrix with θ=0° (should match standard):")
for n_gauss in n_gauss_values
    V_scaled, _ = V_matrix_optimized_scaled(α, grid, potname,
                                             θ_deg=θ_deg,
                                             n_gauss=n_gauss,
                                             return_components=true)

    diff_11 = abs(V_scaled[1,1] - V_standard[1,1])
    diff_22 = abs(V_scaled[2,2] - V_standard[2,2])

    @printf("  n_gauss=%2d: V[1,1] = %12.6f, diff = %.2e\n", n_gauss, real(V_scaled[1,1]), diff_11)
    @printf("             V[2,2] = %12.6f, diff = %.2e\n", real(V_scaled[2,2]), diff_22)
end

println("\n" * "="^80)
println("If differences are large, the quadrature itself is the problem.")
println("If differences are small, the problem is specific to θ≠0.")
