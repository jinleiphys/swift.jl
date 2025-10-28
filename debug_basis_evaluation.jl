#!/usr/bin/env julia

"""
Debug script to understand basis function evaluation and normalization
"""

using Printf

# Add modules
push!(LOAD_PATH, joinpath(@__DIR__, "swift"))
include("general_modules/mesh.jl")
using .mesh

include("swift/matrices_optimized.jl")
using .matrices_optimized

println("="^70)
println("Debugging Basis Function Evaluation")
println("="^70)

# Create a simple mesh
nθ = 4
nx = 5
ny = 5
xmax = 10.0
ymax = 10.0
alpha = 1.0

grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

println("\nMesh information:")
println("  nx = $(grid.nx)")
println("  Mesh points xi = ", grid.xi)
println("  Weights dxi = ", grid.dxi)
println("  Normalization ϕx = ", grid.ϕx)

# Test basis function evaluation at mesh points
println("\n" * "="^70)
println("Testing φᵢ(xⱼ) for i,j = 1:nx")
println("="^70)
println("Should satisfy Lagrange property: φᵢ(xⱼ) ≈ δᵢⱼ")
println()

for ix in 1:grid.nx
    print("φ_$(ix)(x_j): ")
    for jx in 1:grid.nx
        r = grid.xi[jx]  # grid.xi is already in physical units (scaled)
        phi_val = matrices_optimized.evaluate_laguerre_basis_at_point(r, ix, grid.xi, grid.ϕx, grid.α, grid.hsx)
        @printf("%8.4f  ", real(phi_val))
    end
    println()
end

# Check normalization
println("\n" * "="^70)
println("Checking normalization: Σⱼ wⱼ φᵢ(xⱼ) φₖ(xⱼ)")
println("="^70)
println("Should give δᵢₖ for orthonormal basis")
println()

for ix in 1:grid.nx
    print("<φ_$(ix)|φ_k>: ")
    for kx in 1:grid.nx
        overlap = 0.0
        for jx in 1:grid.nx
            r = grid.xi[jx]  # grid.xi is already in physical units (scaled)
            phi_i = matrices_optimized.evaluate_laguerre_basis_at_point(r, ix, grid.xi, grid.ϕx, grid.α, grid.hsx)
            phi_k = matrices_optimized.evaluate_laguerre_basis_at_point(r, kx, grid.xi, grid.ϕx, grid.α, grid.hsx)
            overlap += grid.dxi[jx] * real(phi_i * conj(phi_k))
        end
        @printf("%8.4f  ", overlap)
    end
    println()
end

println("\n" * "="^70)
