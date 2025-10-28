#!/usr/bin/env julia

"""
Test V matrix elements directly with different n_gauss
This isolates the problem to the V matrix computation
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

# Need matrices_optimized module
include("matrices_optimized.jl")
using .matrices_optimized

println("="^80)
println("Testing V matrix elements with different n_gauss at theta=5 degrees")
println("="^80)

# Very small system for detailed inspection
fermion = true; Jtot = 0.5; T = 0.5; Parity = 1
lmax = 2; lmin = 0; λmax = 4; λmin = 0
s1 = 0.5; s2 = 0.5; s3 = 0.5
t1 = 0.5; t2 = 0.5; t3 = 0.5
MT = -0.5
j2bmax = 1.0
nθ = 6; nx = 8; ny = 8; xmax = 12; ymax = 12; alpha = 1

α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

println("System: nx=$(grid.nx), ny=$(grid.ny), channels=$(α.nchmax)")
println()

potname = "AV18"
e2b, ψ = bound2b(grid, potname)

θ_test = 5.0
n_gauss_values = [16, 32, 48, 64, 80]  # 2x to 10x grid size

@printf("Computing V matrices with theta=%.1f degrees for different n_gauss:\n", θ_test)
println("="^80)

V_matrices = []

for n_gauss in n_gauss_values
    println("\nn_gauss = $n_gauss ($(round(n_gauss/grid.nx, digits=1))x grid.nx)")
    println("-"^80)
    
    V, V_x_diag_ch = V_matrix_optimized_scaled(α, grid, potname, 
                                                θ_deg=θ_test, 
                                                n_gauss=n_gauss, 
                                                return_components=true)
    
    push!(V_matrices, (n_gauss, V, V_x_diag_ch))
    
    # Print some statistics
    println("  Matrix size: $(size(V))")
    println("  Matrix type: $(eltype(V))")
    println("  Norm: $(norm(V))")
    
    # Sample some diagonal elements
    n_sample = min(5, size(V,1))
    println("  Sample diagonal elements:")
    for i in 1:n_sample
        val = V[i,i]
        @printf("    V[%d,%d] = %.6f + %.6fi\n", i, i, real(val), imag(val))
    end
end

println("\n" * "="^80)
println("COMPARING MATRIX ELEMENTS")
println("="^80)

# Compare first matrix with others
n_ref, V_ref, _ = V_matrices[1]
println("Reference: n_gauss = $n_ref")
println()

for i in 2:length(V_matrices)
    n_gauss, V, _ = V_matrices[i]
    
    # Compute differences
    diff = V - V_ref
    max_diff = maximum(abs.(diff))
    mean_diff = sum(abs.(diff)) / length(diff)
    rel_diff = max_diff / maximum(abs.(V_ref))
    
    println("Comparison with n_gauss = $n_gauss:")
    @printf("  Max absolute difference: %.6e\n", max_diff)
    @printf("  Mean absolute difference: %.6e\n", mean_diff)
    @printf("  Max relative difference: %.6e (%.2f%%)\n", rel_diff, rel_diff*100)
    
    # Check specific elements that changed most
    idx = argmax(abs.(diff))
    i_max, j_max = Tuple(idx)
    @printf("  Largest change at V[%d,%d]:\n", i_max, j_max)
    @printf("    n_gauss=%d: %.6e + %.6ei\n", n_ref, real(V_ref[i_max, j_max]), imag(V_ref[i_max, j_max]))
    @printf("    n_gauss=%d: %.6e + %.6ei\n", n_gauss, real(V[i_max, j_max]), imag(V[i_max, j_max]))
    @printf("    Difference: %.6e\n", abs(diff[i_max, j_max]))
    println()
end

# Also check the V_x diagonal components
println("="^80)
println("CHECKING V_x DIAGONAL COMPONENTS (channel 1)")
println("="^80)

for i in 1:length(V_matrices)
    n_gauss, _, V_x_diag_ch = V_matrices[i]
    V_x_1 = V_x_diag_ch[1]  # First channel
    
    println("\nn_gauss = $n_gauss:")
    println("  V_x size: $(size(V_x_1))")
    println("  V_x norm: $(norm(V_x_1))")
    
    # Sample diagonal
    n_sample = min(3, size(V_x_1,1))
    println("  Sample V_x diagonal:")
    for ix in 1:n_sample
        val = V_x_1[ix, ix]
        @printf("    V_x[%d,%d] = %.6f + %.6fi\n", ix, ix, real(val), imag(val))
    end
end

println("\n" * "="^80)
println("ANALYSIS")
println("="^80)

# Check if matrices are converging
if length(V_matrices) >= 3
    _, V1, _ = V_matrices[end-1]
    _, V2, _ = V_matrices[end]
    
    final_diff = maximum(abs.(V2 - V1))
    final_rel = final_diff / maximum(abs.(V2))
    
    if final_rel < 1e-4
        println("OK: V matrices converging: relative change < 0.01%")
    else
        println("ERROR: V matrices NOT converging: relative change = $(round(final_rel*100, digits=3))%")
        println("  This is the source of the energy convergence problem!")
    end
end

