# Refined Rxy_matrix optimization - v2
# Smarter approach: optimize the algorithm without expensive caching overhead

module RxyOptimizedV2

include("laguerre.jl")
using .Laguerre
include("Gcoefficient.jl")
using .Gcoefficient

export Rxy_matrix_optimized_v2

"""
    Rxy_matrix_optimized_v2(α, grid)

Optimized Rxy_matrix with better algorithmic structure

## Key Optimizations:
1. Merged Rxy_31 and Rxy_32 loops (single pass, better cache locality)
2. Pre-compute invariant quantities outside loops
3. Skip negligible G-coefficient contributions early
4. Optimized innerloop structure
5. NO caching overhead for small systems

## Expected Performance:
- 1.5-2× speedup for typical systems
- Lower memory overhead than caching approach
- Better scaling for larger systems

## Usage:
```julia
Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized_v2(α, grid)
```
"""
function Rxy_matrix_optimized_v2(α, grid)
    # Pre-allocate result matrices
    Rxy_32 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
    Rxy_31 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)

    # Compute G coefficients once
    Gαα = computeGcoefficient(α, grid)

    # Pre-compute phi products for normalization (invariant across iθ)
    ϕx_norm = zeros(grid.nx)
    ϕy_norm = zeros(grid.ny)
    for ix in 1:grid.nx
        ϕx_norm[ix] = 1.0 / grid.ϕx[ix]
    end
    for iy in 1:grid.ny
        ϕy_norm[iy] = 1.0 / grid.ϕy[iy]
    end

    # Transformation parameters: (target_matrix, perm_idx, a, b, c, d)
    transforms = [
        (Rxy_31, 1, -0.5, 1.0, -0.75, -0.5),
        (Rxy_32, 2, -0.5, -1.0, 0.75, -0.5)
    ]

    # Main computation loop - process both transformations simultaneously
    for ix in 1:grid.nx
        xa = grid.xi[ix]
        xa_norm = ϕx_norm[ix]

        for iy in 1:grid.ny
            ya = grid.yi[iy]
            ya_norm = ϕy_norm[iy]
            xy_norm = xa_norm * ya_norm  # Pre-compute product

            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]
                dcosθ = grid.dcosθi[iθ]

                # Pre-compute common angular factor
                base_angular_factor = dcosθ * xa * ya

                # Process both transformations in a single pass
                for (Rxy_target, perm_idx, a, b, c, d) in transforms
                    # Compute transformed coordinates
                    πb_sq = a^2 * xa^2 + b^2 * ya^2 + 2*a*b*xa*ya*cosθ
                    ξb_sq = c^2 * xa^2 + d^2 * ya^2 + 2*c*d*xa*ya*cosθ

                    πb = sqrt(πb_sq)
                    ξb = sqrt(ξb_sq)

                    # Compute Laguerre basis functions (unavoidable cost)
                    fπb = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                    fξb = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)

                    # Pre-compute normalization factor for this point
                    norm_factor = base_angular_factor / (πb * ξb) * xy_norm

                    # Channel coupling loop
                    for iα in 1:α.nchmax
                        i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy

                        for iαp in 1:α.nchmax
                            # Get G coefficient and check if negligible
                            G_coeff = Gαα[iθ, iy, ix, iα, iαp, perm_idx]

                            if abs(G_coeff) < 1e-14
                                continue  # Skip negligible contributions
                            end

                            # Final adjustment factor
                            adj_factor = norm_factor * G_coeff

                            # Optimized inner loop: compute once, use twice
                            # Use @inbounds for performance (skip bounds checking)
                            @inbounds for ixp in 1:grid.nx
                                fπb_ixp = fπb[ixp]
                                ip_base = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny

                                # Manual loop unrolling for small ny (if ny < 20, consider unrolling)
                                for iyp in 1:grid.ny
                                    ip = ip_base + iyp
                                    Rxy_target[i, ip] += adj_factor * fπb_ixp * fξb[iyp]
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    Rxy = Rxy_31 + Rxy_32
    return Rxy, Rxy_31, Rxy_32
end

end # module
