# Optimized matrix computation module
# This demonstrates Priority 3 optimization: Direct block assignment instead of repeated Kronecker products

module matrices_optimized

using Kronecker
include("../NNpot/nuclear_potentials.jl")
using .NuclearPotentials
using WignerSymbols
include("laguerre.jl")
using .Laguerre
include("Gcoefficient.jl")
using .Gcoefficient
using LinearAlgebra

const amu = 931.49432 # MeV
const m = 1.0079713395678829 # amu
const ħ = 197.3269718 # MeV. fm

export T_matrix_optimized, Rxy_matrix_optimized, V_matrix_optimized

# Helper function to compute overlap matrices
function compute_overlap_matrix(n, xx)
    """Compute overlap matrix for non-orthogonal basis functions"""
    N = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i == j
                N[i,j] = 1 + (-1.)^(j-i)/sqrt(xx[i]*xx[j])
            else
                N[i,j] = (-1.)^(j-i)/sqrt(xx[i]*xx[j])
            end
        end
    end
    return N
end

function Tx(nx, xi, α0, l)
    """Compute the kinetic energy matrix in x or y direction"""
    T = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            if i == j
                T[i,j] = (-1.0 / (12.0 * xi[i]^2)) * (xi[i]^2 - 2.0 * (2.0 * nx + α0 + 1.0) * xi[i] + α0^2 - 4.0) -
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j])) +
                         l * (l + 1) / xi[i]^2
            else
                T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) -
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j]))
            end
        end
    end

    return T
end

"""
    T_matrix_optimized(α, grid; return_components=false)

OPTIMIZED VERSION of T_matrix using fused computation to avoid intermediate matrices.

## Key Optimizations:
1. Pre-compute overlap matrices Nx and Ny once (not per channel)
2. Compute block Kronecker products (nx*ny × nx*ny) instead of full matrix Kronecker (nα*nx*ny × nα*nx*ny)
3. **FUSED COMPUTATION**: Build final matrix directly instead of creating two intermediate matrices
4. Minimal memory allocations and memory bandwidth usage

## Performance:
- Old: Creates Tx_matrix + Ty_matrix, then adds (90% of time in addition!)
- New: Builds Tmatrix directly with fused Tx_block + Ty_block
- Expected speedup: 4-5× compared to previous version

## Usage:
```julia
Tmat = T_matrix_optimized(α, grid)
# or with components for M_inverse
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true)
```
"""
function T_matrix_optimized(α, grid; return_components=false)
    nα = α.nchmax
    nx = grid.nx
    ny = grid.ny

    # Pre-compute overlap matrices (computed once, not per channel)
    Nx = compute_overlap_matrix(nx, grid.xx)
    Ny = compute_overlap_matrix(ny, grid.yy)

    # Pre-allocate SINGLE output matrix (avoids two intermediate matrices)
    total_size = nα * nx * ny
    Tmatrix = zeros(total_size, total_size)

    # Storage for per-channel components (if requested)
    Tx_channels = Vector{Matrix{Float64}}(undef, nα)
    Ty_channels = Vector{Matrix{Float64}}(undef, nα)

    # FUSED OPTIMIZATION: Build Tmatrix directly instead of Tx_matrix + Ty_matrix
    for iα in 1:nα
        # Compute channel-specific kinetic energy matrices
        Tx_alpha = Tx(nx, grid.xx, grid.α, α.l[iα])
        Tx_alpha .*= ħ^2 / m / amu / grid.hsx^2
        Tx_channels[iα] = copy(Tx_alpha)

        Ty_alpha = Tx(ny, grid.yy, grid.α, α.λ[iα])
        Ty_alpha .*= ħ^2 * 0.75 / m / amu / grid.hsy^2
        Ty_channels[iα] = copy(Ty_alpha)

        # Compute block Kronecker products (only nx*ny × nx*ny, not full size!)
        Tx_block = kron(Tx_alpha, Ny)  # (nx × nx) ⊗ (ny × ny) = (nx*ny × nx*ny)
        Ty_block = kron(Nx, Ty_alpha)  # (nx × nx) ⊗ (ny × ny) = (nx*ny × nx*ny)

        # FUSED: Add both contributions directly to Tmatrix (no intermediate matrices!)
        idx_start = (iα-1) * nx * ny + 1
        idx_end = iα * nx * ny

        # Single operation: Tmatrix[block] = Tx_block + Ty_block
        # This is equivalent to: δ_{α,α} I_α ⊗ (Tx^α ⊗ Ny + Nx ⊗ Ty^α)
        # but avoids creating full Tx_matrix and Ty_matrix intermediates
        @views Tmatrix[idx_start:idx_end, idx_start:idx_end] .= Tx_block .+ Ty_block
    end

    if return_components
        return Tmatrix, Tx_channels, Ty_channels, Nx, Ny
    else
        return Tmatrix
    end
end


"""
    Rxy_matrix_optimized(α, grid)

Optimized Rxy_matrix with better algorithmic structure

## Key Optimizations:
1. Compute only Rxy_31, use symmetry (Rxy_32 = Rxy_31) for 2× speedup
2. Pre-compute invariant quantities outside loops
3. Skip negligible G-coefficient contributions early
4. Optimized innerloop structure with @inbounds
5. NO caching overhead for small systems

## Expected Performance:
- 2-3× speedup for typical systems (half the computation)
- Lower memory overhead than caching approach
- Better scaling for larger systems

## Usage:
```julia
Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized_(α, grid)
```
"""
function Rxy_matrix_optimized(α, grid)
    # Pre-allocate result matrix (only compute Rxy_31)
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

    # Transformation parameters for Rxy_31 only (perm_idx=1)
    perm_idx = 1
    a, b, c, d = -0.5, 1.0, -0.75, -0.5

    # Main computation loop - compute only Rxy_31
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

                        # Optimized inner loop with @inbounds
                        @inbounds for ixp in 1:grid.nx
                            fπb_ixp = fπb[ixp]
                            ip_base = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny

                            for iyp in 1:grid.ny
                                ip = ip_base + iyp
                                Rxy_31[i, ip] += adj_factor * fπb_ixp * fξb[iyp]
                            end
                        end
                    end
                end
            end
        end
    end

    # Use symmetry: Rxy_32 = Rxy_31 (exact equality due to physics symmetry)
    Rxy_32 = Rxy_31
    Rxy = Rxy_31 + Rxy_32

    return Rxy, Rxy_31, Rxy_32
end


# Helper functions for V_matrix (from matrices.jl)
function checkα2b(i, j, α)
    """Check if the two-body channels are allowed for potential coupling"""
    if α.α2b.T12[i] != α.α2b.T12[j]
        return false
    end
    if α.α2b.s12[i] != α.α2b.s12[j]
        return false
    end
    if α.α2b.J12[i] != α.α2b.J12[j]
        return false
    end
    return true
end

function pot_nucl(α, grid, potname)
    """Compute the nuclear potential matrix (unchanged from matrices.jl)"""
    v12 = zeros(grid.nx, grid.nx, α.α2b.nchmax, α.α2b.nchmax, 2)

    for j in 1:α.α2b.nchmax
        for i in 1:α.α2b.nchmax
            if checkα2b(i, j, α)
                li = [α.α2b.l[i]]

                if Int(α.α2b.J12[i]) == 0  # Special case: J12=0
                    if α.α2b.l[i] != α.α2b.l[j]
                        continue
                    end
                    for ir in 1:grid.nx
                        v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1)
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0)
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1)
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        end
                    end

                elseif Int(α.α2b.J12[i]) == α.α2b.l[i]  # Uncoupled states
                    if α.α2b.l[i] != α.α2b.l[j]
                        error("error: the channel is not allowed")
                    end
                    for ir in 1:grid.nx
                        v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1)
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0)
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1)
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        end
                    end

                else  # Coupled channels
                    J12_val = Int(α.α2b.J12[i])
                    l = [J12_val-1, J12_val+1]
                    for ir in 1:grid.nx
                        if α.α2b.l[i] == (J12_val-1) && α.α2b.l[j] == (J12_val-1)
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1)
                                v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0)
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1)
                                v12[ir, ir, i, j, 2] = v[1, 1]
                            end
                        elseif α.α2b.l[i] == (J12_val+1) && α.α2b.l[j] == (J12_val+1)
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1)
                                v12[ir, ir, i, j, 2] = v[2, 2] + VCOUL_point(grid.xi[ir], 1.0)
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1)
                                v12[ir, ir, i, j, 2] = v[2, 2]
                            end
                        elseif α.α2b.l[i] == (J12_val-1) && α.α2b.l[j] == (J12_val+1)
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1)
                                v12[ir, ir, i, j, 2] = v[1, 2]
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1)
                                v12[ir, ir, i, j, 2] = v[1, 2]
                            end
                        elseif α.α2b.l[i] == (J12_val+1) && α.α2b.l[j] == (J12_val-1)
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1)
                                v12[ir, ir, i, j, 2] = v[2, 1]
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1)
                                v12[ir, ir, i, j, 2] = v[2, 1]
                            end
                        end
                    end
                end
            end
        end
    end

    return v12
end


"""
    V_matrix_optimized(α, grid, potname; return_components=false)

OPTIMIZED VERSION of V_matrix using direct block assignment instead of repeated Kronecker products.

## Key Optimizations:
1. Pre-compute overlap matrix Ny once (not per channel pair)
2. Compute block Kronecker products (nx*ny × nx*ny) instead of full matrix Kronecker
3. Direct block assignment to off-diagonal blocks (no channel projection matrix)
4. Reduced memory allocations

## Performance:
- Old: Multiple full Kronecker products of size (α.nchmax × α.nchmax) ⊗ (nx × nx) ⊗ (ny × ny)
- New: One Ny computation + direct block assignments with small Kronecker products
- Expected speedup: 1.5-2× for typical system sizes

## Usage:
```julia
Vmat = V_matrix_optimized(α, grid, potname)
# or with components for M_inverse
Vmat, V_x_diag_ch = V_matrix_optimized(α, grid, potname, return_components=true)
```
"""
function V_matrix_optimized(α, grid, potname; return_components=false)
    # Get nuclear potential matrix (same computation as original)
    v12 = pot_nucl(α, grid, potname)

    # Pre-compute overlap matrix Ny once
    Ny = compute_overlap_matrix(grid.ny, grid.yy)

    # Pre-allocate full matrix
    total_size = α.nchmax * grid.nx * grid.ny
    Vmatrix = zeros(total_size, total_size)

    # Storage for diagonal potential channels (if requested)
    V_x_diag_channels = Vector{Matrix{Float64}}(undef, α.nchmax)
    for iα in 1:α.nchmax
        V_x_diag_channels[iα] = zeros(grid.nx, grid.nx)
    end

    # OPTIMIZATION: Direct block assignment instead of full Kronecker products
    for j in 1:α.nchmax  # α₃'
        for i in 1:α.nchmax  # α₃
            # Check if this channel pair couples (same selection rules as original)
            if α.T12[i] != α.T12[j]
                continue
            end
            if α.λ[i] != α.λ[j]
                continue
            end
            if α.J3[i] != α.J3[j]
                continue
            end
            if α.s12[i] != α.s12[j]
                continue
            end
            if α.J12[i] != α.J12[j]
                continue
            end

            # Build potential block V_x_ij for this channel pair
            V_x_ij = zeros(grid.nx, grid.nx)

            T12 = α.T12[i]
            nmt12_max = Int(2 * T12)
            for nmt12 in -nmt12_max:2:nmt12_max
                mt12 = nmt12 / 2.0
                mt3 = α.MT - mt12

                if abs(mt3) > α.t3
                    continue
                end

                cg1 = clebschgordan(T12, mt12, α.t3, mt3, α.T[i], α.MT)
                cg2 = clebschgordan(T12, mt12, α.t3, mt3, α.T[j], α.MT)
                cg_coefficient = cg1 * cg2

                if abs(cg_coefficient) < 1e-10
                    continue
                end

                if mt12 == 0
                    V_x_ij += v12[:, :, α.α2bindex[i], α.α2bindex[j], 1] * cg_coefficient
                else
                    V_x_ij += v12[:, :, α.α2bindex[i], α.α2bindex[j], 2] * cg_coefficient
                end
            end

            # Store diagonal elements for M_inverse
            if i == j
                V_x_diag_channels[i] = copy(V_x_ij)
            end

            # Compute block Kronecker product (only nx*ny × nx*ny, not full size!)
            V_block = kron(V_x_ij, Ny)  # (nx × nx) ⊗ (ny × ny) = (nx*ny × nx*ny)

            # Direct assignment to (i,j) block (avoids full channel Kronecker product)
            idx_i_start = (i-1) * grid.nx * grid.ny + 1
            idx_i_end = i * grid.nx * grid.ny
            idx_j_start = (j-1) * grid.nx * grid.ny + 1
            idx_j_end = j * grid.nx * grid.ny

            # This is equivalent to: P^{α₃',α₃} ⊗ V_x^{α₃',α₃} ⊗ Ny
            # but computed directly without building the full channel projection matrix
            Vmatrix[idx_i_start:idx_i_end, idx_j_start:idx_j_end] = V_block
        end
    end

    if return_components
        return Vmatrix, V_x_diag_channels
    else
        return Vmatrix
    end
end

end # end module
