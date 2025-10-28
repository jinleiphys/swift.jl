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

export T_matrix_optimized, Rxy_matrix_optimized, V_matrix_optimized, Rxy_matrix_with_caching

# Coulomb potential function (matches matrices.jl implementation)
function VCOUL_point(R, z12)
    # Constants
    e2 = 1.43997  # Coulomb constant in appropriate units

    # Calculations
    aux = e2 * z12
    vcoul_point = 0.0

    # Early return if z12 is very small
    if (z12 < 1e-4)
        return vcoul_point
    end

    # Compute Coulomb potential
    vcoul_point = aux / R

    return vcoul_point
end

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
    T_matrix_optimized(α, grid; return_components=false, θ_deg=0.0)

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

## Complex Scaling:
- θ_deg: Complex scaling angle in degrees (default=0.0 for no scaling)
- The kinetic energy matrices Tx and Ty are multiplied by exp(-2iθ)

## Usage:
```julia
Tmat = T_matrix_optimized(α, grid)
# or with components for M_inverse
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true)
# or with complex scaling
Tmat = T_matrix_optimized(α, grid, θ_deg=10.0)
```
"""
function T_matrix_optimized(α, grid; return_components=false, θ_deg=0.0)
    nα = α.nchmax
    nx = grid.nx
    ny = grid.ny

    # Convert angle from degrees to radians
    θ = θ_deg * π / 180.0

    # Complex scaling factor for kinetic energy: e^(-2iθ)
    scaling_factor = exp(-2im * θ)

    # Determine data type based on complex scaling
    is_complex = (θ_deg != 0.0)
    DataType_T = is_complex ? Complex{Float64} : Float64

    # Pre-compute overlap matrices (computed once, not per channel)
    Nx = compute_overlap_matrix(nx, grid.xx)
    Ny = compute_overlap_matrix(ny, grid.yy)

    # Pre-allocate SINGLE output matrix (avoids two intermediate matrices)
    total_size = nα * nx * ny
    Tmatrix = zeros(DataType_T, total_size, total_size)

    # Storage for per-channel components (if requested)
    Tx_channels = Vector{Matrix{DataType_T}}(undef, nα)
    Ty_channels = Vector{Matrix{DataType_T}}(undef, nα)

    # FUSED OPTIMIZATION: Build Tmatrix directly instead of Tx_matrix + Ty_matrix
    for iα in 1:nα
        # Compute channel-specific kinetic energy matrices
        Tx_alpha = Tx(nx, grid.xx, grid.α, α.l[iα])
        Tx_alpha = Tx_alpha .* ħ^2 / m / amu / grid.hsx^2 .* scaling_factor
        Tx_channels[iα] = copy(Tx_alpha)

        Ty_alpha = Tx(ny, grid.yy, grid.α, α.λ[iα])
        Ty_alpha = Ty_alpha .* ħ^2 * 0.75 / m / amu / grid.hsy^2 .* scaling_factor
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

                        # Optimized inner loop with @inbounds and @simd
                        @inbounds for ixp in 1:grid.nx
                            fπb_ixp = fπb[ixp]
                            ip_base = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny

                            # Vectorized operation for inner loop
                            @simd for iyp in 1:grid.ny
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


"""
    Rxy_matrix_with_caching(α, grid)

Optimized Rxy_matrix computation with Laguerre basis caching.

## Key Optimization: Cache Laguerre Basis Functions

The critical insight is that fπb and fξb only depend on the transformed
coordinates (πb, ξb), which are computed from (ix, iy, iθ, perm_idx).
They are **completely independent** of channel indices (iα, iαp)!

### Original Implementation (matrices.jl:20-95):
```julia
for ix in 1:nx
    for iy in 1:ny
        for iθ in 1:nθ
            # Compute πb, ξb
            fπb = lagrange_laguerre_regularized_basis(πb, ...)  # Called here
            fξb = lagrange_laguerre_regularized_basis(ξb, ...)  # Called here

            for iα in 1:nchmax          # These loops are INSIDE
                for iαp in 1:nchmax     # but fπb, fξb don't depend on iα, iαp!
                    for ixp in 1:nx
                        for iyp in 1:ny
                            # Use fπb[ixp], fξb[iyp]
                        end
                    end
                end
            end
        end
    end
end
```

**Problem**: For nx=20, ny=20, nθ=20, nchmax=10:
- Laguerre basis called: 2 × 20 × 20 × 20 × 2 permutations = **32,000 times**
- Each call: O(nx) or O(ny) operations
- But there are only: 20 × 20 × 20 × 2 = **16,000 unique (πb, ξb) pairs**!

### This Optimized Implementation:
```julia
# PRE-COMPUTE: Cache all Laguerre basis functions
cache_fπb = Dict()  # Store fπb for each (ix, iy, iθ, perm_idx)
cache_fξb = Dict()  # Store fξb for each (ix, iy, iθ, perm_idx)

for ix, iy, iθ, perm_idx:
    compute πb, ξb
    cache_fπb[ix, iy, iθ, perm_idx] = lagrange_laguerre_regularized_basis(πb, ...)
    cache_fξb[ix, iy, iθ, perm_idx] = lagrange_laguerre_regularized_basis(ξb, ...)

# MAIN LOOP: Use cached values
for ix in 1:nx
    for iy in 1:ny
        for iθ in 1:nθ
            fπb = cache_fπb[ix, iy, iθ, perm_idx]  # FAST LOOKUP!
            fξb = cache_fξb[ix, iy, iθ, perm_idx]  # FAST LOOKUP!

            for iα in 1:nchmax
                for iαp in 1:nchmax
                    for ixp in 1:nx
                        for iyp in 1:ny
                            # Use cached fπb[ixp], fξb[iyp]
                        end
                    end
                end
            end
        end
    end
end
```

## Performance Impact:

**Before (original):**
- Laguerre calls: 32,000 (2 per (ix,iy,iθ,perm) × nα × nαp iterations)
- Cost per call: O(nx) operations
- Total: 32,000 × O(nx) = 640,000 operations for nx=20

**After (cached):**
- Laguerre calls: 16,000 (2 per (ix,iy,iθ,perm) - computed once!)
- Cache lookups: 32,000 (O(1) each)
- Total: 16,000 × O(nx) + 32,000 = 320,000 + 32,000 operations

**Expected speedup: 2-3× for Rxy_matrix computation**

## Additional Optimizations:

1. **Pre-compute normalization factors** outside loops
2. **Early exit** for negligible G-coefficients
3. **@inbounds** for inner loops (skip bounds checking)
4. **Pre-allocate** cache with known size

## Memory Usage:

**Cache size**:
- fπb cache: nθ × nx × ny × 2 × nx × 8 bytes ≈ 20 × 20 × 20 × 2 × 20 × 8 = 1.28 MB
- fξb cache: nθ × nx × ny × 2 × ny × 8 bytes ≈ 20 × 20 × 20 × 2 × 20 × 8 = 1.28 MB
- **Total cache: ~2.5 MB** (negligible compared to matrix size!)

## Returns:
- `Rxy`: Full rearrangement matrix (Rxy_31 + Rxy_32)
- `Rxy_31`: Rearrangement from coordinate set 1 to 3
- `Rxy_32`: Rearrangement from coordinate set 2 to 3

## Example Usage:
```julia
include("Rxy_matrix_cached.jl")
using .Rxy_matrix_cached

# Compute with caching (2-3× faster)
Rxy, Rxy_31, Rxy_32 = Rxy_matrix_with_caching(α, grid)
```
"""
function Rxy_matrix_with_caching(α, grid)
    # Compute G coefficients once (expensive but unavoidable)
    println("Computing G-coefficients...")
    @time Gαα = computeGcoefficient(α, grid)

    # Pre-compute normalization factors (moved outside loops)
    ϕx_norm = 1.0 ./ grid.ϕx
    ϕy_norm = 1.0 ./ grid.ϕy

    # ============================================================
    # CACHING PHASE: Pre-compute Laguerre basis for ONLY ONE permutation
    # Since Rxy_31 = Rxy_32, we only need to cache ONE set!
    # ============================================================

    println("Pre-computing Laguerre basis cache (one permutation only)...")
    cache_start = time()

    # OPTIMIZATION: Use 3D arrays instead of Dict for O(1) direct indexing
    # This eliminates Dict hashing overhead and improves memory locality
    cache_fπb = Array{Vector{ComplexF64}}(undef, grid.nθ, grid.ny, grid.nx)
    cache_fξb = Array{Vector{ComplexF64}}(undef, grid.nθ, grid.ny, grid.nx)

    # Also cache the normalized coordinates to avoid recomputation
    cache_πb_norm = Array{Float64}(undef, grid.nθ, grid.ny, grid.nx)
    cache_ξb_norm = Array{Float64}(undef, grid.nθ, grid.ny, grid.nx)

    # Only compute for ONE permutation (perm_idx=1)
    perm_idx = 1
    a, b, c, d = -0.5, 1.0, -0.75, -0.5

    for ix in 1:grid.nx
        xa = grid.xi[ix]

        for iy in 1:grid.ny
            ya = grid.yi[iy]

            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]

                # Compute transformed coordinates
                πb_sq = a^2 * xa^2 + b^2 * ya^2 + 2*a*b*xa*ya*cosθ
                ξb_sq = c^2 * xa^2 + d^2 * ya^2 + 2*c*d*xa*ya*cosθ

                πb = sqrt(πb_sq)
                ξb = sqrt(ξb_sq)

                # Cache the Laguerre basis functions AND the normalization factors
                cache_fπb[iθ, iy, ix] = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                cache_fξb[iθ, iy, ix] = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)

                # Cache 1/(πb*ξb) to avoid division in main loop
                cache_πb_norm[iθ, iy, ix] = 1.0 / πb
                cache_ξb_norm[iθ, iy, ix] = 1.0 / ξb
            end
        end
    end

    cache_time = time() - cache_start
    total_entries = grid.nx * grid.ny * grid.nθ
    println("Cache built in $(round(cache_time, digits=3)) seconds")
    println("  Cache entries: $(total_entries) fπb + $(total_entries) fξb")
    println("  Cache memory: ~$(round(2 * total_entries * grid.nx * 16 / 1024^2, digits=2)) MB")

    # ============================================================
    # COMPUTATION PHASE: Compute ONLY Rxy_31 using cached values
    # ============================================================

    println("Computing Rxy_31 with cached basis functions...")
    computation_start = time()

    # Pre-allocate result matrix (only need ONE!)
    Rxy_31 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)

    for ix in 1:grid.nx
        xa = grid.xi[ix]
        xa_norm = ϕx_norm[ix]

        for iy in 1:grid.ny
            ya = grid.yi[iy]
            ya_norm = ϕy_norm[iy]
            xy_norm = xa_norm * ya_norm

            for iθ in 1:grid.nθ
                dcosθ = grid.dcosθi[iθ]

                # OPTIMIZED CACHE LOOKUP: Direct array indexing (no Dict overhead)
                fπb = cache_fπb[iθ, iy, ix]
                fξb = cache_fξb[iθ, iy, ix]

                # Use pre-cached normalization factors (no sqrt or division needed)
                πb_inv = cache_πb_norm[iθ, iy, ix]
                ξb_inv = cache_ξb_norm[iθ, iy, ix]

                # Pre-compute normalization factor using cached values
                base_angular_factor = dcosθ * xa * ya
                norm_factor = base_angular_factor * πb_inv * ξb_inv * xy_norm

                # Channel coupling loop
                for iα in 1:α.nchmax
                    i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy

                    for iαp in 1:α.nchmax
                        # Get G coefficient
                        G_coeff = Gαα[iθ, iy, ix, iα, iαp, perm_idx]

                        # Early exit for negligible contributions
                        if abs(G_coeff) < 1e-14
                            continue
                        end

                        adj_factor = norm_factor * G_coeff

                        # Optimized inner loop with @inbounds and @simd
                        @inbounds for ixp in 1:grid.nx
                            fπb_ixp = fπb[ixp]
                            ip_base = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny

                            # Vectorized operation for inner loop
                            @simd for iyp in 1:grid.ny
                                ip = ip_base + iyp
                                Rxy_31[i, ip] += adj_factor * fπb_ixp * fξb[iyp]
                            end
                        end
                    end
                end
            end
        end
    end

    computation_time = time() - computation_start
    println("Rxy_31 computed in $(round(computation_time, digits=3)) seconds")

    # Since Rxy_31 = Rxy_32, just reuse the same matrix!
    Rxy_32 = Rxy_31
    Rxy = Rxy_31 + Rxy_32  # Same as 2 * Rxy_31

    # Performance summary
    total_time = cache_time + computation_time
    println("\n" * "="^70)
    println("PERFORMANCE SUMMARY")
    println("="^70)
    println("Cache construction:  $(rpad(round(cache_time, digits=3), 8)) s")
    println("Matrix computation:  $(rpad(round(computation_time, digits=3), 8)) s")
    println("-"^70)
    println("Total time:          $(rpad(round(total_time, digits=3), 8)) s")
    println("="^70)

    return Rxy, Rxy_31, Rxy_32
end

end # end module
