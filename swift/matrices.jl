module matrices 
using Kronecker
include("../NNpot/nuclear_potentials.jl")
using .NuclearPotentials
using WignerSymbols
include("laguerre.jl")
using .Laguerre
include("Gcoefficient.jl")
using .Gcoefficient
using LinearAlgebra

const amu= 931.49432 # MeV
const m=1.0079713395678829 # amu
const ħ=197.3269718 # MeV. fm

export T_matrix, V_matrix, Bmatrix,
       MInverseCacheVSector, precompute_M_inverse_cache_vsector, M_inverse_operator_cached_vsector,
       group_channels_by_v_sector

# 1.008665 amu for neutron  amu=931.49432 MeV

function pot_nucl(α, grid, potname)
    # Compute the nuclear potential matrix
    # Parameters:
    # α: channel index
    # grid: grid object containing nx, xi, and other parameters
    # proton m1=+1/2  neutron m2=-1/2
    # for the current function, I only consider the local potential(AV8,NIJM,REID,AV14,AV18), for the non-local potential, one needs to modify this function 
    v12 = zeros(grid.nx, grid.nx, α.α2b.nchmax, α.α2b.nchmax, 2)  # Initialize potential matrix the last dimension is for the isospin 1 for np pair and 2 for nn(MT<0) or pp pair(MT>0)

    for j in 1:α.α2b.nchmax
        for i in 1:α.α2b.nchmax
            if checkα2b(i, j, α)
                li=[α.α2b.l[i]]
                # Compute the potential matrix elements
                if Int(α.α2b.J12[i]) == 0  # Special case: J12=0
                    if α.α2b.l[i] != α.α2b.l[j]
                        continue  # Skip if l[i] != l[j] for J12=0 case
                    end
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir],li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                        end
                    end
                elseif Int(α.α2b.J12[i]) == α.α2b.l[i]  # Uncoupled states: J12=l (but not J12=0)
                    if α.α2b.l[i] != α.α2b.l[j]
                        error("error: the channel is not allowed")
                    end 
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir],li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                        end
                    end
                else
                    # For coupled channels, both i and j should have the same J12 due to delta function constraint
                    J12_val = Int(α.α2b.J12[i])  # Could also use α.α2b.J12[j] since they should be equal
                    l = [J12_val-1, J12_val+1]
                    for ir in 1:grid.nx  
                        if α.α2b.l[i] == (J12_val-1) && α.α2b.l[j] == (J12_val-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 1]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.α2b.l[i] == (J12_val+1) && α.α2b.l[j] == (J12_val+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 2] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 2]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.α2b.l[i] == (J12_val-1) && α.α2b.l[j] == (J12_val+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 2] 
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 2]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.α2b.l[i] == (J12_val+1) && α.α2b.l[j] == (J12_val-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 1]  
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 1]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        end
                    end 
                end 
            end
        end
    end
    
    return v12  
end

 function Bmatrix(α,grid)
    # compute the B matrix for the Generalized eigenvalue problem
    Iα = Matrix{Float64}(I, α.nchmax, α.nchmax)
    Nx=zeros(grid.nx, grid.nx)
    Ny=zeros(grid.ny, grid.ny)
    for i in 1:grid.nx
        for j in 1:grid.nx
            if i == j
                Nx[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
            else
                Nx[i,j] = (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
            end
        end
    
    end 

    for i in 1:grid.ny
        for j in 1:grid.ny
            if i == j
                Ny[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
            else
                Ny[i,j] = (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
            end
        end
    
    end

    Bmatrix = Iα ⊗ Nx ⊗ Ny

    return Bmatrix


 end 


 function checkα2b(i,j,α)
    # Check if the two-body channels are allowed for potential coupling
    # The two-body potential should only couple channels with identical quantum numbers
    if α.α2b.T12[i] == α.α2b.T12[j] && α.α2b.s12[i] == α.α2b.s12[j] && α.α2b.J12[i] == α.α2b.J12[j] && (-1)^α.α2b.l[i] == (-1)^α.α2b.l[j]
        return true
    else
        return false
    end
 end 


 function VCOUL_point(R, z12)   # use to compute the Coulomb potential
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



# ============================================================================
# V-sector block-diagonal M⁻¹ (generalised Malfiet-Tjon split)
# ============================================================================
#
# In the V-sector formulation, channels are grouped by the V-conservation
# sector key q = (J12, T12, s12, λ, J3). V is block-diagonal across sectors,
# so M(E) = EB - H₀ - V is also block-diagonal in q (in contrast to the
# strict channel-diagonal M which uses only V_αα).
#
# Each sector block of M is inverted by a Kronecker eigendecomposition of
# the (n_q · n_x) × (n_q · n_x) coupled matrix N^{(q)}_x⁻¹ · H^{(q)}_x
# (where H^{(q)}_x = block-diag(T_x^a) + V^(q)), and the standard
# n_y × n_y problem N_y⁻¹·T_y (sector-uniform by construction).

"""
    group_channels_by_v_sector(α) -> Vector{Vector{Int}}

Group three-body channels into V-conservation sectors. Within each sector all
channels share the same (J12, T12, s12, λ, J3); these are exactly the deltas
enforced by `V_matrix` / `V_matrix_optimized` channel-coupling selection rules.

Returns a vector of channel-index vectors, one entry per sector. Sectors are
ordered by first-occurrence of their member channels.
"""
function group_channels_by_v_sector(α)
    seen = Dict{NTuple{5, Float64}, Int}()  # key → sector index
    sector_channels = Vector{Vector{Int}}()
    for i in 1:α.nchmax
        key = (α.J12[i], α.T12[i], α.s12[i], Float64(α.λ[i]), α.J3[i])
        if haskey(seen, key)
            push!(sector_channels[seen[key]], i)
        else
            push!(sector_channels, [i])
            seen[key] = length(sector_channels)
        end
    end
    return sector_channels
end

"""
    MInverseCacheVSector

Cache for V-sector block-diagonal M⁻¹ preconditioner.  Per-sector eigen-
decompositions are stored.  Energy enters only through the diagonal
`D_inv_blocks` recomputed via `M_inverse_operator_cached_vsector(E, cache)`.

# Fields
- `sector_channels`: Vector{Vector{Int}}, channel indices belonging to each sector
- `U_blocks`: Per-sector  𝒰_x ⊗ U_y, size (n_q · n_x · n_y) × (n_q · n_x · n_y)
- `U_inv_N_inv_blocks`: Per-sector (𝒰_x ⊗ U_y)⁻¹ · (I_{n_q} ⊗ N_x⁻¹ ⊗ N_y⁻¹)
- `dx_arrays`: Per-sector eigenvalues of (I_{n_q} ⊗ N_x⁻¹) · H_x^{(q)} (length n_q · n_x)
- `dy_arrays`: Per-sector eigenvalues of N_y⁻¹ · T_y (length n_y) — sector-uniform but cached per sector for symmetry
- `nx`, `ny`: mesh sizes
- `nchmax`: total channel count (for vector indexing)
"""
struct MInverseCacheVSector{T<:Union{Float64, ComplexF64}}
    sector_channels::Vector{Vector{Int}}
    U_blocks::Vector{Matrix{T}}
    U_inv_N_inv_blocks::Vector{Matrix{T}}
    dx_arrays::Vector{Vector{T}}
    dy_arrays::Vector{Vector{T}}
    nx::Int
    ny::Int
    nchmax::Int
end

"""
    precompute_M_inverse_cache_vsector(α, grid, Tx_channels, Ty_channels, V_x_full, Nx, Ny)

Precompute the V-sector block-diagonal M⁻¹ cache.

# Arguments
- `α`, `grid`: channel + mesh structures
- `Tx_channels::Vector`, `Ty_channels::Vector`: per-channel kinetic matrices (n_x × n_x and n_y × n_y)
- `V_x_full::Matrix{Matrix{T}}` (size α.nchmax × α.nchmax): cross-channel V_x blocks. Entry [i, j] must be the n_x × n_x matrix V_{ij}(x); entries between channels in different V-sectors are unused (may be zero). The strict-channel diagonal entries V_x_full[i, i] equal the existing `V_x_diag_ch[i]`.
- `Nx`, `Ny`: overlap matrices

# Returns
`MInverseCacheVSector` for use with `M_inverse_operator_cached_vsector(E, cache)`.
"""
function precompute_M_inverse_cache_vsector(α, grid, Tx_channels, Ty_channels, V_x_full, Nx, Ny)
    sector_channels = group_channels_by_v_sector(α)
    n_sec = length(sector_channels)
    nx = grid.nx
    ny = grid.ny

    DataType_T = eltype(Tx_channels[1])

    Nx_inv = inv(Nx)
    Ny_inv = inv(Ny)
    N_inv_xy = kron(Nx_inv, Ny_inv)  # n_x n_y × n_x n_y

    U_blocks = Vector{Matrix{DataType_T}}(undef, n_sec)
    U_inv_N_inv_blocks = Vector{Matrix{DataType_T}}(undef, n_sec)
    dx_arrays = Vector{Vector{DataType_T}}(undef, n_sec)
    dy_arrays = Vector{Vector{DataType_T}}(undef, n_sec)

    for (q, chans) in enumerate(sector_channels)
        n_q = length(chans)

        # Build the coupled x-Hamiltonian H^{(q)}_x of size (n_q · n_x) × (n_q · n_x):
        #   diagonal-in-channel block (i_a, i_a) = T_x^{chans[i_a]} + V_{chans[i_a], chans[i_a]}
        #   off-diagonal block (i_a, i_b) = V_{chans[i_a], chans[i_b]} (for i_a ≠ i_b)
        Hx_q = zeros(DataType_T, n_q * nx, n_q * nx)
        for (i_a, a) in enumerate(chans)
            row = (i_a - 1) * nx + 1 : i_a * nx
            # Kinetic on the diagonal channel block
            Hx_q[row, row] .+= Tx_channels[a]
            for (i_b, b) in enumerate(chans)
                col = (i_b - 1) * nx + 1 : i_b * nx
                Hx_q[row, col] .+= V_x_full[a, b]
            end
        end

        # Sector overlap N^{(q)}_x = I_{n_q} ⊗ N_x; its inverse is I_{n_q} ⊗ N_x⁻¹
        # Build (I_{n_q} ⊗ N_x⁻¹) · H^{(q)}_x for the generalised eigenvalue problem
        NxInv_Hx_q = zeros(DataType_T, n_q * nx, n_q * nx)
        for i_a in 1:n_q
            row = (i_a - 1) * nx + 1 : i_a * nx
            for i_b in 1:n_q
                col = (i_b - 1) * nx + 1 : i_b * nx
                NxInv_Hx_q[row, col] = Nx_inv * Hx_q[row, col]
            end
        end

        eigen_x = eigen(NxInv_Hx_q)
        Ux_q = eigen_x.vectors             # (n_q n_x) × (n_q n_x)
        dx_q = eigen_x.values              # length n_q n_x
        Ux_q_inv = inv(Ux_q)

        # y-direction: all channels in this sector share the same λ → identical T_y, so
        # one eigendecomposition per sector (could be reused across sectors with the same λ,
        # but we keep one per sector for simplicity).
        a_ref = chans[1]
        eigen_y = eigen(Ny_inv * Ty_channels[a_ref])
        Uy_q = eigen_y.vectors             # n_y × n_y
        dy_q = eigen_y.values              # length n_y
        Uy_q_inv = inv(Uy_q)

        # Precompute Kronecker blocks for fast application
        U_blocks[q] = kron(Ux_q, Uy_q)                                     # (n_q n_x n_y) × (n_q n_x n_y)
        # I_{n_q} ⊗ N_x⁻¹ ⊗ N_y⁻¹ as block-diag of n_q copies of N_inv_xy
        N_inv_block_q = zeros(DataType_T, n_q * nx * ny, n_q * nx * ny)
        for i_a in 1:n_q
            row = (i_a - 1) * nx * ny + 1 : i_a * nx * ny
            N_inv_block_q[row, row] = N_inv_xy
        end
        Ux_Uy_inv = kron(Ux_q_inv, Uy_q_inv)
        U_inv_N_inv_blocks[q] = Ux_Uy_inv * N_inv_block_q
        dx_arrays[q] = dx_q
        dy_arrays[q] = dy_q
    end

    return MInverseCacheVSector{DataType_T}(sector_channels, U_blocks, U_inv_N_inv_blocks,
                                            dx_arrays, dy_arrays, nx, ny, α.nchmax)
end

"""
    M_inverse_operator_cached_vsector(E, cache::MInverseCacheVSector)

Return a function `v -> M(E)⁻¹ * v` using a precomputed V-sector cache.
Only the diagonal D^{(q)}(E)⁻¹ is recomputed for each E.
"""
function M_inverse_operator_cached_vsector(E::Float64, cache::MInverseCacheVSector{T}) where T
    n_sec = length(cache.sector_channels)::Int
    nx = cache.nx::Int
    ny = cache.ny::Int
    block_len = nx * ny

    # Recompute energy-dependent diagonal per sector. Storage order matches
    # kron(Ux_q, Uy_q): outer index μ ∈ 1..n_q*nx, inner index μ_y ∈ 1..ny.
    D_inv_blocks = Vector{Vector{T}}(undef, n_sec)
    @inbounds for q in 1:n_sec
        dx_q = cache.dx_arrays[q]::Vector{T}
        dy_q = cache.dy_arrays[q]::Vector{T}
        len = length(dx_q) * ny
        D_inv = Vector{T}(undef, len)
        idx = 1
        for μ in eachindex(dx_q)
            base = E - dx_q[μ]
            for μ_y in 1:ny
                D_inv[idx] = one(T) / (base - dy_q[μ_y])
                idx += 1
            end
        end
        D_inv_blocks[q] = D_inv
    end

    # Extract typed local references so closure type inference is clean
    sector_channels::Vector{Vector{Int}} = cache.sector_channels
    U_blocks::Vector{Matrix{T}} = cache.U_blocks
    U_inv_N_inv_blocks::Vector{Matrix{T}} = cache.U_inv_N_inv_blocks

    return function(v::AbstractVector)
        # Use eltype(v) for output and buffers so complex Arnoldi vectors flow through
        # without expensive per-element type promotion against the Float64 cache buffers.
        T_v = eltype(v)
        result = similar(v)
        @inbounds for q in 1:n_sec
            chans = sector_channels[q]
            n_q = length(chans)
            len_q = n_q * block_len

            # Gather: pack the sector's channel blocks contiguously
            v_q = Vector{T_v}(undef, len_q)
            for i_a in 1:n_q
                a = chans[i_a]
                src_off = (a - 1) * block_len
                dst_off = (i_a - 1) * block_len
                @simd for k in 1:block_len
                    v_q[dst_off + k] = v[src_off + k]
                end
            end

            t1 = U_inv_N_inv_blocks[q] * v_q
            t2 = D_inv_blocks[q] .* t1
            w_q = U_blocks[q] * t2

            # Scatter back
            for i_a in 1:n_q
                a = chans[i_a]
                src_off = (i_a - 1) * block_len
                dst_off = (a - 1) * block_len
                @simd for k in 1:block_len
                    result[dst_off + k] = w_q[src_off + k]
                end
            end
        end
        return result
    end
end

end # end module matrices