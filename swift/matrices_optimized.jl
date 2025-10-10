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

export T_matrix_optimized

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

OPTIMIZED VERSION of T_matrix using direct block assignment instead of repeated Kronecker products.

## Key Optimizations:
1. Pre-compute overlap matrices Nx and Ny once (not per channel)
2. Compute block Kronecker products (nx*ny × nx*ny) instead of full matrix Kronecker (nα*nx*ny × nα*nx*ny)
3. Direct block assignment to diagonal blocks
4. Reduced memory allocations

## Performance:
- Old: α.nchmax full Kronecker products of size (α.nchmax × α.nchmax) ⊗ (nx × nx) ⊗ (ny × ny)
- New: α.nchmax small Kronecker products of size (nx × nx) ⊗ (ny × ny)
- Expected speedup: 1.5-2× for typical system sizes

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

    # Pre-allocate full matrices
    total_size = nα * nx * ny
    Tx_matrix = zeros(total_size, total_size)
    Ty_matrix = zeros(total_size, total_size)

    # Storage for per-channel components (if requested)
    Tx_channels = Vector{Matrix{Float64}}(undef, nα)
    Ty_channels = Vector{Matrix{Float64}}(undef, nα)

    # OPTIMIZATION: Direct block assignment instead of full Kronecker products
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

        # Direct assignment to diagonal block (avoids full channel Kronecker product)
        idx_start = (iα-1) * nx * ny + 1
        idx_end = iα * nx * ny

        # This is equivalent to: δ_{α,α} I_α ⊗ Tx^α ⊗ Ny
        # but computed directly without building the full channel selector matrix
        Tx_matrix[idx_start:idx_end, idx_start:idx_end] = Tx_block
        Ty_matrix[idx_start:idx_end, idx_start:idx_end] = Ty_block
    end

    Tmatrix = Tx_matrix + Ty_matrix

    if return_components
        return Tmatrix, Tx_channels, Ty_channels, Nx, Ny
    else
        return Tmatrix
    end
end

end # end module
