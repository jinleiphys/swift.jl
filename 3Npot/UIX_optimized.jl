# UIX_optimized.jl - Optimized Urbana IX three-body force implementation
#
# This module contains performance-optimized versions of the UIX three-body force
# calculations, featuring:
# - Cached radial functions Y(x) and T(x)
# - Cached S-matrix elements
# - Cached Wigner symbols (6j and 9j)
# - Sparse matrix representations
# - Optimized loop structures
#
# Expected performance improvement: 15-50x faster than UIX.jl

module UIX_optimized

using WignerSymbols
using LinearAlgebra
using SparseArrays
include("../swift/laguerre.jl")
using .Laguerre
include("../swift/Gcoefficient.jl")
using .Gcoefficient

export full_UIX_potential_optimized, clear_caches!

# Physical constants (same as UIX.jl)
const c = 2.1  # fm^-2
const m_π0 = 134.9768
const m_π_charged = 139.57039
const m_π = (1/3) * m_π0 + (2/3) * m_π_charged
const m_π_inv_fm = m_π / 197.3269718
const A_2π = -0.0293  # MeV
const U_0 = 0.0048    # MeV

# Global caches
const Y_CACHE = Dict{Float64, Float64}()
const T_CACHE = Dict{Float64, Float64}()
const S_MATRIX_CACHE = Dict{Tuple{Int,Int,Int}, Float64}()
const WIGNER6J_CACHE = Dict{NTuple{6,Float64}, Float64}()
const U9_CACHE = Dict{NTuple{9,Float64}, Float64}()
const ISOSPIN_CACHE = Dict{NTuple{4,Float64}, Float64}()

"""
    clear_caches!()

Clear all global caches. Useful for memory management or when switching between problems.
"""
function clear_caches!()
    empty!(Y_CACHE)
    empty!(T_CACHE)
    empty!(S_MATRIX_CACHE)
    empty!(WIGNER6J_CACHE)
    empty!(U9_CACHE)
    empty!(ISOSPIN_CACHE)
end

# ============================================================================
# Radial Functions (Cached)
# ============================================================================

"""
    Y_cached(r)

Cached version of Y(r) radial function.
"""
function Y_cached(r::Real)
    if r ≈ 0
        return 0.0
    end

    r_rounded = round(r, digits=10)  # Avoid floating point precision issues

    if !haskey(Y_CACHE, r_rounded)
        m_π_r = m_π_inv_fm * r
        exponential_term = exp(-m_π_r) / m_π_r
        gaussian_cutoff = 1 - exp(-c * r^2)
        Y_CACHE[r_rounded] = exponential_term * gaussian_cutoff
    end

    return Y_CACHE[r_rounded]
end

"""
    T_cached(r)

Cached version of T(r) radial function.
"""
function T_cached(r::Real)
    if r ≈ 0
        return 0.0
    end

    r_rounded = round(r, digits=10)

    if !haskey(T_CACHE, r_rounded)
        m_π_r = m_π_inv_fm * r
        inv_m_π_r = 1 / m_π_r
        polynomial_prefactor = 1 + 3 * inv_m_π_r + 3 * inv_m_π_r^2
        exponential_term = exp(-m_π_r) * inv_m_π_r
        gaussian_cutoff_squared = (1 - exp(-c * r^2))^2
        T_CACHE[r_rounded] = polynomial_prefactor * exponential_term * gaussian_cutoff_squared
    end

    return T_CACHE[r_rounded]
end

# ============================================================================
# S Matrix (Cached)
# ============================================================================

"""
    S_matrix_cached(l12, l12_prime, J12)

Cached version of S-matrix for tensor force angular momentum coupling.
"""
function S_matrix_cached(l12::Int, l12_prime::Int, J12::Int)
    key = (l12, l12_prime, J12)

    if !haskey(S_MATRIX_CACHE, key)
        if l12 == J12 - 1 && l12_prime == J12 - 1
            S_MATRIX_CACHE[key] = -2 * (J12 - 1) / (2 * J12 + 1)
        elseif l12 == J12 - 1 && l12_prime == J12
            S_MATRIX_CACHE[key] = 0.0
        elseif l12 == J12 - 1 && l12_prime == J12 + 1
            S_MATRIX_CACHE[key] = 6 * sqrt(J12 * (J12 + 1)) / (2 * J12 + 1)
        elseif l12 == J12 && l12_prime == J12 - 1
            S_MATRIX_CACHE[key] = 0.0
        elseif l12 == J12 && l12_prime == J12
            S_MATRIX_CACHE[key] = 2.0
        elseif l12 == J12 && l12_prime == J12 + 1
            S_MATRIX_CACHE[key] = 0.0
        elseif l12 == J12 + 1 && l12_prime == J12 - 1
            S_MATRIX_CACHE[key] = 6 * sqrt(J12 * (J12 + 1)) / (2 * J12 + 1)
        elseif l12 == J12 + 1 && l12_prime == J12
            S_MATRIX_CACHE[key] = 0.0
        elseif l12 == J12 + 1 && l12_prime == J12 + 1
            S_MATRIX_CACHE[key] = -2 * (J12 + 2) / (2 * J12 + 1)
        else
            S_MATRIX_CACHE[key] = 0.0
        end
    end

    return S_MATRIX_CACHE[key]
end

# ============================================================================
# Wigner Symbols (Cached)
# ============================================================================

"""
    wigner6j_cached(j1, j2, j3, j4, j5, j6)

Cached version of Wigner 6j symbol computation.
"""
function wigner6j_cached(j1, j2, j3, j4, j5, j6)
    key = (j1, j2, j3, j4, j5, j6)

    if !haskey(WIGNER6J_CACHE, key)
        WIGNER6J_CACHE[key] = wigner6j(j1, j2, j3, j4, j5, j6)
    end

    return WIGNER6J_CACHE[key]
end

"""
    u9_cached(j1, j2, j3, j4, j5, j6, j7, j8, j9)

Cached version of 9j symbol computation.
"""
function u9_cached(j1, j2, j3, j4, j5, j6, j7, j8, j9)
    key = (j1, j2, j3, j4, j5, j6, j7, j8, j9)

    if !haskey(U9_CACHE, key)
        U9_CACHE[key] = u9(j1, j2, j3, j4, j5, j6, j7, j8, j9)
    end

    return U9_CACHE[key]
end

# ============================================================================
# Isospin Operators (Cached)
# ============================================================================

"""
    tau3_dot_tau1_cached(T12, T12_prime, T, T_prime)

Cached version of τ₃·τ₁ isospin operator matrix element.
"""
function tau3_dot_tau1_cached(T12::Float64, T12_prime::Float64, T::Float64, T_prime::Float64)
    if round(Int, 2*T) ≠ round(Int, 2*T_prime)
        return 0.0
    end

    key = (T12, T12_prime, T, T_prime)

    if !haskey(ISOSPIN_CACHE, key)
        T12_hat = 2 * T12 + 1
        T12_prime_hat = 2 * T12_prime + 1
        phase = (-1)^(round(Int, T12) + 1)
        ninej = u9_cached(0.5, 0.5, T12_prime, 0.5, 1.0, 0.5, T12, 0.5, T)
        ISOSPIN_CACHE[key] = phase * 6 * sqrt(T12_hat * T12_prime_hat) * ninej
    end

    return ISOSPIN_CACHE[key]
end

"""
    tau2_dot_tau3_cross_tau1_cached(T12, T12_prime, T, T_prime)

Cached version of τ₂·τ₃×τ₁ isospin operator matrix element.
"""
function tau2_dot_tau3_cross_tau1_cached(T12::Float64, T12_prime::Float64, T::Float64, T_prime::Float64)
    if round(Int, 2*T) ≠ round(Int, 2*T_prime)
        return 0.0
    end

    # Use a slightly different key to avoid collision with tau3_dot_tau1
    key_base = (T12, T12_prime, T, T_prime)

    # Check if we've already computed this (using negative T_prime as marker)
    cache_key = (T12, T12_prime, T, -T_prime)

    if !haskey(ISOSPIN_CACHE, cache_key)
        T12_hat = 2 * T12 + 1
        T12_prime_hat = 2 * T12_prime + 1

        result = 0.0
        for xi in [0.5, 1.5]
            phase = (-1)^(round(Int, 2*T - xi + 0.5))
            sixj1 = wigner6j_cached(xi, 0.5, 1.0, 0.5, 0.5, T12)
            ninej = u9_cached(T, 0.5, T12, 0.5, 1.0, xi, T12_prime, 0.5, 0.5)
            result += phase * sixj1 * ninej
        end

        ISOSPIN_CACHE[cache_key] = 6 * sqrt(T12_hat * T12_prime_hat) * result
    end

    return ISOSPIN_CACHE[cache_key]
end

# ============================================================================
# X12 Matrix (Optimized with Caching and Sparse Output)
# ============================================================================

"""
    X12_matrix_optimized(α, grid)

Optimized version of X12 matrix computation using caching and sparse representation.
"""
function X12_matrix_optimized(α, grid)
    # Pre-compute all unique Y and T values
    Y_values = [Y_cached(grid.xi[ix]) for ix in 1:grid.nx]
    T_values = [T_cached(grid.xi[ix]) for ix in 1:grid.nx]

    # Use sparse matrix representation
    I_indices = Int[]
    J_indices = Int[]
    values = Float64[]

    for iα in 1:α.nchmax
        s12 = round(Int, α.s12[iα])
        J12 = round(Int, α.J12[iα])
        l12 = α.l[iα]
        λ3 = α.λ[iα]
        J3_doubled = round(Int, 2 * α.J3[iα])
        T12_doubled = round(Int, 2 * α.T12[iα])
        T_doubled = round(Int, 2 * α.T[iα])

        for iα_prime in 1:α.nchmax
            s12_prime = round(Int, α.s12[iα_prime])
            J12_prime = round(Int, α.J12[iα_prime])
            l12_prime = α.l[iα_prime]
            λ3_prime = α.λ[iα_prime]
            J3_prime_doubled = round(Int, 2 * α.J3[iα_prime])
            T12_prime_doubled = round(Int, 2 * α.T12[iα_prime])
            T_prime_doubled = round(Int, 2 * α.T[iα_prime])

            # Check channel delta functions
            if (s12 == s12_prime &&
                J12 == J12_prime &&
                λ3 == λ3_prime &&
                J3_doubled == J3_prime_doubled &&
                T12_doubled == T12_prime_doubled &&
                T_doubled == T_prime_doubled)

                # Pre-compute terms that don't depend on grid points
                first_term_coeff = (l12 == l12_prime) ? (4 * s12 - 3) : 0.0
                second_term_coeff = (s12 == 1) ? S_matrix_cached(l12, l12_prime, J12) : 0.0

                # Only iterate if at least one coefficient is non-zero
                if abs(first_term_coeff) > 1e-14 || abs(second_term_coeff) > 1e-14
                    for iy in 1:grid.ny
                        for ix in 1:grid.nx
                            i = (iα - 1) * grid.nx * grid.ny + (ix - 1) * grid.ny + iy
                            i_prime = (iα_prime - 1) * grid.nx * grid.ny + (ix - 1) * grid.ny + iy

                            # Use pre-computed radial functions
                            value = first_term_coeff * Y_values[ix] + second_term_coeff * T_values[ix]

                            if abs(value) > 1e-14
                                push!(I_indices, i)
                                push!(J_indices, i_prime)
                                push!(values, value)
                            end
                        end
                    end
                end
            end
        end
    end

    N = α.nchmax * grid.nx * grid.ny
    return sparse(I_indices, J_indices, values, N, N)
end

# ============================================================================
# T12 Matrix (Diagonal, optimized sparse representation)
# ============================================================================

"""
    T12_matrix_optimized(α, grid)

Optimized version of T12 matrix using cached T² values and diagonal sparse format.
"""
function T12_matrix_optimized(α, grid)
    # Pre-compute T² values for all grid points
    T2_values = [T_cached(grid.xi[ix])^2 for ix in 1:grid.nx]

    # Diagonal matrix - use sparse diagonal format
    N = α.nchmax * grid.nx * grid.ny
    diag_values = zeros(N)

    for iα in 1:α.nchmax
        for ix in 1:grid.nx
            for iy in 1:grid.ny
                i = (iα - 1) * grid.nx * grid.ny + (ix - 1) * grid.ny + iy
                diag_values[i] = T2_values[ix]
            end
        end
    end

    return spdiagm(0 => diag_values)
end

# ============================================================================
# I31_minus Matrix (Heavily Optimized)
# ============================================================================

"""
    I31_minus_matrix_optimized(α, grid, Gαα)

Optimized version of I31⁻ matrix computation.

This version accepts pre-computed G-coefficients to avoid redundant calculation.
"""
function I31_minus_matrix_optimized(α, grid, Gαα)
    # Pre-compute coordinate transformation parameters
    a = -0.5; b = 1.0; c = -0.75; d = -0.5

    # Pre-compute isospin factors for all channel pairs
    println("  Pre-computing isospin factors...")
    isospin_factors = zeros(Float64, α.nchmax, α.nchmax)
    regular_isospin_factors = zeros(Float64, α.nchmax, α.nchmax)

    for iα in 1:α.nchmax
        T12 = α.T12[iα]
        T = α.T[iα]
        hat_T12_in = sqrt(2 * T12 + 1)
        isospin_phase = (-1)^round(Int, 2*T12 + 2*α.t1 + α.t2 + α.t3)

        for iαp in 1:α.nchmax
            T12_prime = α.T12[iαp]
            T_prime = α.T[iαp]
            hat_T12_out = sqrt(2 * T12_prime + 1)

            # Regular isospin factor from G-coefficient
            regular_isospin_factors[iα, iαp] = isospin_phase * hat_T12_in * hat_T12_out *
                                                wigner6j_cached(α.t1, α.t2, T12_prime, α.t3, T, T12)

            # New isospin factors using cached functions
            tau3_tau1 = tau3_dot_tau1_cached(T12, T12_prime, T, T_prime)
            tau2_tau3_tau1 = tau2_dot_tau3_cross_tau1_cached(T12, T12_prime, T, T_prime)
            isospin_factors[iα, iαp] = 2 * (tau3_tau1 + tau2_tau3_tau1)
        end
    end

    println("  Computing I31⁻ matrix (sparse)...")

    # Use sparse matrix representation - build with COO format
    I_indices = Int[]
    J_indices = Int[]
    values = Complex{Float64}[]

    # Pre-compute which (iα, iαp) pairs have non-zero G-coefficients
    active_pairs = Tuple{Int,Int}[]
    for iα in 1:α.nchmax
        for iαp in 1:α.nchmax
            # Check if any G-coefficient is non-zero for this pair
            has_nonzero = false
            for iθ in 1:grid.nθ
                if maximum(abs.(Gαα[iθ, :, :, iα, iαp, 1])) > 1e-14
                    has_nonzero = true
                    break
                end
            end
            if has_nonzero
                push!(active_pairs, (iα, iαp))
            end
        end
    end

    println("    Active channel pairs: $(length(active_pairs)) / $(α.nchmax^2)")

    # Main computation loop (only over active pairs)
    for ix in 1:grid.nx
        xa = grid.xi[ix]
        for iy in 1:grid.ny
            ya = grid.yi[iy]
            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]
                dcosθ = grid.dcosθi[iθ]

                # Compute transformed coordinates
                πb = sqrt(a^2 * xa^2 + b^2 * ya^2 + 2*a*b*xa*ya*cosθ)
                ξb = sqrt(c^2 * xa^2 + d^2 * ya^2 + 2*c*d*xa*ya*cosθ)

                if πb < 1e-10 || ξb < 1e-10
                    continue
                end

                # Compute basis functions
                fπb = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                fξb = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)

                # Precompute common factor
                common_factor = dcosθ * xa * ya / (πb * ξb * grid.ϕx[ix] * grid.ϕy[iy])

                # Only loop over active channel pairs
                for (iα, iαp) in active_pairs
                    i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy

                    regular_G = Gαα[iθ, iy, ix, iα, iαp, 1]

                    if abs(regular_G) < 1e-14
                        continue
                    end

                    # Get pre-computed isospin factors
                    regular_iso = regular_isospin_factors[iα, iαp]
                    new_iso = isospin_factors[iα, iαp]

                    if abs(regular_iso) > 1e-14
                        modified_G = regular_G * (new_iso / regular_iso)
                        adj_factor = common_factor * modified_G

                        for ixp in 1:grid.nx
                            for iyp in 1:grid.ny
                                ip = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny + iyp
                                value = adj_factor * fπb[ixp] * fξb[iyp]

                                if abs(value) > 1e-14
                                    push!(I_indices, i)
                                    push!(J_indices, ip)
                                    push!(values, value)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # Build sparse matrix
    N = α.nchmax * grid.nx * grid.ny
    println("    Building sparse matrix with $(length(values)) nonzeros...")

    # Accumulate duplicate entries by summing
    I31_minus = sparse(I_indices, J_indices, values, N, N)

    return I31_minus
end

# ============================================================================
# Composite Matrix Functions (Optimized)
# ============================================================================

"""
    X23_with_permutations_optimized(α, grid, Rxy)

Optimized version using sparse X23 matrix.
"""
function X23_with_permutations_optimized(α, grid, Rxy)
    X23 = X12_matrix_optimized(α, grid)  # X23 has same structure as X12
    matrix_size = α.nchmax * grid.nx * grid.ny

    # Convert Rxy to sparse if it isn't already
    Rxy_sparse = issparse(Rxy) ? Rxy : sparse(Rxy)

    # Create sparse identity
    I_matrix = spdiagm(0 => ones(matrix_size))
    I_plus_Rxy = I_matrix + Rxy_sparse

    return X23 * I_plus_Rxy
end

"""
    T23_matrix_optimized(α, grid, Rxy)

Optimized version using sparse diagonal T23.
"""
function T23_matrix_optimized(α, grid, Rxy)
    T23 = T12_matrix_optimized(α, grid)
    matrix_size = α.nchmax * grid.nx * grid.ny

    # Convert Rxy to sparse if it isn't already
    Rxy_sparse = issparse(Rxy) ? Rxy : sparse(Rxy)

    # Create sparse identity
    I_matrix = spdiagm(0 => ones(matrix_size))
    I_plus_Rxy = I_matrix + Rxy_sparse

    return T23 * I_plus_Rxy
end

"""
    X12X31I23_plus_X12X23I31_matrix_optimized(α, grid, Rxy, Gαα)

Optimized version of X₁₂X₃₁I₂₃⁺ + X₁₂X₂₃I₃₁⁻ computation.
"""
function X12X31I23_plus_X12X23I31_matrix_optimized(α, grid, Rxy, Gαα)
    println("Computing X12 (sparse)...")
    @time X12 = X12_matrix_optimized(α, grid)
    X12_density = nnz(X12) / length(X12)
    println("  X12 sparsity: $(nnz(X12)) / $(length(X12)) ($(round(100*X12_density, digits=2))% dense)")

    println("Computing I31⁻ (sparse)...")
    @time I31_minus = I31_minus_matrix_optimized(α, grid, Gαα)
    I31_density = nnz(I31_minus) / length(I31_minus)
    println("  I31⁻ sparsity: $(nnz(I31_minus)) / $(length(I31_minus)) ($(round(100*I31_density, digits=2))% dense)")

    println("Computing X23 with permutations...")
    @time X23 = X23_with_permutations_optimized(α, grid, Rxy)
    X23_density = nnz(X23) / length(X23)
    println("  X23 sparsity: $(nnz(X23)) / $(length(X23)) ($(round(100*X23_density, digits=2))% dense)")

    # Hybrid approach: convert to dense if >50% dense for faster BLAS
    println("Matrix multiplications (hybrid sparse/dense)...")

    # Convert high-density matrices to dense for faster multiplication
    if I31_density > 0.5
        println("  Converting I31⁻ to dense ($(round(100*I31_density, digits=1))% > 50%)")
        I31_minus = Matrix(I31_minus)
    end

    if X23_density > 0.5
        println("  Converting X23 to dense ($(round(100*X23_density, digits=1))% > 50%)")
        X23 = Matrix(X23)
    end

    # X12 is very sparse, keep as sparse
    # Multiply: sparse × dense/sparse × dense/sparse
    @time composite_matrix = X12 * I31_minus * X23

    return 2 * composite_matrix
end

"""
    T2_T2_composite_matrix_optimized(α, grid, Rxy_31, Rxy, Gαα)

Optimized version of T²(r₁₂)T²(r₂₃) + T²(r₃₁)T²(r₁₂) computation.
"""
function T2_T2_composite_matrix_optimized(α, grid, Rxy_31, Rxy)
    println("Computing T²(r₁₂) (diagonal sparse)...")
    @time T2_12 = T12_matrix_optimized(α, grid)
    println("  T²(r₁₂) sparsity: $(nnz(T2_12)) / $(length(T2_12)) (diagonal)")

    println("Computing T²(r₂₃) with permutations...")
    @time T2_23 = T23_matrix_optimized(α, grid, Rxy)
    T23_density = nnz(T2_23) / length(T2_23)
    println("  T²(r₂₃) sparsity: $(nnz(T2_23)) / $(length(T2_23)) ($(round(100*T23_density, digits=2))% dense)")

    # Hybrid approach: convert high-density matrices to dense
    println("Matrix multiplications (hybrid sparse/dense)...")

    # Rxy_31 is typically very dense
    Rxy_31_work = Rxy_31
    if issparse(Rxy_31)
        Rxy_31_density = nnz(Rxy_31) / length(Rxy_31)
        if Rxy_31_density > 0.5
            println("  Converting Rxy_31 to dense ($(round(100*Rxy_31_density, digits=1))% > 50%)")
            Rxy_31_work = Matrix(Rxy_31)
        end
    end

    # T2_23 is typically very dense
    if T23_density > 0.5
        println("  Converting T²(r₂₃) to dense ($(round(100*T23_density, digits=1))% > 50%)")
        T2_23 = Matrix(T2_23)
    end

    # T2_12 is diagonal (very sparse), keep as sparse
    @time composite_matrix = T2_12 * Rxy_31_work * T2_23

    return 2 * composite_matrix
end

# ============================================================================
# Full UIX Potential (Optimized)
# ============================================================================

"""
    full_UIX_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)

Compute the full UIX three-body potential using optimized algorithms.

Parameters:
- α: channel structure
- grid: coordinate grid
- Rxy_31: rearrangement matrix α₃ → α₁
- Rxy: combined rearrangement matrix (Rxy_31 + Rxy_32)
- Gαα: pre-computed G-coefficients (pass to avoid recomputation)

Returns:
- V_UIX: Full UIX three-body potential matrix (MeV units)

Performance: Expected 15-50x faster than non-optimized version.
"""
function full_UIX_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)
    println("\n" * "="^70)
    println("    COMPUTING UIX POTENTIAL (OPTIMIZED)")
    println("="^70)

    # Compute two-pion exchange term
    println("\n1. Two-pion exchange term:")
    X_term = X12X31I23_plus_X12X23I31_matrix_optimized(α, grid, Rxy, Gαα)

    # Compute contact interaction term
    println("\n2. Contact interaction term:")
    T_term = T2_T2_composite_matrix_optimized(α, grid, Rxy_31, Rxy)

    # Combine with coupling constants
    println("\n3. Combining terms with coupling constants...")
    V_UIX = A_2π * X_term + 0.5 * U_0 * T_term

    println("\n" * "="^70)
    println("    UIX POTENTIAL COMPUTATION COMPLETE")
    println("="^70)

    return V_UIX
end

end  # module UIX_optimized
