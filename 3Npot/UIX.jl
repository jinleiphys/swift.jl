# UIX.jl - Urbana IX three-body force model
# Implementation of Y(r) and T(r) functions for three-body nuclear forces
#
# NOTE: These functions are implemented in the Lagrange function angular momentum basis,
# NOT in the Jacobi coordinate angular momentum basis. This basis choice affects the
# angular momentum coupling and coordinate system used in three-body calculations.

module UIX

using WignerSymbols
using LinearAlgebra
include("../swift/laguerre.jl")
using .Laguerre
include("../swift/Gcoefficient.jl")
using .Gcoefficient

export full_UIX_potential

# Physical constants for Urbana IX model
const c = 2.1  # fm^-2, Gaussian cutoff parameter    

# Pion masses in MeV/c^2 (PDG values)
const m_π0 = 134.9768 # neutral pion mass
const m_π_charged = 139.57039  # charged pion mass (π± average)

# Average pion mass: m_π = (1/3)m_π0 + (2/3)m_π±
const m_π = (1/3) * m_π0 + (2/3) * m_π_charged  # MeV/c^2

# Convert to fm^-1 (ħc = 197.3269718 MeV·fm)
const m_π_inv_fm = m_π / 197.3269718  # fm^-1

# UIX three-body force coupling constants
const A_2π = -0.0293  # MeV, two-pion exchange coupling
const U_0 = 0.0048    # MeV, contact interaction coupling

"""
    Y(r)

Urbana IX three-body force function Y(r).

Y(r) = (e^(-m_π r))/(m_π r) * (1 - e^(-c r²))

Where:
- r: distance in fm
- m_π: average pion mass in fm^-1
- c: Gaussian cutoff parameter = 2.1 fm^-2

Returns the Y(r) value (dimensionless).
"""
function Y(r::Real)
    if r ≈ 0
        return 0.0  # Y(0) = 0 due to the r factor in denominator and (1 - e^(-c r²)) → 0
    end

    m_π_r = m_π_inv_fm * r
    exponential_term = exp(-m_π_r) / m_π_r
    gaussian_cutoff = 1 - exp(-c * r^2)

    return exponential_term * gaussian_cutoff
end

"""
    T(r)

Urbana IX three-body force function T(r).

T(r) = [1 + 3/(m_π r) + 3/(m_π r)²] * (e^(-m_π r))/(m_π r) * (1 - e^(-c r²))²

Where:
- r: distance in fm
- m_π: average pion mass in fm^-1
- c: Gaussian cutoff parameter = 2.1 fm^-2

Returns the T(r) value (units: fm^-1).
"""
function T(r::Real)
    if r ≈ 0
        return 0.0  # T(0) = 0 due to similar reasoning as Y(r)
    end

    m_π_r = m_π_inv_fm * r
    inv_m_π_r = 1 / m_π_r

    # Polynomial prefactor: [1 + 3/(m_π r) + 3/(m_π r)²]
    polynomial_prefactor = 1 + 3 * inv_m_π_r + 3 * inv_m_π_r^2

    # Exponential term: e^(-m_π r) / (m_π r)
    exponential_term = exp(-m_π_r) * inv_m_π_r

    # Squared Gaussian cutoff: (1 - e^(-c r²))²
    gaussian_cutoff_squared = (1 - exp(-c * r^2))^2

    return polynomial_prefactor * exponential_term * gaussian_cutoff_squared
end

"""
    S_matrix(l12, l12_prime, J12)

Compute the S matrix elements for angular momentum coupling in Urbana IX three-body force.

The S matrix is defined as:
S_{l₁₂,l'₁₂,J₁₂} for transitions between different l₁₂ values with fixed J₁₂.

Returns the matrix element S_{l₁₂,l'₁₂,J₁₂}.
"""
function S_matrix(l12::Int, l12_prime::Int, J12::Int)
    # S matrix elements according to the given table
    if l12 == J12 - 1 && l12_prime == J12 - 1
        return -2 * (J12 - 1) / (2 * J12 + 1)
    elseif l12 == J12 - 1 && l12_prime == J12
        return 0.0
    elseif l12 == J12 - 1 && l12_prime == J12 + 1
        return 6 * sqrt(J12 * (J12 + 1)) / (2 * J12 + 1)
    elseif l12 == J12 && l12_prime == J12 - 1
        return 0.0
    elseif l12 == J12 && l12_prime == J12
        return 2.0
    elseif l12 == J12 && l12_prime == J12 + 1
        return 0.0
    elseif l12 == J12 + 1 && l12_prime == J12 - 1
        return 6 * sqrt(J12 * (J12 + 1)) / (2 * J12 + 1)
    elseif l12 == J12 + 1 && l12_prime == J12
        return 0.0
    elseif l12 == J12 + 1 && l12_prime == J12 + 1
        return -2 * (J12 + 2) / (2 * J12 + 1)
    else
        return 0.0
    end
end

"""
    X12_matrix(α, grid)

Compute the X12 matrix for Urbana IX three-body force.

The matrix elements are:
⟨f_{k_x} f_{k_y} α₃ | X₁₂ | f_{k_x'} f_{k_y'} α₃' ⟩ =
δ_{k_y,k_y'} δ_{s₁₂,s₁₂'} δ_{J₁₂,J₁₂'} δ_{λ₃,λ₃'} δ_{J₃,J₃'} δ_{J,J'} δ_{T₁₂,T₁₂'} δ_{T,T'} ×
[δ_{l₁₂,l₁₂'}(4s₁₂-3)Y(x_{k_x}) + δ_{s₁₂,1}T(x_{k_x})S_{l₁₂,l₁₂',J₁₂}]

Returns the X12 matrix with the same indexing as V and T matrices:
i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy
"""
function X12_matrix(α, grid)
    # Initialize X12 matrix with same dimensions as V and T matrices
    X12 = zeros(α.nchmax * grid.nx * grid.ny, α.nchmax * grid.nx * grid.ny)

    # Loop over channels first to check delta functions early
    for iα in 1:α.nchmax
        # Extract quantum numbers for channel iα (computed once per iα)
        # For integer quantum numbers
        s12 = round(Int, α.s12[iα])
        J12 = round(Int, α.J12[iα])
        l12 = α.l[iα]
        λ3 = α.λ[iα]
        # For half-integer quantum numbers, double them for exact integer comparison
        J3_doubled = round(Int, 2 * α.J3[iα])
        T12_doubled = round(Int, 2 * α.T12[iα])
        T_doubled = round(Int, 2 * α.T[iα])

        for iα_prime in 1:α.nchmax
            # Extract quantum numbers for channel iα_prime
            s12_prime = round(Int, α.s12[iα_prime])
            J12_prime = round(Int, α.J12[iα_prime])
            l12_prime = α.l[iα_prime]
            λ3_prime = α.λ[iα_prime]
            J3_prime_doubled = round(Int, 2 * α.J3[iα_prime])
            T12_prime_doubled = round(Int, 2 * α.T12[iα_prime])
            T_prime_doubled = round(Int, 2 * α.T[iα_prime])

            # Check channel delta functions first (most restrictive)
            if (s12 == s12_prime &&                    # δ_{s₁₂,s₁₂'}
                J12 == J12_prime &&                    # δ_{J₁₂,J₁₂'}
                λ3 == λ3_prime &&                      # δ_{λ₃,λ₃'}
                J3_doubled == J3_prime_doubled &&      # δ_{J₃,J₃'}
                T12_doubled == T12_prime_doubled &&    # δ_{T₁₂,T₁₂'}
                T_doubled == T_prime_doubled)          # δ_{T,T'}

                # Only proceed if channel quantum numbers match
                for iy in 1:grid.ny
                    # δ_{k_y,k_y'} means iy_prime = iy, so eliminate the inner iy_prime loop
                    iy_prime = iy

                    for ix in 1:grid.nx
                        # δ_{k_x,k_x'} means ix_prime = ix, so eliminate the inner ix_prime loop
                        ix_prime = ix

                        # Compute indices once per (ix, iy) combination
                        i = (iα - 1) * grid.nx * grid.ny + (ix - 1) * grid.ny + iy
                        i_prime = (iα_prime - 1) * grid.nx * grid.ny + (ix_prime - 1) * grid.ny + iy_prime

                        # Get x coordinate for the radial function
                        x_kx = grid.xi[ix]

                        # First term: δ_{l₁₂,l₁₂'}(4s₁₂-3)Y(x_{k_x})
                        first_term = 0.0
                        if l12 == l12_prime
                            first_term = (4 * s12 - 3) * Y(x_kx)
                        end

                        # Second term: δ_{s₁₂,1}T(x_{k_x})S_{l₁₂,l₁₂',J₁₂}
                        second_term = 0.0
                        if s12 == 1
                            second_term = T(x_kx) * S_matrix(l12, l12_prime, J12)
                        end

                        X12[i, i_prime] = first_term + second_term
                    end
                end
            end
        end
    end

    return X12
end

"""
    tau3_dot_tau1(T12, T12_prime, T)

Compute the isospin matrix element ⟨((t₁t₂)T₁₂ t₃)T | τ₃·τ₁ |((t₂t₃)T₁₂' t₁)T'⟩

Returns the matrix element using 6j symbols according to:
δ_{T,T'}(-1)^{T₁₂+1} × 6√(T̂₁₂T̂₁₂') × {1/2  1/2  T₁₂' ; 1/2  1  1/2 ; T₁₂  1/2  T}

Where T̂ = 2T + 1 (dimension factor).
"""
function tau3_dot_tau1(T12::Float64, T12_prime::Float64, T::Float64, T_prime::Float64)
    # Check if T == T' (delta function) - use integer comparison for exact match
    if round(Int, 2*T) ≠ round(Int, 2*T_prime)
        return 0.0
    end

    # Compute dimension factors: T̂ = 2T + 1
    T12_hat = 2 * T12 + 1
    T12_prime_hat = 2 * T12_prime + 1

    # Phase factor: (-1)^{T₁₂+1}
    phase = (-1)^(round(Int, T12) + 1)

    # 9j symbol: {1/2  1/2  T₁₂' ; 1/2  1  1/2 ; T₁₂  1/2  T}
    ninej = u9(0.5, 0.5, T12_prime,
               0.5, 1.0, 0.5,
               T12, 0.5, T)

    return phase * 6 * sqrt(T12_hat * T12_prime_hat) * ninej
end

"""
    tau2_dot_tau3_cross_tau1(T12, T12_prime, T)

Compute the isospin matrix element -i/4⟨((t₁t₂)T₁₂ t₃)T | τ₂·τ₃×τ₁ |((t₂t₃)T₁₂' t₁)T'⟩

Returns the matrix element using 6j symbols according to:
δ_{T,T'} × 6√(T̂₁₂T̂₁₂') × Σ_ξ∈{1/2,3/2} (-1)^{2T-ξ+1/2} ×
{ξ  1/2  1 ; 1/2  1/2  T₁₂} × {T  1/2  T₁₂ ; 1/2  1  ξ ; T₁₂'  1/2  1/2}
"""
function tau2_dot_tau3_cross_tau1(T12::Float64, T12_prime::Float64, T::Float64, T_prime::Float64)
    # Check if T == T' (delta function) - use integer comparison for exact match
    if round(Int, 2*T) ≠ round(Int, 2*T_prime)
        return 0.0
    end

    # Compute dimension factors: T̂ = 2T + 1
    T12_hat = 2 * T12 + 1
    T12_prime_hat = 2 * T12_prime + 1

    # Sum over ξ ∈ {1/2, 3/2} as specified in UIX reference
    result = 0.0
    for xi in [0.5, 1.5]
        # Phase factor: (-1)^{2T-ξ+1/2}
        phase = (-1)^(round(Int, 2*T - xi + 0.5))

        # First 6j symbol: {ξ  1/2  1 ; 1/2  1/2  T₁₂}
        sixj1 = wigner6j(xi, 0.5, 1.0,
                         0.5, 0.5, T12)

        # Second 9j symbol: {T  1/2  T₁₂ ; 1/2  1  ξ ; T₁₂'  1/2  1/2}
        ninej = u9(T, 0.5, T12,
                   0.5, 1.0, xi,
                   T12_prime, 0.5, 0.5)

        result += phase * sixj1 * ninej
    end

    return 6 * sqrt(T12_hat * T12_prime_hat) * result
end

"""
    I31_minus_matrix(α, grid)

Compute the I₃₁⁻ matrix for Urbana IX three-body force.

I₃₁⁻ = 2(τ₃·τ₁ - i/4 τ₂·τ₃×τ₁)

The matrix elements are computed similarly to Rxy_31 but with modified G-coefficients
where only the isospin part is replaced by the new isospin operators.
The spatial (spherical harmonics) and spin parts remain the same as the regular G-coefficient.

Returns the I31⁻ matrix with the same indexing as other matrices:
i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy
"""
function I31_minus_matrix(α, grid)
    # Initialize I31⁻ matrix
    I31_minus = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)

    # Compute the regular G-coefficient to get spatial and spin parts
    Gαα = computeGcoefficient(α, grid)

    # Coordinate transformation parameters (same as Rxy_31)
    a = -0.5; b = 1.0; c = -0.75; d = -0.5

    # Loop over coordinate grids
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

                # Compute basis functions at transformed coordinates
                fπb = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                fξb = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)

                # Loop over channel combinations
                for iα in 1:α.nchmax
                    i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy

                    for iαp in 1:α.nchmax
                        # Get the regular G-coefficient (contains spatial + spin + isospin)
                        regular_G = Gαα[iθ, iy, ix, iα, iαp, 1]  # permutation index 1 for α1→α3

                        # Skip if regular G-coefficient is zero
                        if abs(regular_G) < 1e-14
                            continue
                        end

                        # Extract isospin quantum numbers for channels
                        T12 = α.T12[iα]
                        T12_prime = α.T12[iαp]
                        T = α.T[iα]
                        T_prime = α.T[iαp]

                        # Compute the regular isospin part that was used in G-coefficient
                        # For Rxy_31 (α1→α3): Cisospin = hat(T12_in) * hat(T12_out) * wigner6j(t1,t2,T12_out,t3,T,T12_in)
                        # where hat(x) = √(2x+1), and includes isospin phase factor
                        # Phase must match Gcoefficient.jl line 91: (-1)^(T12_in + 2*t1 + t2 + t3)
                        # T12_prime is in the ket (incoming channel), T12 is in the bra (outgoing channel)
                        hat_T12_in = sqrt(2 * T12 + 1)
                        hat_T12_out = sqrt(2 * T12_prime + 1)
                        isospin_phase = (-1)^round(Int, T12_prime + 2*α.t1 + α.t2 + α.t3)
                        regular_isospin = isospin_phase * hat_T12_in * hat_T12_out * wigner6j(α.t1, α.t2, T12_prime, α.t3, T, T12)

                        # Compute new isospin matrix elements for I31⁻
                        tau3_tau1_element = tau3_dot_tau1(T12, T12_prime, T, T_prime)
                        tau2_tau3_tau1_element = tau2_dot_tau3_cross_tau1(T12, T12_prime, T, T_prime)

                        # Combined new isospin factor: 2(τ₃·τ₁ + τ₂·τ₃×τ₁)
                        # Note: tau2_tau3_tau1_element already includes the -i/4 factor
                        new_isospin_factor = 2 * (tau3_tau1_element + tau2_tau3_tau1_element)

                        # Replace isospin part: G_new = G_regular * (new_isospin / old_isospin)
                        if abs(regular_isospin) > 1e-14
                            modified_G = regular_G * (new_isospin_factor / regular_isospin)
                        else
                            modified_G = 0.0
                        end

                        # Apply the same transformation as Rxy_31
                        adj_factor = dcosθ * modified_G * xa * ya / (πb * ξb * grid.ϕx[ix] * grid.ϕy[iy])

                        for ixp in 1:grid.nx
                            for iyp in 1:grid.ny
                                ip = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny + iyp
                                I31_minus[i, ip] += adj_factor * fπb[ixp] * fξb[iyp]
                            end
                        end
                    end
                end
            end
        end
    end

    return I31_minus
end

"""
    X23_with_permutations(α, grid, Rxy)

Compute X23 × (1 + P⁺ + P⁻) for Urbana IX three-body force.

This function computes X23 matrix (equivalent to X12 but for particles 2-3)
and multiplies it by (I + Rxy) where:
- I is the identity matrix
- Rxy = Rxy_31 + Rxy_32 (rearrangement matrices from permutation operators)

The result represents the full three-body force contribution X23(1 + P⁺ + P⁻).

Parameters:
- α: channel structure
- grid: coordinate grid
- Rxy: rearrangement matrix Rxy = Rxy_31 + Rxy_32

Returns:
- X23_full: Matrix representing X23 × (I + Rxy)
"""
function X23_with_permutations(α, grid, Rxy)
    # Compute X23 matrix (same structure as X12, but for particles 2-3)
    # For UIX model, X23 has the same functional form as X12
    X23 = X12_matrix(α, grid)  # X23 has same structure as X12

    # Create identity matrix of same size
    matrix_size = α.nchmax * grid.nx * grid.ny
    I_matrix = Matrix{Float64}(I, matrix_size, matrix_size)

    # Compute (I + Rxy)
    I_plus_Rxy = I_matrix + Rxy

    # Compute X23 × (I + Rxy)
    X23_full = X23 * I_plus_Rxy

    return X23_full
end

"""
    X12X31I23_plus_X12X23I31_matrix(α, grid)

Compute the composite UIX three-body force matrix element:
X₁₂X₃₁I₂₃⁺ + X₁₂X₂₃I₃₁⁻

Based on the mathematical formula:
⟨f_{k_x}f_{k_y} α₃ |X₁₂X₃₁I₂₃⁺ + X₁₂X₂₃I₃₁⁻ |Ψ⟩ =
(1+ε_{α₃'}ε_{α₁}) ∑_{k_x'k_y'α₃'} ⟨f_{k_x} f_{k_y} α₃ | X₁₂ | f_{k_x'} f_{k_y'} α₃'⟩
∑_{i_x i_yα₁} ⟨f_{k'_x}f_{k'_y}α₃'| I₃₁⁻|f_{i_x}f_{i_y}α₁⟩ ⟨f_{i_x}f_{i_y}α₁ |X₂₃ | Ψ⟩

This represents a three-step process:
1. X₁₂ acts on the initial state
2. I₃₁⁻ performs coordinate transformation with modified isospin coupling
3. X₂₃ acts on the final transformed state
4. Includes permutation symmetry factor (1+ε_{α₃'}ε_{α₁})

The matrix multiplication order follows the operator sequence: X₁₂ × I₃₁⁻ × X₂₃

Parameters:
- α: channel structure containing quantum numbers
- grid: coordinate grid for spatial integration

Returns:
- Composite matrix representing the full X₁₂X₃₁I₂₃⁺ + X₁₂X₂₃I₃₁⁻ operator
"""
function X12X31I23_plus_X12X23I31_matrix(α, grid, Rxy)
    # Compute individual matrix components
    X12 = X12_matrix(α, grid)
    I31_minus = I31_minus_matrix(α, grid)
    X23 = X23_with_permutations(α, grid, Rxy)  # Use the proper X23 function with permutations

    # Compute the composite matrix: X12 × I31⁻ × X23
    # Note: Matrix multiplication order follows operator sequence
    composite_matrix = X12 * I31_minus * X23

    # Apply permutation symmetry factor (1+ε_{α₃'}ε_{α₁}) = 2
    # This factor of 2 accounts for both X₁₂X₃₁I₂₃⁺ and X₁₂X₂₃I₃₁⁻ terms
    final_matrix = 2 * composite_matrix

    return final_matrix
end

"""
    T12_matrix(α, grid)

Compute the T12 matrix for UIX three-body force with delta function constraints.

The matrix elements are:
⟨f_{k_x} f_{k_y} α | T₁₂ | f_{k_x'} f_{k_y'} α'⟩ =
δ_{α,α'} δ_{k_x,k_x'} δ_{k_y,k_y'} T²(x_{k_x})

Where:
- δ_{α,α'}: Channel delta function (all quantum numbers must match)
- δ_{k_x,k_x'}: x-coordinate grid point delta function
- δ_{k_y,k_y'}: y-coordinate grid point delta function
- T²(x_{k_x}): UIX T function squared evaluated at x coordinate

Returns the T12 matrix with the same indexing as other matrices:
i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy
"""
function T12_matrix(α, grid)
    # Initialize T12 matrix with same dimensions as other matrices
    T12 = zeros(α.nchmax * grid.nx * grid.ny, α.nchmax * grid.nx * grid.ny)

    # Loop over all matrix elements - all delta functions mean i = i_prime (diagonal matrix)
    for iα in 1:α.nchmax
        for ix in 1:grid.nx
            for iy in 1:grid.ny
                # Compute matrix index (same for both i and i_prime due to delta functions)
                i = (iα - 1) * grid.nx * grid.ny + (ix - 1) * grid.ny + iy
                # Get x coordinate for T function evaluation
                x_kx = grid.xi[ix]
                # Apply T² function: T12[i, i] = T²(x_{k_x}) (diagonal element)
                T12[i, i] = T(x_kx)^2
            end
        end
    end

    return T12
end

"""
    T23_matrix(α, grid, Rxy)

Compute T23 matrix with permutations: T23(1 + P⁺ + P⁻).

Since T23 = T12 (same functional form), this function computes:
T23_with_permutations = T23 × (I + Rxy)

where:
- T23 has the same structure as T12 (diagonal matrix with T(x) function)
- I is the identity matrix
- Rxy = Rxy_31 + Rxy_32 (rearrangement matrices from permutation operators P⁺ + P⁻)

The result represents the full T23 contribution with particle permutations:
T23(1 + P⁺ + P⁻) = T23 × (I + Rxy)

Parameters:
- α: channel structure containing quantum numbers
- grid: coordinate grid for spatial integration
- Rxy: rearrangement matrix (Rxy_31 + Rxy_32) representing permutation operators

Returns:
- T23_with_permutations: Matrix representing T23 × (I + Rxy)
"""
function T23_matrix(α, grid, Rxy)
    # T23 has the same structure as T12 (diagonal matrix with T function)
    T23 = T12_matrix(α, grid)

    # Create identity matrix of same size
    matrix_size = α.nchmax * grid.nx * grid.ny
    I_matrix = Matrix{Float64}(I, matrix_size, matrix_size)

    # Compute (I + Rxy) representing (1 + P⁺ + P⁻)
    I_plus_Rxy = I_matrix + Rxy

    # Compute T23 × (I + Rxy)
    T23_with_permutations = T23 * I_plus_Rxy

    return T23_with_permutations
end

"""
    T2_T2_composite_matrix(α, grid, Rxy_31)

Compute the composite matrix for T²(r₁₂)T²(r₂₃) + T²(r₃₁)T²(r₁₂).

Based on the mathematical formula:
⟨f_{k_x}f_{k_y} α₃ | T²(r₁₂)T²(r₂₃) + T²(r₃₁)T²(r₁₂) | Ψ⟩ =
(1+ε_{α₃'}ε_{α₁}) ∑_{k_x'k_y'α₃'} ∑_{i_x i_y α₁}
⟨f_{k_x}f_{k_y} α₃ | T²(r₁₂) | f_{k'_x}f_{k_y'}α₃'⟩ ×
⟨f_{k'_x}f_{k_y'}α₃' | f_{i_x}f_{i_y}α₁⟩ ⟨f_{i_x}f_{i_y}α₁ | T²(r₂₃) | Ψ⟩

Where:
- ⟨f_{k'_x}f_{k_y'}α₃' | f_{i_x}f_{i_y}α₁⟩ is Rxy_31
- (1+ε_{α₃'}ε_{α₁}) = 2 (symmetry factor)

This computes: 2 × T²(r₁₂) × Rxy_31 × T²(r₂₃)

Parameters:
- α: channel structure containing quantum numbers
- grid: coordinate grid for spatial integration
- Rxy_31: rearrangement matrix for coordinate transformation
- Rxy: rearrangement matrix (Rxy_31 + Rxy_32) for T23 permutations

Returns:
- Composite matrix representing 2 × T²(r₁₂) × Rxy_31 × T²(r₂₃)
"""
function T2_T2_composite_matrix(α, grid, Rxy_31, Rxy)
    # Use existing T12_matrix function (now returns T²(r₁₂))
    T2_12 = T12_matrix(α, grid)

    # Use existing T23_matrix function (returns T²(r₂₃) with permutations)
    T2_23 = T23_matrix(α, grid, Rxy)

    # Compute the composite matrix: T²(r₁₂) × Rxy_31 × T²(r₂₃)
    composite_matrix = T2_12 * Rxy_31 * T2_23

    # Apply symmetry factor: (1+ε_{α₃'}ε_{α₁}) = 2
    # This factor of 2 accounts for both T²(r₁₂)T²(r₂₃) and T²(r₃₁)T²(r₁₂) terms
    final_matrix = 2 * composite_matrix

    return final_matrix
end

"""
    full_UIX_potential(α, grid, Rxy_31, Rxy)

Compute the full UIX three-body potential V₁₂⁽³⁾(1+P⁺+P⁻).

Based on the mathematical formula:
⟨x₃ y₃ α₃ | V₁₂⁽³⁾(1+P⁺+P⁻) | ψ₃⟩ = ⟨x₃ y₃ α₃ | V₁₂⁽³⁾ | Ψ⟩ =
A₂π ⟨x₃ y₃ α₃| X₁₂X₃₁I₂₃⁺ + X₁₂X₂₃I₃₁⁻ |Ψ⟩ +
½U₀ ⟨x₃ y₃ α₃| T²(r₁₂)T²(r₂₃) + T²(r₃₁)T²(r₁₂) |Ψ⟩

Where:
- A₂π = -0.0293 MeV (two-pion exchange coupling constant)
- U₀ = 0.0048 MeV (contact interaction coupling constant)

The potential consists of two terms:
1. Two-pion exchange: A₂π × (X₁₂X₃₁I₂₃⁺ + X₁₂X₂₃I₃₁⁻)
2. Contact interaction: ½U₀ × (T²(r₁₂)T²(r₂₃) + T²(r₃₁)T²(r₁₂))

Parameters:
- α: channel structure containing quantum numbers
- grid: coordinate grid for spatial integration
- Rxy_31: rearrangement matrix for coordinate transformation
- Rxy: rearrangement matrix (Rxy_31 + Rxy_32) for permutations

Returns:
- V_UIX: Full UIX three-body potential matrix (MeV units)
"""
function full_UIX_potential(α, grid, Rxy_31, Rxy)
    # Compute the two-pion exchange term: X₁₂X₃₁I₂₃⁺ + X₁₂X₂₃I₃₁⁻
    X_term = X12X31I23_plus_X12X23I31_matrix(α, grid, Rxy)

    # Compute the contact interaction term: T²(r₁₂)T²(r₂₃) + T²(r₃₁)T²(r₁₂)
    T_term = T2_T2_composite_matrix(α, grid, Rxy_31, Rxy)

    # Combine terms with their coupling constants:
    # V₁₂⁽³⁾ = A₂π × X_term + ½U₀ × T_term
    V_UIX = A_2π * X_term + 0.5 * U_0 * T_term

    return V_UIX
end

end  # module UIX