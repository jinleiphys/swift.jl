# UIX.jl - Urbana IX three-body force model
# Implementation of Y(r) and T(r) functions for three-body nuclear forces
#
# NOTE: These functions are implemented in the Lagrange function angular momentum basis,
# NOT in the Jacobi coordinate angular momentum basis. This basis choice affects the
# angular momentum coupling and coordinate system used in three-body calculations.

module UIX

export Y, T, S_matrix, X12_matrix

# Physical constants for Urbana IX model
const c = 2.1  # fm^-2, Gaussian cutoff parameter    

# Pion masses in MeV/c^2 (PDG values)
const m_π0 = 134.9768 # neutral pion mass
const m_π_charged = 139.57039  # charged pion mass (π± average)

# Average pion mass: m_π = (1/3)m_π0 + (2/3)m_π±
const m_π = (1/3) * m_π0 + (2/3) * m_π_charged  # MeV/c^2

# Convert to fm^-1 (ħc = 197.3269718 MeV·fm)
const m_π_inv_fm = m_π / 197.3269718  # fm^-1

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
    end

    return X12
end

end  # module UIX