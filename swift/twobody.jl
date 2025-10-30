# this module use to compute the two body bound state for testing the NN force
module twobodybound
using LinearAlgebra
using FastGaussQuadrature
using Printf
include("../NNpot/nuclear_potentials.jl")
using .NuclearPotentials
using Kronecker
include("laguerre.jl")
using .Laguerre

export bound2b, compute_deuteron_wavefunction


const amu= 931.49432 # MeV
const mn=1.008665 # amu
const mp=1.007276 # amu 
const μ=mn*mp/(mn+mp) # amu
const ħ=197.3269718 # MeV. fm


mutable struct nch2b # channel index for the three body coupling
    nchmax::Int # maximum number of channels
    s1::Float64
    s2::Float64
    l::Vector{Int}
    s12::Vector{Float64}
    J12 :: Float64
    
    # Constructor with default initialization
    function nch2b()
        new(0, 0.0, 0.0, Int[], Float64[], 0.0)
    end
end


function bound2b(grid, potname; θ_deg=0.0, n_gauss=nothing, verbose=false)
    """
    Solve two-body bound state problem with optional complex scaling

    Parameters:
    - grid: Mesh structure
    - potname: Nuclear potential name (e.g., "AV18")
    - θ_deg: Complex scaling angle in degrees (default: 0.0)
    - n_gauss: Number of Gauss quadrature points (default: 5*grid.nx)
    - verbose: Print detailed information
    """
    α = channelindex()

    # Initialize matrices with complex scaling
    Tαx = T_matrix_scaled(α, grid, θ_deg=θ_deg)
    V = V_matrix_scaled(α, grid, potname, θ_deg=θ_deg, n_gauss=n_gauss, verbose=verbose)
    B = Bmatrix(α, grid)

    # Construct the Hamiltonian matrix
    H = Tαx + V

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigen(H, B)

    # Extract the bound state energies and wave functions
    bound_energies = []
    bound_wavefunctions = []

    println("\n" * "="^60)
    println("           TWO-BODY BOUND STATE ANALYSIS")
    if θ_deg != 0.0
        println("           Complex Scaling: θ = $(θ_deg)°")
    end
    println("="^60)

    # Adjust imaginary part tolerance for complex scaling
    imag_tol = θ_deg == 0.0 ? 1e-6 : 0.1  # More lenient for complex scaling

    bound_count = 0
    for i in 1:grid.nx*α.nchmax
        # For complex scaling, check both real part negative and small imaginary part
        if real(eigenvalues[i]) < 0.0 && abs(imag(eigenvalues[i])) < imag_tol
            bound_count += 1
            eigenvec = eigenvectors[:, i]
            norm = sum(abs2.(eigenvec))

            if norm ≠ 1.0
                eigenvec = eigenvec / sqrt(norm)
            end

            # Compute the wave function with multiple components
            # Use appropriate type based on whether eigenvector is complex
            WaveType = eltype(eigenvec)
            wavefunction = zeros(WaveType, grid.nx, α.nchmax)
            for ich in 1:α.nchmax
                for j in 1:grid.nx
                    idx = (ich-1)*grid.nx + j
                    wavefunction[j, ich] = grid.ϕx[j] * eigenvec[idx]
                end
            end 
            
            # Calculate component probabilities
            component_norms = zeros(α.nchmax)
            for ich in 1:α.nchmax
                component_norms[ich] = sum(abs2.(wavefunction[:, ich]) .* grid.dxi)
            end
            
            # Normalize component probabilities
            total_norm = sum(component_norms)
            component_probs = component_norms / total_norm * 100
            
            # Print detailed bound state information
            println("\nBound State #$bound_count:")
            if θ_deg == 0.0
                println("  Binding Energy: $(round(real(eigenvalues[i]), digits=6)) MeV")
            else
                @printf("  Binding Energy: %.6f + %.6f i MeV\n", real(eigenvalues[i]), imag(eigenvalues[i]))
            end
            println("  Total J^π = $(Int(α.J12))⁺")
            println("\n  Channel Composition:")
            
            for ich in 1:α.nchmax
                l_val = α.l[ich]
                s_val = α.s12[ich]
                # Determine spectroscopic notation
                if l_val == 0
                    l_notation = "S"
                elseif l_val == 1
                    l_notation = "P"
                elseif l_val == 2
                    l_notation = "D"
                elseif l_val == 3
                    l_notation = "F"
                else
                    l_notation = "L=$l_val"
                end
                
                # Format as 2S+1L_J notation
                spectro_notation = "$(Int(2*s_val+1))$(l_notation)₁"
                
                println("    Channel $ich: $spectro_notation (l=$l_val, s=$(s_val)) - $(round(component_probs[ich], digits=2))%")
            end
            
            # Calculate D-state probability specifically
            d_state_prob = 0.0
            for ich in 1:α.nchmax
                if α.l[ich] == 2  # D-state has l=2
                    d_state_prob += component_probs[ich]
                end
            end
            
            println("\n  D-state Probability: $(round(d_state_prob, digits=3))%")
            println("  S-state Probability: $(round(100.0 - d_state_prob, digits=3))%")
            
            push!(bound_energies, eigenvalues[i])
            push!(bound_wavefunctions, wavefunction)
        end
    end
    
    println("\n" * "="^60)
    println("SUMMARY: Found $bound_count bound state(s)")
    if bound_count > 0
        println("Binding energies (MeV): ", [round(e, digits=6) for e in bound_energies])
    end
    println("="^60)

    return bound_energies, bound_wavefunctions

end 

 function Bmatrix(α,grid)
    # compute the B matrix for the Generalized eigenvalue problem
    Iα = Matrix{Float64}(I, α.nchmax, α.nchmax)
    Ix=zeros(grid.nx, grid.nx)
    for i in 1:grid.nx
        for j in 1:grid.nx
            if i == j
                Ix[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
            else
                Ix[i,j] = (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
            end
        end
    
    end 


    Bmatrix = Iα ⊗ Ix

    return Bmatrix


 end 


function V_matrix_scaled(α, grid, potname; θ_deg=0.0, n_gauss=nothing, verbose=false)
    """
    Compute potential matrix with complex scaling using backward rotation

    For θ = 0: Use standard method (evaluate at mesh points)
    For θ ≠ 0: Use backward rotation with Gauss quadrature
                V_ij(θ) = exp(-iθ) ∫ φᵢ(r exp(-iθ)) V(r) φⱼ(r exp(-iθ)) dr

    Parameters:
    - α: Channel structure
    - grid: Mesh structure
    - potname: Nuclear potential name
    - θ_deg: Complex scaling angle in degrees (default: 0.0)
    - n_gauss: Number of Gauss quadrature points (default: 5*grid.nx)
    - verbose: Print detailed information
    """
    # For θ=0, use standard method (diagonal in coordinate space)
    if θ_deg == 0.0
        l = [α.l[i] for i in 1:α.nchmax]
        V = zeros(α.nchmax*grid.nx, α.nchmax*grid.nx)

        for ir in 1:grid.nx
            vpot = potential_matrix(potname, grid.xi[ir], l, 1, Int(α.J12), 0, 0)  # np pair
            for i in 1:α.nchmax
                for j in 1:α.nchmax
                    V[(i-1)*grid.nx+ir, (j-1)*grid.nx+ir] = vpot[i,j]
                end
            end
        end
        if verbose
            println("Standard V (θ=0): V[1,1] = $(V[1,1]), V[nx,nx] = $(V[grid.nx, grid.nx])")
        end
        return V
    end

    # For θ ≠ 0, use backward rotation
    if n_gauss === nothing
        n_gauss = 5 * grid.nx  # Sufficient quadrature accuracy
    end

    θ = θ_deg * π / 180.0
    jacobian_factor = exp(-im * θ)

    if verbose
        println("\nComputing complex-scaled 2-body potential:")
        println("  θ = $(θ_deg)° = $(round(θ, digits=4)) rad")
        println("  Gauss quadrature points: $(n_gauss)")
        println("  Using backward rotation method")
    end

    # Gauss-Legendre quadrature on [0, rmax]
    rmax = grid.xmax  # Use the actual mesh range
    r_quad_std, w_quad_std = gausslegendre(n_gauss)
    # Map from [-1, 1] to [0, rmax]
    r_quad = Float64.((r_quad_std .+ 1.0) .* (rmax / 2.0))
    w_quad = Float64.(w_quad_std .* (rmax / 2.0))

    # Get channel l values
    l = [α.l[i] for i in 1:α.nchmax]

    # Initialize complex matrix
    V = zeros(Complex{Float64}, α.nchmax*grid.nx, α.nchmax*grid.nx)

    # Compute matrix elements via quadrature
    # V_ij(θ) = e^(-iθ) ∫ φᵢ(r e^(-iθ)) V(r) φⱼ(r e^(-iθ)) dr

    max_phi = 0.0
    max_integrand = 0.0

    for ich in 1:α.nchmax
        for jch in 1:α.nchmax
            for ix in 1:grid.nx
                for jx in 1:grid.nx
                    integral = zero(Complex{Float64})

                    for iq in 1:n_gauss
                        r_k = Float64(r_quad[iq])  # Physical coordinate (fm)
                        w_k = Float64(w_quad[iq])

                        # BACKWARD ROTATION: Evaluate basis at ROTATED coordinate
                        phi_all = lagrange_laguerre_regularized_basis(r_k, grid.xi, grid.ϕx, grid.α, grid.hsx, θ)
                        phi_i = phi_all[ix]
                        phi_j = phi_all[jx]

                        max_phi = max(max_phi, abs(phi_i), abs(phi_j))

                        # Evaluate potential at REAL coordinate r_k (not rotated!)
                        vpot = potential_matrix(potname, r_k, l, 1, Int(α.J12), 0, 0)

                        # Integrand: φᵢ(r/e^(iθ)) V(r) φⱼ(r/e^(iθ))
                        # IMPORTANT: NO CONJUGATE! (COLOSS line 360-361)
                        integrand = phi_i * vpot[ich, jch] * phi_j
                        max_integrand = max(max_integrand, abs(integrand))
                        integral += w_k * integrand
                    end

                    # Apply Jacobian factor: exp(-iθ)
                    i_idx = (ich-1)*grid.nx + ix
                    j_idx = (jch-1)*grid.nx + jx
                    V[i_idx, j_idx] = jacobian_factor * integral
                end
            end
        end
    end

    if verbose
        println("  Max |φ|: $(max_phi)")
        println("  Max |integrand|: $(max_integrand)")
        println("  V[1,1] = $(V[1,1])")
        println("  V[grid.nx, grid.nx] = $(V[grid.nx, grid.nx])")
        println("  Complex-scaled potential matrix computed successfully")
    end

    return V
end

function V_matrix(α, grid, potname)
    """Legacy wrapper for backward compatibility"""
    return V_matrix_scaled(α, grid, potname, θ_deg=0.0)
end 


function gaussianpot(α,grid)
    # Define the Gaussian potential parameters
    V0 = -72.15  # Depth of the potential well
    r0 = 0  # Range of the potential
    a = 1.484  # Width of the potential
    V = zeros(α.nchmax*grid.nx,α.nchmax*grid.nx)
    # Define the Gaussian potential function
    for ir in 1:grid.nx
        r = grid.xi[ir]
        V[ir,ir] = V0 * exp(-((r - r0)^2) /  a^2)
    end 


    return V

end 



function T_matrix_scaled(α, grid; θ_deg=0.0)
    """
    Compute kinetic energy matrix with complex scaling

    T(θ) = exp(-2iθ) T(0)

    Parameters:
    - α: Channel structure
    - grid: Mesh structure
    - θ_deg: Complex scaling angle in degrees (default: 0.0)
    """
    # Convert angle to radians
    θ = θ_deg * π / 180.0

    # Complex scaling factor for kinetic energy
    scaling_factor = exp(-2im * θ)

    # Determine data type - always use Complex when θ != 0
    is_complex = (θ_deg != 0.0)
    DataType_T = is_complex ? Complex{Float64} : Float64

    Tαx = zeros(DataType_T, α.nchmax*grid.nx, α.nchmax*grid.nx)

    for i in 1:α.nchmax
        T = Tx(grid.nx, grid.xx, grid.α, α.l[i])

        if is_complex
            T = T .* ħ^2 / (2.0 * μ * amu * grid.hsx^2) .* scaling_factor
        else
            T = T .* ħ^2 / (2.0 * μ * amu * grid.hsx^2)
        end

        row_start = (i-1)*grid.nx + 1
        row_end = i*grid.nx
        col_start = (i-1)*grid.nx + 1
        col_end = i*grid.nx
        Tαx[row_start:row_end, col_start:col_end] = T
    end

    return Tαx
end

function T_matrix(α, grid)
    """Legacy wrapper for backward compatibility"""
    return T_matrix_scaled(α, grid, θ_deg=0.0)
end 


function channelindex()
    # Initialize the channel index
    α = nch2b()
    α.s1 = 0.5
    α.s2 = 0.5
    α.J12 = 1.0 
    for l in 0:2
        if (-1)^l == -1
            continue
        end
        for ns in Int(2*( α.s1- α.s2)):2:Int(2*( α.s1+ α.s2))
            s12 = ns/2.0
            for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))  # Fixed min/max calculation
                if nJ12 != Int(2*α.J12)
                    continue
                end
                α.nchmax += 1
            end 
        end 
    end 

    println("\nTwo-body channel configuration:")
    println("  Total angular momentum J = ", α.J12)
    println("  Parity = +")
    println("  Number of channels: ", α.nchmax)

    α.l = zeros(Int,  α.nchmax)
    α.s12 = zeros(Float64, α.nchmax)
    nch=0
    for l in 0:2
        if (-1)^l == -1
            continue
        end
        for ns in Int(2*( α.s1- α.s2)):2:Int(2*( α.s1+ α.s2))
            s12 = ns/2.0
            for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))  # Fixed min/max calculation
                if nJ12 != Int(2*α.J12)
                    continue
                end
                nch += 1
                α.l[nch] = l
                α.s12[nch] = s12
                
                # Print channel information
                l_notation = l == 0 ? "S" : l == 1 ? "P" : l == 2 ? "D" : "L=$l"
                spectro_notation = "$(Int(2*s12+1))$(l_notation)₁"
                println("    Channel $nch: $spectro_notation (l=$l, s=$(s12))")
                
            end 
        end 
    end
    return α
end



function Tx(nx,xi,α0,l)
    # Compute the T matrix 
    # Parameters:
    # nx: number of points in the radial mesh
    # xi: radial mesh points
    # α0: parameter for the Laguerre function
    # l: angular momentum quantum number
    T = zeros(nx,nx)  # Initialize T matrix

    for i in 1:nx
        for j in 1:nx
            if i == j
                # Diagonal elements 
                T[i,j] = (-1.0 / (12.0 * xi[i]^2)) * (xi[i]^2 - 2.0 * (2.0 * nx + α0 + 1.0) * xi[i] + α0^2 - 4.0) - 
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j])) + 
                         l * (l + 1) / xi[i]^2

                # T[i,j] = (-1.0 / (12.0 * xi[i]^2)) * (xi[i]^2 - 2.0 * (2.0 * nx + α0 + 1.0) * xi[i] + α0^2 - 4.0) + 
                #          l * (l + 1) / xi[i]^2
            else
                # Off-diagonal elements 
                T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) - 
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j]))


                # T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) 
            end
        end
    end


    return T

 end # end function Tx


"""
    compute_deuteron_wavefunction(α3b, grid, bound_energies, bound_wavefunctions, channel_idx=1)

Extract deuteron wavefunction for a specific three-body channel from bound2b results.

# Problem
The `bound2b` function computes two-body bound states (e.g., deuteron) in a small model space
with J=1 (typically ³S₁ + ³D₁ channels). However, three-body channels may have different
two-body quantum numbers (different J12, l, s12 values). This function remaps the bound2b
wavefunction to match a specific three-body channel.

# Mapping Strategy
1. Get the two-body channel index from three-body channel: `i2b = α3b.α2bindex[channel_idx]`
2. Get quantum numbers from three-body α2b structure: (l, s12, J12) from `α3b.α2b`
3. Match these quantum numbers to the bound2b channel structure
4. Extract the corresponding wavefunction component

# Arguments
- `α3b`: Three-body channel structure from α3b() function
  - Contains `α3b.α2b` (two-body channels) and `α3b.α2bindex` (mapping)
- `grid`: Mesh structure containing grid points (must match bound2b grid)
- `bound_energies`: Binding energies from bound2b (Vector)
- `bound_wavefunctions`: Wavefunctions from bound2b (Vector of matrices)
  - Each wavefunction is (grid.nx, n_2b_channels) matrix
- `channel_idx`: Index of the three-body channel (default=1)
  - Uses this to look up corresponding two-body channel

# Returns
- `φ_d::Vector{ComplexF64}`: Deuteron wavefunction on x-grid (length grid.nx)
  - Extracted from the appropriate two-body channel component
  - Returns ground state (first bound state) by default

# Notes
- For J12≠1 channels: If bound2b was computed with J=1, the wavefunction will be zeros
  - In this case, you need to compute bound2b with the appropriate J12 value
- The function performs quantum number matching: (l, s12, J12, T12)
- Complex scaling: Supports complex wavefunctions from bound2b with θ≠0

# Example
```julia
# Compute bound state with bound2b
bound_energies, bound_wavefunctions = bound2b(grid, "AV18")

# Get three-body channels
α = α3b(true, 0.5, 0.5, +1, 2, 0, 2, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 1.0, +1)

# Extract deuteron wavefunction for channel 1
φ_d = compute_deuteron_wavefunction(α, grid, bound_energies, bound_wavefunctions, 1)

# Use in scattering calculation
φ_in = compute_initial_state_vector(grid, α, φ_d, E, z1z2)
```

# Warning
If the three-body channel requires J12 ≠ 1, you must recompute bound2b with that J12:
```julia
# For J12 = 2 channel (e.g., ⁵D₂)
# You would need to modify channelindex() in twobody.jl to use J12 = 2
```
"""
"""
    compute_deuteron_wavefunction(grid, bound_energies, bound_wavefunctions)

Extract the deuteron wavefunction matrix from bound2b results.

Returns the complete deuteron wavefunction with all angular momentum components.
For the deuteron (J=1), this includes both ³S₁ (l=0) and ³D₁ (l=2) channels.

# Arguments
- `grid`: Mesh structure (used for consistency checking)
- `bound_energies`: Vector of bound state energies from bound2b
- `bound_wavefunctions`: Vector of wavefunction matrices from bound2b

# Returns
- `φ_d_matrix`: Deuteron wavefunction matrix (grid.nx × n_2b_channels)
  - Typically size (nx, 2) for deuteron with ³S₁ and ³D₁ components
  - Column 1: ³S₁ component (l=0, s12=1, J12=1)
  - Column 2: ³D₁ component (l=2, s12=1, J12=1)

# Example
```julia
# Compute deuteron bound state
bound_energies, bound_wavefunctions = bound2b(grid, potential)

# Extract deuteron wavefunction (all components)
φ_d_matrix = compute_deuteron_wavefunction(grid, bound_energies, bound_wavefunctions)

# Use in scattering calculation
φ = compute_initial_state_vector(grid, α, φ_d_matrix, E, z1z2)
```
"""
function compute_deuteron_wavefunction(grid, bound_energies, bound_wavefunctions)
    # Check if we have any bound states
    if isempty(bound_wavefunctions)
        error("No bound states found from bound2b. Check potential or grid parameters.")
    end

    # Use the ground state (first bound state)
    # wavefunction is a matrix of size (grid.nx, n_2b_channels)
    φ_d_matrix = bound_wavefunctions[1]

    # Get dimensions for validation
    nx_check, n_channels = size(φ_d_matrix)
    if nx_check != grid.nx
        @warn "Wavefunction grid size ($nx_check) doesn't match grid.nx ($(grid.nx))"
    end

    # Print information about the deuteron
    println("\nDeuteron wavefunction extraction:")
    println("  Binding energy: $(round(real(bound_energies[1]), digits=6)) MeV")
    println("  Number of channels: $n_channels")
    println("  Grid size: $(grid.nx) points")

    if n_channels == 2
        # Calculate S and D state probabilities
        # Note: This assumes proper normalization from bound2b
        norm_S = sum(abs2.(φ_d_matrix[:, 1]) .* grid.wx)
        norm_D = sum(abs2.(φ_d_matrix[:, 2]) .* grid.wx)
        total = norm_S + norm_D

        if total > 0
            prob_S = norm_S / total * 100
            prob_D = norm_D / total * 100
            println("  ³S₁ probability: $(round(prob_S, digits=2))%")
            println("  ³D₁ probability: $(round(prob_D, digits=2))%")
        end
    end

    return ComplexF64.(φ_d_matrix)
end

# Legacy version for backward compatibility (deprecated)
function compute_deuteron_wavefunction(α3b, grid, bound_energies, bound_wavefunctions,
                                      channel_idx=1)
    """
    DEPRECATED: Legacy version that extracts a single channel.
    Use compute_deuteron_wavefunction(grid, bound_energies, bound_wavefunctions) instead.

    This version is kept for backward compatibility but should not be used in new code.
    """
    @warn """
    Deprecated: compute_deuteron_wavefunction with channel_idx parameter is deprecated.
    Use: φ_d_matrix = compute_deuteron_wavefunction(grid, bound_energies, bound_wavefunctions)
    This returns the full matrix with all deuteron components.
    """

    # Get the corresponding two-body channel index for this three-body channel
    i2b = α3b.α2bindex[channel_idx]

    # Get quantum numbers from the three-body α2b structure
    l_3b = α3b.α2b.l[i2b]
    s12_3b = α3b.α2b.s12[i2b]
    J12_3b = α3b.α2b.J12[i2b]
    T12_3b = α3b.α2b.T12[i2b]

    # Check if we have any bound states
    if isempty(bound_wavefunctions)
        error("No bound states found from bound2b. Check potential or grid parameters.")
    end

    # Use the ground state (first bound state)
    # wavefunction is a matrix of size (grid.nx, n_2b_channels)
    wavefunction = bound_wavefunctions[1]

    # Get the two-body channel structure from bound2b
    # This is typically J=1 with ³S₁ and ³D₁ channels
    α2b_bound = channelindex()  # Recreate the same channel structure as bound2b

    # Find matching channel in bound2b wavefunction
    # Match by quantum numbers (l, s12, J12)
    matched_channel = 0

    for ich_2b in 1:α2b_bound.nchmax
        l_2b = α2b_bound.l[ich_2b]
        s12_2b = α2b_bound.s12[ich_2b]
        J12_2b = α2b_bound.J12

        # Match quantum numbers
        if l_2b == l_3b && abs(s12_2b - s12_3b) < 1e-10 && abs(J12_2b - J12_3b) < 1e-10
            matched_channel = ich_2b
            break
        end
    end

    # Extract wavefunction component
    if matched_channel > 0
        # Get the wavefunction for this channel
        φ_d = wavefunction[:, matched_channel]

        println("\nDeuteron wavefunction extraction (LEGACY):")
        println("  Three-body channel: $channel_idx → Two-body index: $i2b")
        println("  Quantum numbers: l=$l_3b, s12=$s12_3b, J12=$J12_3b, T12=$T12_3b")
        println("  Matched to bound2b channel: $matched_channel")
        println("  Binding energy: $(round(real(bound_energies[1]), digits=6)) MeV")

        return ComplexF64.(φ_d)
    else
        # No matching channel found - might need different J12
        @warn """
        No matching two-body channel found for three-body channel $channel_idx!
        Three-body requires: l=$l_3b, s12=$s12_3b, J12=$J12_3b
        Available bound2b channels have J=$(α2b_bound.J12)

        Returning zero wavefunction. You may need to:
        1. Recompute bound2b with J12=$J12_3b
        2. Or check if this channel should couple to the bound state
        """

        return zeros(ComplexF64, grid.nx)
    end
end

end 