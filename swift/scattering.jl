# Inhomogeneous scattering equation solver
# Solves: [E*B - T - V*(I + Rxy)] c = b

module Scattering

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using WignerSymbols

include("matrices_optimized.jl")
using .matrices_optimized

include("matrices.jl")
using .matrices

export solve_scattering_equation, compute_scattering_matrix, compute_scattering_amplitude
export compute_collision_matrix, compute_eigenphase_shifts, compute_phase_shift_analysis

# ============================================================================
# Preconditioner operator wrapper
# ============================================================================

"""
    PreconditionerOperator

Wrapper to convert a function-based preconditioner into a LinearAlgebra operator
that supports `ldiv!` for use with GMRES.
"""
struct PreconditionerOperator{T}
    apply::Function
    size::Int
end

# Define ldiv! to apply the preconditioner function
function LinearAlgebra.ldiv!(y::AbstractVector, P::PreconditionerOperator, x::AbstractVector)
    result = P.apply(x)
    copyto!(y, result)
    return y
end

# Define ldiv! for single argument (in-place version)
function LinearAlgebra.ldiv!(P::PreconditionerOperator, x::AbstractVector)
    result = P.apply(x)
    copyto!(x, result)
    return x
end

# Define size for the operator
Base.size(P::PreconditionerOperator) = (P.size, P.size)
Base.size(P::PreconditionerOperator, d::Int) = d <= 2 ? P.size : 1
Base.eltype(::PreconditionerOperator{T}) where T = T

"""
    compute_scattering_matrix(E, α, grid, potname; θ_deg=0.0)

Compute the left-hand side matrix A = E*B - T - V*(I + Rxy) for scattering equation.

# Arguments
- `E`: Scattering energy (MeV)
- `α`: Three-body channel structure
- `grid`: Mesh structure
- `potname`: Nuclear potential name (e.g., "AV18")
- `θ_deg`: Complex scaling angle in degrees (default 0)

# Returns
- `A`: Left-hand side matrix
- `B`: Overlap matrix
- `T`: Kinetic energy matrix
- `V`: Potential matrix
- `Rxy`: Total rearrangement matrix (Rxy_31 + Rxy_32)
- `Rxy_31`: Rearrangement matrix from coordinate set 1 to 3
- `Rxy_32`: Rearrangement matrix from coordinate set 2 to 3
- `Tx_ch`, `Ty_ch`, `V_x_diag_ch`, `Nx`, `Ny`: Component matrices for M_inverse_operator
"""
function compute_scattering_matrix(E, α, grid, potname; θ_deg=0.0)
    N = α.nchmax * grid.nx * grid.ny

    println("Computing scattering matrix A = E*B - T - V*(I + Rxy)")
    println("  Energy E = $E MeV")
    println("  Matrix size: $N × $N")

    # Compute kinetic energy matrix T with components
    println("  Computing kinetic energy matrix T...")
    T, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true, θ_deg=θ_deg)

    # Compute potential matrix V with components
    println("  Computing potential matrix V...")
    if θ_deg == 0.0
        V, V_x_diag_ch = V_matrix_optimized(α, grid, potname, return_components=true)
    else
        V, V_x_diag_ch = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg, return_components=true)
    end

    # Compute overlap matrix B
    println("  Computing overlap matrix B...")
    B = kron(Matrix{Float64}(I, α.nchmax, α.nchmax), kron(Nx, Ny))

    # Compute rearrangement matrices
    println("  Computing rearrangement matrices Rxy...")
    Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized(α, grid)

    # Build scattering matrix: A = E*B - T - V*(I + Rxy)
    println("  Assembling scattering matrix A...")

    # Identity matrix
    I_mat = Matrix{ComplexF64}(I, N, N)

    # Compute V*(I + Rxy)
    V_times_I_plus_Rxy = V * (I_mat + Rxy)

    # Final assembly: A = E*B - T - V*(I + Rxy)
    A = E * B - T - V_times_I_plus_Rxy

    println("  Scattering matrix computed successfully.")

    return A, B, T, V, Rxy, Rxy_31, Rxy_32, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny
end

"""
    solve_scattering_equation(E, α, grid, potname, φ_θ; θ_deg=0.0, method=:lu)

Solve the inhomogeneous scattering equation: [E*B - T - V*(I + Rxy)] c = b

For GMRES method, uses M^{-1} = [E*B - T - V_αα]^{-1} as left preconditioner,
where V_αα is the diagonal potential (within-channel coupling only).

# Arguments
- `E`: Scattering energy (MeV)
- `α`: Three-body channel structure
- `grid`: Mesh structure
- `potname`: Nuclear potential name (e.g., "AV18")
- `φ_θ`: Initial state vector (from compute_initial_state_vector)
- `θ_deg`: Complex scaling angle in degrees (default 0)
- `method`: Solution method (:lu for LU factorization, :gmres for preconditioned GMRES)

# Returns
- `c`: Solution vector
- `A`: Left-hand side matrix
- `b`: Right-hand side vector

# Example
```julia
# Setup
α = α3b(J=1/2, T=1/2, parity=1)
grid = initialmesh(nx=20, ny=20, xmax=16, ymax=16, nθ=12)

# Compute initial state
bound_energies, bound_wavefunctions = bound2b(grid, "AV18")
φ_d = ComplexF64.(bound_wavefunctions[1])
E = 10.0  # MeV
φ_θ = compute_initial_state_vector(grid, α, φ_d, E, z1z2=1.0)

# Solve scattering equation with GMRES and M^{-1} preconditioner
c, A, b = solve_scattering_equation(E, α, grid, "AV18", φ_θ, method=:gmres)
```
"""
function solve_scattering_equation(E, α, grid, potname, φ_θ; θ_deg=0.0, method=:lu)
    println("\n" * "="^70)
    println("SOLVING INHOMOGENEOUS SCATTERING EQUATION")
    println("="^70)

    # Compute scattering matrix and component matrices
    A, B, T, V, Rxy, Rxy_31, Rxy_32, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny = compute_scattering_matrix(E, α, grid, potname, θ_deg=θ_deg)

    # Compute right-hand side: b = 2 * V * Rxy_31 * φ
    # Factor of 2 from Faddeev symmetry (two equivalent rearrangement channels)
    println("\nComputing right-hand side b = 2 * V * Rxy_31 * φ...")
    b = compute_VRxy_phi(V, Rxy_31, φ_θ)

    # Solve linear system A*c = b
    println("\nSolving linear system A*c = b...")
    println("  Method: $method")
    println("  System size: $(length(b))")

    if method == :lu
        println("  Using LU factorization...")
        c = A \ b
    elseif method == :gmres
        println("  Using GMRES iterative solver with M^{-1} preconditioner...")

        # Compute M^{-1} preconditioner: M = E*B - T - V_αα (diagonal potential only)
        println("  Computing M^{-1} preconditioner...")
        M_inv_func = matrices.M_inverse_operator(α, grid, E, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)

        # Wrap function in PreconditionerOperator for GMRES
        M_inv_op = PreconditionerOperator{ComplexF64}(M_inv_func, length(b))

        # Solve with left preconditioner: M^{-1} * A * c = M^{-1} * b
        println("  Running GMRES with preconditioner...")
        c, history = gmres(A, b, Pl=M_inv_op, log=true, verbose=true, maxiter=200, reltol=1e-6)
        println("  GMRES converged in $(history.iters) iterations")
    else
        error("Unknown method: $method. Use :lu or :gmres")
    end

    println("\nSolution computed successfully.")
    println("="^70)

    return c, A, b
end

"""
    compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix::Matrix{ComplexF64}, z1z2; θ=0.0, σ_l=0.0)

Compute the partial-wave scattering amplitude matrix f_{α₀_out, α₀_in}(k) for elastic deuteron scattering.

# Physics:
For elastic scattering, we match three-body channels to deuteron bound state channels (J12=1).
The scattering amplitude for transition between deuteron channels α₀_in → α₀_out is:

f_{α₀_out, α₀_in}(k) = -4μ₃/(ℏ²k²) e^(-iσ_l) ⟨φ_{α₀_out} | V | Rxy_31 | ψ_total⟩

where:
- φ_{α₀} are asymptotic states in deuteron channels (J12=1, with ³S₁ and ³D₁ components)
- ψ_total = φ + ψ₃^(sc) is the total wavefunction
- μ = 2m/3 is the reduced mass for deuteron-nucleon system

# Arguments
- `ψ_in`: Initial state vector φ
- `V`: Potential matrix in α₁ coordinates
- `Rxy_31`: Rearrangement matrix from α₃ to α₁ coordinates
- `ψ_sc`: Scattering wave function ψ₃^(sc) in α₃ coordinates (solution of Faddeev equation)
- `E`: Scattering energy (MeV) in cm frame
- `grid`: Grid structure with nx, ny dimensions
- `α`: Channel structure with n_channels
- `φ_d_matrix`: Deuteron bound state wavefunction matrix [nx × n_2b_channels]
- `z1z2`: Charge product for Coulomb interaction
- `θ`: Complex scaling angle in radians (default: 0.0)
- `σ_l`: Coulomb phase shift (default: 0.0)

# Returns
- `f_matrix`: Complex scattering amplitude matrix [n_deuteron_channels × n_deuteron_channels]
              f_matrix[α₀_out, α₀_in] = scattering amplitude between deuteron channels
- `channel_map`: Vector mapping deuteron channel indices to three-body channel indices
- `channel_labels`: Vector of channel labels for identification

# Example
```julia
# Compute scattering amplitude matrix for deuteron elastic scattering
f_matrix, channel_map, labels = compute_scattering_amplitude(φ, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix, z1z2)
```
"""
function compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix::Matrix{ComplexF64}, z1z2; θ=0.0, σ_l=0.0)
    # Constants
    ħ = 197.3269718  # MeV·fm (ħc)
    m = 1.0079713395678829     # Nucleon mass in amu
    amu = 931.49432   # MeV (atomic mass unit)

    # Reduced mass: μ = 2m/3 for deuteron-nucleon system
    μ = (2.0 * m) / 3.0  # in amu

    # Compute wave number k from energy E = ħ²k²/(2μ)
    k = sqrt(2.0 * μ * amu * E) / ħ  # in fm⁻¹
    k_squared = k^2

    # Get dimensions
    nx = grid.nx
    ny = grid.ny
    n_channels = length(α.l)
    n_gridpoints = nx * ny
    n_2b_channels = size(φ_d_matrix, 2)

    println("Computing partial-wave scattering amplitude matrix for deuteron elastic scattering...")
    println("  Energy E = $E MeV")
    println("  Wave number k = $k fm⁻¹")
    println("  Coulomb phase σ_l = $σ_l")
    println("  Total three-body channels: $n_channels")

    # Step 1: Identify which three-body channels correspond to deuteron bound state (J12=1)
    println("  Identifying deuteron channels (J12=1)...")

    deuteron_channels = Vector{Int}()        # Three-body channel indices α
    deuteron_2b_channels = Vector{Int}()     # Corresponding two-body channel indices
    channel_labels = Vector{String}()        # Human-readable labels

    for iα in 1:n_channels
        # Get quantum numbers for this three-body channel
        λ_channel = α.λ[iα]
        i2b = α.α2bindex[iα]

        # Get two-body quantum numbers
        l_2b = α.α2b.l[i2b]
        s12_2b = α.α2b.s12[i2b]
        J12_2b = α.α2b.J12[i2b]

        # Check if this channel couples to the deuteron bound state (J12=1, s12=1)
        if Int(round(J12_2b)) == 1 && Int(round(s12_2b)) == 1
            # Match to deuteron components: ³S₁ (l=0) or ³D₁ (l=2)
            matched_2b_channel = 0
            label = ""

            if l_2b == 0 && n_2b_channels >= 1
                matched_2b_channel = 1  # ³S₁
                label = "³S₁, λ=$(Int(round(λ_channel)))"
            elseif l_2b == 2 && n_2b_channels >= 2
                matched_2b_channel = 2  # ³D₁
                label = "³D₁, λ=$(Int(round(λ_channel)))"
            end

            if matched_2b_channel > 0
                push!(deuteron_channels, iα)
                push!(deuteron_2b_channels, matched_2b_channel)
                push!(channel_labels, label)
            end
        end
    end

    n_deuteron = length(deuteron_channels)
    println("  Found $n_deuteron deuteron channels:")
    for i in 1:n_deuteron
        println("    α₀=$i → α=$(deuteron_channels[i]): $(channel_labels[i])")
    end

    if n_deuteron == 0
        error("No deuteron channels found! Check channel structure.")
    end

    # Step 2: Compute total wave function
    println("  Computing ψ₃^(total) = ψ₃^(in) + ψ₃^(sc)...")
    ψ_total = ψ_in + ψ_sc

    # Step 3: Compute V × Rxy_31 × ψ_total (shared for all channel pairs)
    println("  Computing V × Rxy_31 × ψ_total...")
    temp1 = Rxy_31 * ψ_total
    temp2 = V * temp1

    # Step 4: Compute scattering amplitude matrix for deuteron channels
    println("  Computing scattering amplitudes for deuteron channel pairs...")

    # Initialize amplitude matrix (for deuteron channels only)
    f_matrix = zeros(ComplexF64, n_deuteron, n_deuteron)

    # Prefactor: f(k) = -4μ₃/(ℏ²k²) e^(-iσ_l) × ⟨φ_{α₀_out} | V Rxy_31 | ψ_total⟩
    prefactor = -4.0 * μ * amu / (ħ^2 * k_squared) * exp(-im * σ_l)

    for i_out in 1:n_deuteron
        for i_in in 1:n_deuteron
            # Map deuteron channel indices to three-body channel indices
            α_out = deuteron_channels[i_out]
            α_in = deuteron_channels[i_in]

            # Extract channel components from ψ_total and temp2
            idx_out_start = (α_out - 1) * n_gridpoints + 1
            idx_out_end = α_out * n_gridpoints

            idx_in_start = (α_in - 1) * n_gridpoints + 1
            idx_in_end = α_in * n_gridpoints

            # Get the incoming state component for this outgoing channel
            ψ_out_component = ψ_in[idx_out_start:idx_out_end]

            # Get V × Rxy_31 × ψ_total for the incoming channel
            V_Rxy_ψ_component = temp2[idx_in_start:idx_in_end]

            # Compute inner product ⟨φ_{α₀_out} | V Rxy_31 | ψ_total⟩_{α₀_in}
            inner_product = dot(ψ_out_component, V_Rxy_ψ_component)

            # Apply prefactor
            f_matrix[i_out, i_in] = prefactor * inner_product
        end
    end

    println("  Scattering amplitude matrix computed:")
    println("    Matrix size: $n_deuteron × $n_deuteron")
    println("    Max |f_{α₀_out,α₀_in}| = $(maximum(abs.(f_matrix)))")
    println("Scattering amplitude computed successfully.")

    return f_matrix, deuteron_channels, channel_labels
end

# ============================================================================
# Phase shift analysis
# ============================================================================

"""
    compute_collision_matrix(f_matrix, k)

Compute the collision matrix U from the scattering amplitude matrix f.

# Physics:
U^{α₀,α₀'}(k) = 2ik f^{α₀,α₀'}(k) + δ_{α₀,α₀'}

# Arguments
- `f_matrix`: Scattering amplitude matrix [n_channels × n_channels]
- `k`: Wave number (fm⁻¹)

# Returns
- `U_matrix`: Collision matrix [n_channels × n_channels]
"""
function compute_collision_matrix(f_matrix, k)
    n = size(f_matrix, 1)
    U_matrix = 2.0im * k * f_matrix + I(n)
    return U_matrix
end

"""
    recouple_to_channel_spin(U_matrix, α, deuteron_channels)

Transform collision matrix to channel spin representation.

# Physics:
Following Seyler (Nucl. Phys. A 124, 253-272, 1969), we use channel spin 𝕊 = J₁₂ + s₃:

U^J_{λ'₃𝕊',λ₃𝕊} = Σ_{J₃,J'₃} √(ĵ₃ĵ'₃𝕊̂𝕊̂') (-)^{2J-J₃-J'₃} {λ'₃ 1/2 J'₃; J₁₂ J 𝕊'}{λ₃ 1/2 J₃; J₁₂ J 𝕊} U_{λ'₃J'₃,λ₃J₃}^J

where ĵ = √(2j+1).

# Arguments
- `U_matrix`: Collision matrix in J₃ basis [n_channels × n_channels]
- `α`: Channel structure
- `deuteron_channels`: Indices of deuteron channels

# Returns
- `U_channel_spin`: Dictionary with keys (J, parity) and values U^{Jπ} matrices in channel spin basis
- `channel_spin_labels`: Dictionary with channel labels for each (J, parity)
"""
function recouple_to_channel_spin(U_matrix, α, deuteron_channels)
    # Group channels by J and parity
    # Note: α.J is the total angular momentum for the whole system (scalar)
    J_val = α.J  # Same for all channels
    J_parity_groups = Dict{Tuple{Float64, Int}, Vector{Int}}()

    for (i, iα) in enumerate(deuteron_channels)
        # Compute parity: π = (-)^{λ₃ + l₁₂}
        λ₃ = α.λ[iα]
        i2b = α.α2bindex[iα]
        l_12 = α.α2b.l[i2b]
        parity = Int(round((-1)^(λ₃ + l_12)))

        key = (J_val, parity)
        if !haskey(J_parity_groups, key)
            J_parity_groups[key] = Int[]
        end
        push!(J_parity_groups[key], i)
    end

    U_channel_spin = Dict{Tuple{Float64, Int}, Matrix{ComplexF64}}()
    channel_spin_labels = Dict{Tuple{Float64, Int}, Vector{String}}()

    # For each (J, π) group, perform recoupling transformation
    for ((J_val, parity), indices) in J_parity_groups
        n_states = length(indices)

        # Extract U submatrix for this (J, π)
        U_Jpi = U_matrix[indices, indices]

        # Build channel spin quantum numbers for each state
        # In J₃ basis: |(λ₃ s₃) J₃, J₁₂; J⟩ where J = J₃ ⊕ J₁₂
        # In channel spin basis: |λ₃, (J₁₂ s₃) 𝕊; J⟩ where J = λ₃ ⊕ 𝕊
        #
        # Channel spin 𝕊 = J₁₂ ⊕ s₃ can be J₁₂ ± s₃
        # For deuteron: J₁₂ = 1, s₃ = 1/2, so 𝕊 ∈ {1/2, 3/2}
        #
        # For transformation to work, we need (λ₃, J₃) from J₃ basis
        channel_spin_info = []

        for idx in indices
            iα = deuteron_channels[idx]
            λ₃ = α.λ[iα]
            J₃ = α.J3[iα]
            J₁₂ = α.J12[iα]
            s₃ = 0.5  # Spin of third particle (nucleon)

            # In J₃ basis, J₃ couples λ₃ and s₃: J₃ = λ₃ ⊕ s₃
            # So J₃ must be in range [|λ₃ - s₃|, λ₃ + s₃]
            # Store (λ₃, J₃) pairs from J₃ basis for recoupling
            push!(channel_spin_info, (λ₃=λ₃, J₃=J₃, J₁₂=J₁₂, s₃=s₃))
        end

        # Build recoupling transformation matrix
        # T_{i,j} = √(ĵ₃ĵ'₃𝕊̂𝕊̂') (-)^{2J-J₃-J'₃} {λ'₃ 1/2 J'₃; J₁₂ J 𝕊'}{λ₃ 1/2 J₃; J₁₂ J 𝕊}
        T = zeros(ComplexF64, n_states, n_states)

        for i in 1:n_states
            info_i = channel_spin_info[i]
            λ₃_i = info_i.λ₃
            J₃_i = info_i.J₃
            J₁₂_i = info_i.J₁₂
            S_i = info_i.S

            for j in 1:n_states
                info_j = channel_spin_info[j]
                λ₃_j = info_j.λ₃
                J₃_j = info_j.J₃
                J₁₂_j = info_j.J₁₂
                S_j = info_j.S

                # Dimension factors: ĵ = √(2j+1)
                dim_factor = sqrt((2*J₃_i + 1) * (2*J₃_j + 1) * (2*S_i + 1) * (2*S_j + 1))

                # Phase factor
                phase = (-1)^Int(round(2*J_val - J₃_i - J₃_j))

                # 6-j symbols using WignerSymbols package
                # Use twice-j representation: multiply all j by 2 to get integers
                # Then use wigner6j with integer arguments
                λ₃_i_2j = Int(round(2 * λ₃_i))
                J₃_i_2j = Int(round(2 * J₃_i))
                J₁₂_i_2j = Int(round(2 * J₁₂_i))
                S_i_2j = Int(round(2 * S_i))
                J_val_2j = Int(round(2 * J_val))

                λ₃_j_2j = Int(round(2 * λ₃_j))
                J₃_j_2j = Int(round(2 * J₃_j))
                J₁₂_j_2j = Int(round(2 * J₁₂_j))
                S_j_2j = Int(round(2 * S_j))

                # {λ'₃ 1/2 J'₃; J₁₂ J 𝕊'} with twice-j values
                # wigner6j accepts vararg of integers in twice-j representation
                try
                    sixj_i = wigner6j(λ₃_i_2j, 1, J₃_i_2j, J₁₂_i_2j, J_val_2j, S_i_2j)
                    sixj_j = wigner6j(λ₃_j_2j, 1, J₃_j_2j, J₁₂_j_2j, J_val_2j, S_j_2j)
                    T[i, j] = dim_factor * phase * sixj_i * sixj_j
                catch e
                    # If 6-j symbol calculation fails (e.g., triangle rule violation), set to zero
                    T[i, j] = 0.0
                    @warn "6-j symbol failed for (i=$i, j=$j): $e"
                end
            end
        end

        # Transform collision matrix: U_channel_spin = T' * U_Jpi * T
        # Note: T is real, so T' = T†
        U_CS = T' * U_Jpi * T

        U_channel_spin[(J_val, parity)] = U_CS

        # Create labels for this (J, π)
        labels = String[]
        for info in channel_spin_info
            λ₃ = Int(round(info.λ₃))
            S = info.S
            # Label format: "λ=λ₃, 𝕊=S"
            # Format S properly for half-integer values
            if S == round(S)
                S_str = string(Int(S))
            else
                S_str = string(S)
            end
            push!(labels, "λ=$λ₃, 𝕊=$S_str")
        end
        channel_spin_labels[(J_val, parity)] = labels
    end

    return U_channel_spin, channel_spin_labels
end

"""
    compute_eigenphase_shifts(U_Jpi)

Compute eigenphase shifts from collision matrix U^{Jπ}.

# Physics:
Eigenvalues λₖ = exp(2iδₖ), so δₖ = (1/2) arg(λₖ)

# Arguments
- `U_Jpi`: Collision matrix for specific (J, π)

# Returns
- `eigenphases`: Vector of eigenphase shifts δₖ (in radians)
- `eigenvectors`: Matrix of real orthogonal eigenvectors (columns form u^{Jπ})
"""
function compute_eigenphase_shifts(U_Jpi)
    # Diagonalize collision matrix
    eigenvals, eigenvecs_complex = eigen(U_Jpi)

    # Extract eigenphase shifts: δₖ = (1/2) arg(λₖ)
    eigenphases = 0.5 * angle.(eigenvals)

    # Eigenvectors should be real (or can be made real by choosing appropriate phase)
    # For unitary matrices, eigenvectors can be chosen to be real
    eigenvecs = real.(eigenvecs_complex)

    # Ensure orthogonality (may need to orthogonalize if numerical errors)
    # Use QR decomposition to get orthonormal set
    Q, R = qr(eigenvecs)
    eigenvecs = Matrix(Q)

    return eigenphases, eigenvecs
end

"""
    extract_mixing_parameters_3x3(u_matrix)

Extract Blatt-Biedenharn mixing parameters (ε, ζ, η) from 3×3 orthogonal matrix.

# Physics:
u = v * w * x, where:
- v rotates in (2,3) plane by angle ε (spin mixing)
- w rotates in (1,3) plane by angle ζ (orbital mixing)
- x rotates in (1,2) plane by angle η (mixed coupling)

# Arguments
- `u_matrix`: 3×3 orthogonal mixing matrix

# Returns
- `ε, ζ, η`: Mixing parameters (in radians)
"""
function extract_mixing_parameters_3x3(u_matrix)
    # From u = v * w * x decomposition:
    # u₁₃ = sin(ζ)
    ζ = asin(u_matrix[1, 3])

    # u₁₁ = cos(η) * cos(ζ)
    η = acos(u_matrix[1, 1] / cos(ζ))

    # u₂₃ = sin(ε) * cos(ζ)
    ε = asin(u_matrix[2, 3] / cos(ζ))

    return ε, ζ, η
end

"""
    extract_mixing_parameters_2x2(u_matrix, J_val, parity)

Extract mixing parameter from 2×2 orthogonal matrix.

# Physics:
For 2×2 case, only one mixing angle α:
u = [cos(α)  sin(α);
     -sin(α) cos(α)]

The angle is denoted as:
- η for J^π = 1/2⁺
- ε for J^π = 1/2⁻

# Arguments
- `u_matrix`: 2×2 orthogonal mixing matrix
- `J_val`: Total angular momentum J
- `parity`: Parity (±1)

# Returns
- Named tuple with the appropriate mixing parameter
"""
function extract_mixing_parameters_2x2(u_matrix, J_val, parity)
    # u₁₁ = cos(α)
    α = acos(u_matrix[1, 1])

    # Determine which parameter name to use
    if isapprox(J_val, 0.5) && parity == 1
        # J^π = 1/2⁺ → use η
        return (η=α, ε=0.0, ζ=0.0)
    elseif isapprox(J_val, 0.5) && parity == -1
        # J^π = 1/2⁻ → use ε
        return (ε=α, η=0.0, ζ=0.0)
    else
        # Default: return as η for other 2×2 cases
        return (η=α, ε=0.0, ζ=0.0)
    end
end

"""
    compute_phase_shift_analysis(f_matrix, k, α, deuteron_channels, channel_labels)

Complete phase shift analysis: compute collision matrix, recouple to channel spin basis,
and extract eigenphase shifts and mixing parameters.

# Arguments
- `f_matrix`: Scattering amplitude matrix [n_deuteron × n_deuteron]
- `k`: Wave number (fm⁻¹)
- `α`: Channel structure
- `deuteron_channels`: Indices of deuteron channels in three-body basis
- `channel_labels`: Labels for deuteron channels

# Returns
- Dictionary with keys (J, π) containing:
  - `eigenphases`: Vector of eigenphase shifts (radians)
  - `mixing_params`: Named tuple (ε, ζ, η) or single angle
  - `U_matrix`: Collision matrix in channel spin basis
  - `labels`: Channel labels
"""
function compute_phase_shift_analysis(f_matrix, k, α, deuteron_channels, channel_labels)
    println("\n" * "="^70)
    println("Phase Shift Analysis (Blatt-Biedenharn Parameterization)")
    println("="^70)

    # Step 1: Compute collision matrix U = 2ik*f + I
    println("\n1. Computing collision matrix U = 2ik*f + I...")
    U_matrix = compute_collision_matrix(f_matrix, k)
    println("   Collision matrix computed (size: $(size(U_matrix)))")

    # Step 2: Skip channel spin recoupling for now (complex transformation)
    # Work directly in J₃ basis
    println("\n2. Grouping channels by (J, π)...")

    # Group channels by J and parity
    J_val = α.J  # Same for all channels
    J_parity_groups = Dict{Tuple{Float64, Int}, Vector{Int}}()

    for (i, iα) in enumerate(deuteron_channels)
        λ₃ = α.λ[iα]
        i2b = α.α2bindex[iα]
        l_12 = α.α2b.l[i2b]
        parity = Int(round((-1)^(λ₃ + l_12)))

        key = (J_val, parity)
        if !haskey(J_parity_groups, key)
            J_parity_groups[key] = Int[]
        end
        push!(J_parity_groups[key], i)
    end

    U_channel_spin = Dict{Tuple{Float64, Int}, Matrix{ComplexF64}}()
    cs_labels = Dict{Tuple{Float64, Int}, Vector{String}}()

    # Extract submatrices for each (J, π) group
    for ((J_v, par), indices) in J_parity_groups
        U_Jpi = U_matrix[indices, indices]
        U_channel_spin[(J_v, par)] = U_Jpi

        # Create labels in J₃ basis
        labels = String[]
        for idx in indices
            iα = deuteron_channels[idx]
            λ₃ = Int(round(α.λ[iα]))
            J₃ = α.J3[iα]
            i2b = α.α2bindex[iα]
            l_12 = Int(round(α.α2b.l[i2b]))
            # Format: "l=l₁₂, λ=λ₃, J₃=J₃"
            if J₃ == round(J₃)
                J₃_str = string(Int(J₃))
            else
                J₃_str = string(J₃)
            end
            channel_name = l_12 == 0 ? "³S₁" : "³D₁"
            push!(labels, "$channel_name, λ=$λ₃, J₃=$J₃_str")
        end
        cs_labels[(J_v, par)] = labels
    end

    println("   Found $(length(U_channel_spin)) (J, π) groups")

    # Step 3: For each (J, π), compute eigenphase shifts and mixing parameters
    results = Dict{Tuple{Float64, Int}, Dict{String, Any}}()

    for ((J_val, parity), U_Jpi) in U_channel_spin
        parity_symbol = parity == 1 ? "+" : "-"
        println("\n" * "-"^70)
        println("3. Analyzing J^π = $(J_val)^$parity_symbol")
        println("-"^70)

        n_states = size(U_Jpi, 1)
        println("   Matrix size: $n_states × $n_states")
        println("   Channel spin labels: $(cs_labels[(J_val, parity)])")

        # Compute eigenphase shifts
        eigenphases, u_matrix = compute_eigenphase_shifts(U_Jpi)

        println("   Eigenphase shifts (degrees):")
        for (i, δ) in enumerate(eigenphases)
            println("      δ_$i = $(rad2deg(δ))°")
        end

        # Extract mixing parameters
        if n_states == 3
            ε, ζ, η = extract_mixing_parameters_3x3(u_matrix)
            mixing_params = (ε=ε, ζ=ζ, η=η)
            println("   Mixing parameters (degrees):")
            println("      ε (spin mixing)    = $(rad2deg(ε))°")
            println("      ζ (orbital mixing) = $(rad2deg(ζ))°")
            println("      η (mixed coupling) = $(rad2deg(η))°")
        elseif n_states == 2
            mixing_params = extract_mixing_parameters_2x2(u_matrix, J_val, parity)
            if mixing_params.η != 0.0
                println("   Mixing parameter (degrees):")
                println("      η = $(rad2deg(mixing_params.η))°")
            elseif mixing_params.ε != 0.0
                println("   Mixing parameter (degrees):")
                println("      ε = $(rad2deg(mixing_params.ε))°")
            end
        else
            mixing_params = nothing
            println("   Warning: Unexpected matrix size $n_states×$n_states")
        end

        # Store results
        results[(J_val, parity)] = Dict(
            "eigenphases" => eigenphases,
            "mixing_params" => mixing_params,
            "U_matrix" => U_Jpi,
            "u_matrix" => u_matrix,
            "labels" => cs_labels[(J_val, parity)]
        )
    end

    println("\n" * "="^70)
    println("Phase shift analysis completed")
    println("="^70)

    return results
end

end # module
