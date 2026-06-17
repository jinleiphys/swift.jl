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
- `Tx_ch`, `Ty_ch`, `Nx`, `Ny`: Component matrices for the V-sector M⁻¹ preconditioner
- `V_x_diag_ch`: Per-channel diagonal V blocks (kept for diagnostics)
- `V_x_full`: Per-pair within-x V blocks for the V-sector M⁻¹ cache
"""
function compute_scattering_matrix(E, α, grid, potname; θ_deg=0.0)
    N = α.nchmax * grid.nx * grid.ny

    println("Computing scattering matrix A = E*B - T - V*(I + Rxy)")
    println("  Energy E = $E MeV")
    println("  Matrix size: $N × $N")

    # Compute kinetic energy matrix T with components
    println("  Computing kinetic energy matrix T...")
    T, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true, θ_deg=θ_deg)

    # Compute potential matrix V with components.
    # V_x_full = per-pair within-x V blocks, needed by the V-sector M⁻¹ preconditioner.
    println("  Computing potential matrix V...")
    if θ_deg == 0.0
        V, V_x_diag_ch = V_matrix_optimized(α, grid, potname, return_components=true)
        V_x_full = V_x_pair_blocks(α, grid, potname)
    else
        V, V_x_diag_ch, V_x_full = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg,
                                                             return_components=true, return_vsector_blocks=true)
    end

    # Compute overlap matrix B
    println("  Computing overlap matrix B...")
    B = kron(Matrix{Float64}(I, α.nchmax, α.nchmax), kron(Nx, Ny))

    # Compute rearrangement matrices
    println("  Computing rearrangement matrices Rxy...")
    Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)

    # Build scattering matrix: A = E*B - T - V*(I + Rxy)
    println("  Assembling scattering matrix A...")

    # Identity matrix
    I_mat = Matrix{ComplexF64}(I, N, N)

    # Compute V*(I + Rxy)
    V_times_I_plus_Rxy = V * (I_mat + Rxy)

    # Final assembly: A = E*B - T - V*(I + Rxy)
    A = E * B - T - V_times_I_plus_Rxy

    println("  Scattering matrix computed successfully.")

    return A, B, T, V, Rxy, Rxy_31, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny, V_x_full
end

"""
    solve_scattering_equation(E, α, grid, potname, φ_θ; θ_deg=0.0)

Solve the inhomogeneous scattering equation [E*B - T - V*(I + Rxy)] c = b by
preconditioned GMRES. The left preconditioner is the V-sector block-diagonal
M^{-1} (Lazauskas split): M = E*B - T - V (full within-sector V), the same fast
cache used by the bound-state solver. Dense LU is intractable at production size.

# Arguments
- `E`: Scattering energy (MeV)
- `α`: Three-body channel structure
- `grid`: Mesh structure
- `potname`: Nuclear potential name (e.g., "AV18")
- `φ_θ`: Initial state vector (from compute_initial_state_vector)
- `θ_deg`: Complex scaling angle in degrees (default 0)

# Returns
- `c`: Solution vector
- `A`: Left-hand side matrix
- `b`: Right-hand side vector

# Example
```julia
α = α3b(J=1/2, T=1/2, parity=1)
grid = initialmesh(nx=20, ny=20, xmax=16, ymax=16, nθ=12)
bound_energies, bound_wavefunctions = bound2b(grid, "AV18")
φ_d = ComplexF64.(bound_wavefunctions[1])
E = 10.0  # MeV
φ_θ = compute_initial_state_vector(grid, α, φ_d, E, z1z2=1.0)
c, A, b = solve_scattering_equation(E, α, grid, "AV18", φ_θ)
```
"""
function solve_scattering_equation(E, α, grid, potname, φ_θ; θ_deg=0.0)
    println("\n" * "="^70)
    println("SOLVING INHOMOGENEOUS SCATTERING EQUATION")
    println("="^70)

    # Compute scattering matrix and component matrices
    A, B, T, V, Rxy, Rxy_31, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny, V_x_full = compute_scattering_matrix(E, α, grid, potname, θ_deg=θ_deg)

    # Compute right-hand side: b = 2 * V * Rxy_31 * φ
    # Factor of 2 from Faddeev symmetry (two equivalent rearrangement channels)
    println("\nComputing right-hand side b = 2 * V * Rxy_31 * φ...")
    b = compute_VRxy_phi(V, Rxy_31, φ_θ)

    # Solve A*c = b by GMRES with the V-sector block-diagonal M^{-1} preconditioner
    # (Lazauskas split): M = E*B - T - V (full within-sector V), block-diagonal over
    # V-sectors. Same cache as the bound-state solver. Dense LU is intractable here.
    println("\nSolving linear system A*c = b (preconditioned GMRES)...")
    println("  System size: $(length(b))")
    println("  Building V-sector M^{-1} cache...")
    M_cache    = matrices.precompute_M_inverse_cache_vsector(α, grid, Tx_ch, Ty_ch, V_x_full, Nx, Ny)
    M_inv_func = matrices.M_inverse_operator_cached_vsector(E, M_cache)
    M_inv_op   = PreconditionerOperator{ComplexF64}(M_inv_func, length(b))

    println("  Running GMRES with preconditioner...")
    c, history = gmres(A, b, Pl=M_inv_op, log=true, verbose=true, maxiter=200, reltol=1e-6)
    println("  GMRES converged in $(history.iters) iterations")

    println("\nSolution computed successfully.")
    println("="^70)

    return c, A, b
end

"""
    compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix::Matrix{ComplexF64}, z1z2; θ=0.0, σ_l=0.0)

Compute the partial-wave scattering amplitude matrix f_{α₀_out, α₀_in}(k) for elastic deuteron scattering.

# Physics:
For elastic scattering, this implements the Lazauskas-Carbonell Green-theorem
integral in the reduced partial-wave representation. In swift's identical-particle
Faddeev basis the Eq.17 short-range source is evaluated as

    [V_j+V_k]Ψ_m = V_i(ψ_j+ψ_k) = V * Rxy * ψ_i,total,

with Rxy = 2Rxy_31 and ψ_i,total = ψ_in + ψ_sc. The projection uses the CS
bilinear product and the reduced deuteron c-norm C_n. The current API carries
one driven solution, so only diagonal projections are populated; a full coupled
on-shell matrix requires one solve per independent incoming channel.

where:
- φ_{α₀} are asymptotic deuteron channels (J12=1, with coherent ³S₁ and ³D₁ components)
- ψ_i,total = ψ_in + ψ_sc is the total Faddeev component for this driven solve
- μ = 2m/3 is the physical n-d reduced mass

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
function compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix::Matrix{ComplexF64}, z1z2; θ=0.0, σ_l=0.0, conj_bra::Bool=false)
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

    # Step 1: Identify physical deuteron channels (J12=1) and collect the
    # ³S₁/³D₁ internal components belonging to the same asymptotic (λ,J3) channel.
    println("  Identifying deuteron channels (J12=1)...")

    deuteron_channels = Vector{Int}()        # Representative three-body channel indices
    deuteron_components = Vector{Vector{Tuple{Int, Int}}}()  # (three-body α, deuteron 2b component)
    channel_labels = Vector{String}()        # Human-readable labels
    group_keys = Vector{Tuple{Int, Float64, Float64, Float64, Float64}}()

    for iα in 1:n_channels
        # Get quantum numbers for this three-body channel
        λ_channel = α.λ[iα]
        J3_channel = α.J3[iα]
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

            if matched_2b_channel > 0 && norm(φ_d_matrix[:, matched_2b_channel]) > 1e-8
                key = (Int(round(λ_channel)), J3_channel, J12_2b, s12_2b, α.T12[iα])
                group_idx = findfirst(==(key), group_keys)
                if group_idx === nothing
                    push!(group_keys, key)
                    push!(deuteron_channels, iα)
                    push!(deuteron_components, [(iα, matched_2b_channel)])
                    push!(channel_labels, "λ=$(Int(round(λ_channel))), J3=$(J3_channel)")
                else
                    push!(deuteron_components[group_idx], (iα, matched_2b_channel))
                    # Prefer the ³S₁ channel as representative for recoupling labels.
                    if l_2b == 0
                        deuteron_channels[group_idx] = iα
                    end
                end
            end
        end
    end

    n_deuteron = length(deuteron_channels)
    println("  Found $n_deuteron deuteron channels:")
    for i in 1:n_deuteron
        comps = join(["α=$(c[1])" for c in deuteron_components[i]], ", ")
        println("    α₀=$i → rep α=$(deuteron_channels[i]): $(channel_labels[i]) [$comps]")
    end

    if n_deuteron == 0
        error("No deuteron channels found! Check channel structure.")
    end

    # Step 2: Compute the total Faddeev component for this incoming state.
    # Eq.17 can be evaluated from the Faddeev source operator
    # [V_j+V_k]Ψ_m ≡ V_i(ψ_j+ψ_k) = V * Rxy * ψ_i,total in swift's component basis.
    # Rxy = Rxy_31 + Rxy_32 = 2 Rxy_31 for the identical-nucleon basis used here.
    println("  Computing ψ₃,total = ψ₃,in + ψ₃,sc...")
    ψ_total = ψ_in + ψ_sc

    # Step 3: Compute [V_j+V_k]Ψ_m on the mesh.
    println("  Computing Eq.17 source operator W = V × Rxy × ψ₃,total...")
    Rxy_ψ_total = 2.0 .* (Rxy_31 * ψ_total)
    W = V * Rxy_ψ_total

    # Reduced deuteron c-norm. Lazauskas writes C_n with the 3D e^{3iθ} volume
    # factor; after the deuteron 1/x radial reduction this is e^{iθ}.
    Nx = matrices_optimized.compute_overlap_matrix(nx, grid.xx)
    evec = ComplexF64[
        φ_d_matrix[j, ich] / grid.ϕx[j]
        for ich in 1:n_2b_channels for j in 1:nx
    ]
    B2b = kron(Matrix{Float64}(I, n_2b_channels, n_2b_channels), Nx)
    C_n = (transpose(evec) * B2b * evec) * exp(im * θ)

    # Step 4: Compute scattering amplitude matrix for deuteron channels
    println("  Computing scattering amplitudes for deuteron channel pairs...")

    # Initialize amplitude matrix (for deuteron channels only)
    f_matrix = zeros(ComplexF64, n_deuteron, n_deuteron)

    # Eq.17 in physical y coordinates: 2μ_y/ℏ² = 4m_N/(3ℏ²).
    # The partial-wave reduced form uses the Riccati regular wave F_λ=qy j_λ(qy),
    # already present in ψ_in, which contributes the extra 1/q² relative to the
    # plane-wave kernel exp(-iq·y)/|y|.
    prefactor = -(2.0 * μ * amu) / (ħ^2 * k_squared) * exp(-im * σ_l) / C_n

    for i_out in 1:n_deuteron
        for i_in in 1:n_deuteron
            if i_in != i_out
                f_matrix[i_out, i_in] = 0.0
                continue
            end

            inner_product = 0.0 + 0.0im
            for (α_out, _) in deuteron_components[i_out]
                idx_start = (α_out - 1) * n_gridpoints + 1
                idx_end = α_out * n_gridpoints
                bra_component = ψ_in[idx_start:idx_end]
                ket_component = W[idx_start:idx_end]
                inner_product += conj_bra ? dot(bra_component, ket_component) :
                                            transpose(bra_component) * ket_component
            end

            # The current API carries one driven solution. A full coupled on-shell
            # matrix requires one solve per independent incoming channel, so only
            # the diagonal projection is populated here.
            f_matrix[i_out, i_in] = prefactor * inner_product
        end
    end

    println("  Scattering amplitude matrix computed:")
    println("    Matrix size: $n_deuteron × $n_deuteron")
    println("    Max |f_{α₀_out,α₀_in}| = $(maximum(abs.(f_matrix)))")
    println("    Deuteron reduced c-norm C_n = $(C_n) (|C_n|=$(abs(C_n)))")
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

        # Step 1: Collect J₃ basis states for this (J, π)
        J3_basis_states = []
        s₃ = 0.5  # Nucleon spin

        for idx in indices
            iα = deuteron_channels[idx]
            λ₃ = α.λ[iα]
            J₃ = α.J3[iα]
            J₁₂ = α.J12[iα]
            push!(J3_basis_states, (λ₃=λ₃, J₃=J₃, J₁₂=J₁₂))
        end

        # Step 2: Generate allowed channel spin states (λ, 𝕊) for this J
        # Channel spin: 𝕊 = J₁₂ ⊕ s₃ (for deuteron: J₁₂=1, so 𝕊 = 1/2 or 3/2)
        # Total: J = λ ⊕ 𝕊
        channel_spin_states = []

        # Get unique λ values
        λ_values = unique([st.λ₃ for st in J3_basis_states])
        J₁₂ = J3_basis_states[1].J₁₂  # Same for all (deuteron J₁₂=1)

        # Generate all (λ, 𝕊) combinations that couple to J
        for λ in λ_values
            for 𝕊 in [abs(J₁₂ - s₃), J₁₂ + s₃]  # 𝕊 = 1/2 or 3/2
                # Check if J is in allowed range: |λ - 𝕊| ≤ J ≤ λ + 𝕊
                if abs(λ - 𝕊) <= J_val <= λ + 𝕊
                    push!(channel_spin_states, (λ=λ, 𝕊=𝕊))
                end
            end
        end

        n_cs = length(channel_spin_states)

        # Step 3: Build transformation matrix T[n_cs, n_states]
        # U^J_{λ𝕊,λ'𝕊'} = Σ_{J₃,J'₃} T*_{λ𝕊,J₃} U_{J₃,J'₃} T_{J'₃,λ'𝕊'}
        # where T_{J₃,λ𝕊} = √(ĵ₃𝕊̂) (-)^{2J-J₃} {λ s₃ J₃; J₁₂ J 𝕊}
        T = zeros(ComplexF64, n_cs, n_states)

        for (i, cs_state) in enumerate(channel_spin_states)
            λ = cs_state.λ
            𝕊 = cs_state.𝕊

            for (j, j3_state) in enumerate(J3_basis_states)
                λ₃ = j3_state.λ₃
                J₃ = j3_state.J₃

                # Only non-zero if λ values match
                if !isapprox(λ, λ₃)
                    continue
                end

                # Unitary recoupling coefficient
                #   ⟨(λ s₃)J₃, J₁₂; J | (J₁₂ s₃)𝕊, λ; J⟩
                #     = (-1)^{λ+s₃+J₁₂+J} √((2J₃+1)(2𝕊+1)) {λ s₃ J₃; J₁₂ J 𝕊}
                # NOTE: WignerSymbols.wigner6j takes PHYSICAL angular momenta, NOT 2j.
                s₃ = 0.5
                dim_factor = sqrt((2*J₃ + 1) * (2*𝕊 + 1))
                phase = (-1)^Int(round(λ + s₃ + J₁₂ + J_val))
                try
                    sixj = wigner6j(λ, s₃, J₃, J₁₂, J_val, 𝕊)
                    T[i, j] = dim_factor * phase * sixj
                catch e
                    T[i, j] = 0.0
                end
            end
        end

        # Step 4: Transform collision matrix to channel spin basis
        # U_CS[n_cs, n_cs] = T * U_Jpi * T†
        U_CS = T * U_Jpi * T'

        U_channel_spin[(J_val, parity)] = U_CS

        # Step 5: Create labels
        labels = String[]
        for cs_state in channel_spin_states
            λ = Int(round(cs_state.λ))
            𝕊 = cs_state.𝕊
            # Format 𝕊 properly for half-integer
            if 𝕊 == round(𝕊)
                𝕊_str = string(Int(𝕊))
            else
                𝕊_str = string(𝕊)
            end
            push!(labels, "λ=$λ, 𝕊=$𝕊_str")
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

    # Step 2: Recouple to channel spin basis (λ, 𝕊)
    println("\n2. Recoupling to channel spin basis...")
    U_channel_spin, cs_labels = recouple_to_channel_spin(U_matrix, α, deuteron_channels)
    println("   Recoupled to (λ, 𝕊) basis")
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
