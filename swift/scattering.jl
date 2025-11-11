# Inhomogeneous scattering equation solver
# Solves: [E*B - T - V*(I + Rxy)] c = b

module Scattering

using LinearAlgebra
using SparseArrays
using IterativeSolvers

include("matrices_optimized.jl")
using .matrices_optimized

include("matrices.jl")
using .matrices

export solve_scattering_equation, compute_scattering_matrix, compute_scattering_amplitude

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

end # module
