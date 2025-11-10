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
    compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E; σ_l=0.0)

Compute the scattering amplitude f(k) for three-body scattering.

# Physics:
The scattering amplitude is:
f(k) = -4μ₃/(ℏ²k_d²) e^(-iσ_l) ⟨φ | V | Rxy_31 | φ + ψ₃^(sc)⟩

where:
- φ is the initial state (ψ_in = φ)
- ψ₃^(sc) is the solution of the Faddeev scattering equation
- μ = 2m/3 is the reduced mass for deuteron-nucleon system

The calculation proceeds as:
ψ_in* × V × Rxy_31 × (ψ_in + ψ_sc)

# Arguments
- `ψ_in`: Initial state vector φ (ψ_in = φ)
- `V`: Potential matrix in α₁ coordinates
- `Rxy_31`: Rearrangement matrix from α₃ to α₁ coordinates
- `ψ_sc`: Scattering wave function ψ₃^(sc) in α₃ coordinates (solution of Faddeev equation)
- `E`: Scattering energy (MeV) in cm frame
- `σ_l`: Coulomb phase shift (default: 0.0)

# Returns
- `f_k`: Complex scattering amplitude f(k)

# Example
```julia
# Compute scattering amplitude where ψ_in = φ
f_k = compute_scattering_amplitude(φ, V, Rxy_31, ψ_sc, E)
```
"""
function compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E; σ_l=0.0)
    # Constants
    ħ = 197.3269718  # MeV·fm (ħc)
    m = 1.0079713395678829     # Nucleon mass in amu
    amu = 931.49432   # MeV (atomic mass unit)

    # Reduced mass: μ = 2m/3 for deuteron-nucleon system
    μ = (2.0 * m) / 3.0  # in amu

    # Compute wave number k from energy E = ħ²k²/(2μ)
    k = sqrt(2.0 * μ * amu * E) / ħ  # in fm⁻¹
    k_squared = k^2

    println("Computing scattering amplitude f(k)...")
    println("  Energy E = $E MeV")
    println("  Wave number k = $k fm⁻¹")
    println("  Coulomb phase σ_l = $σ_l")

    # Compute total wave function in α₃ coordinates
    println("  Computing ψ₃^(total) = ψ₃^(in) + ψ₃^(sc)...")
    ψ_total = ψ_in + ψ_sc

    # Compute the matrix-vector products
    # f(k) ∝ ⟨ψ_in* | V | Rxy_31 | ψ_total⟩

    # Compute Rxy_31 × ψ_total first (more efficient order)
    println("  Computing Rxy_31 × ψ_total...")
    temp1 = Rxy_31 * ψ_total

    # Compute V × temp1
    println("  Computing V × (Rxy_31 × ψ_total)...")
    temp2 = V * temp1

    # Compute inner product ⟨ψ_in | temp2⟩ = ψ_in* · temp2
    println("  Computing ⟨ψ_in | V × Rxy_31 × ψ_total⟩...")
    inner_product = dot(ψ_in, temp2)

    # Apply prefactor
    # f(k) = -4μ₃/(ℏ²k²) e^(-iσ_l) × inner_product
    prefactor = -4.0 * μ * amu / (ħ^2 * k_squared) * exp(-im * σ_l)
    f_k = prefactor * inner_product

    println("  |f(k)| = $(abs(f_k))")
    println("  arg(f(k)) = $(angle(f_k)) rad")
    println("Scattering amplitude computed successfully.")

    return f_k
end

end # module
