# Inhomogeneous scattering equation solver
# Solves: [E*B - T - V*(I + Rxy)] c = b

module Scattering

using LinearAlgebra
using SparseArrays

include("matrices_optimized.jl")
using .matrices_optimized

export solve_scattering_equation, compute_scattering_matrix

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
"""
function compute_scattering_matrix(E, α, grid, potname; θ_deg=0.0)
    N = α.nchmax * grid.nx * grid.ny

    println("Computing scattering matrix A = E*B - T - V*(I + Rxy)")
    println("  Energy E = $E MeV")
    println("  Matrix size: $N × $N")

    # Compute overlap matrix B
    println("  Computing overlap matrix B...")
    Nx = matrices_optimized.compute_overlap_matrix(grid.nx, grid.xx)
    Ny = matrices_optimized.compute_overlap_matrix(grid.ny, grid.yy)
    B = kron(Matrix{Float64}(I, α.nchmax, α.nchmax), kron(Nx, Ny))

    # Compute kinetic energy matrix T
    println("  Computing kinetic energy matrix T...")
    T = T_matrix_optimized(α, grid, θ_deg=θ_deg)

    # Compute potential matrix V
    println("  Computing potential matrix V...")
    if θ_deg == 0.0
        V = V_matrix_optimized(α, grid, potname)
    else
        V = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg)
    end

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

    return A, B, T, V, Rxy, Rxy_31, Rxy_32
end

"""
    solve_scattering_equation(E, α, grid, potname, φ_θ; θ_deg=0.0, method=:lu)

Solve the inhomogeneous scattering equation: [E*B - T - V*(I + Rxy)] c = b

# Arguments
- `E`: Scattering energy (MeV)
- `α`: Three-body channel structure
- `grid`: Mesh structure
- `potname`: Nuclear potential name (e.g., "AV18")
- `φ_θ`: Initial state vector (from compute_initial_state_vector)
- `θ_deg`: Complex scaling angle in degrees (default 0)
- `method`: Solution method (:lu for LU factorization, :iterative for GMRES)

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

# Solve scattering equation
c, A, b = solve_scattering_equation(E, α, grid, "AV18", φ_θ)
```
"""
function solve_scattering_equation(E, α, grid, potname, φ_θ; θ_deg=0.0, method=:lu)
    println("\n" * "="^70)
    println("SOLVING INHOMOGENEOUS SCATTERING EQUATION")
    println("="^70)

    # Compute scattering matrix
    A, B, T, V, Rxy, Rxy_31, Rxy_32 = compute_scattering_matrix(E, α, grid, potname, θ_deg=θ_deg)

    # Compute right-hand side: b = V * Rxy_31 * φ
    println("\nComputing right-hand side b = V * Rxy_31 * φ...")
    b = compute_VRxy_phi(V, Rxy_31, φ_θ)

    # Solve linear system A*c = b
    println("\nSolving linear system A*c = b...")
    println("  Method: $method")
    println("  System size: $(length(b))")

    if method == :lu
        println("  Using LU factorization...")
        c = A \ b
    elseif method == :iterative
        println("  Using GMRES iterative solver...")
        using IterativeSolvers
        c, history = gmres(A, b, log=true, verbose=true)
        println("  GMRES converged in $(length(history)) iterations")
    else
        error("Unknown method: $method. Use :lu or :iterative")
    end

    println("\nSolution computed successfully.")
    println("="^70)

    return c, A, b
end

end # module
