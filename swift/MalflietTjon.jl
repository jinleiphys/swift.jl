module MalflietTjon

include("matrices.jl")
using .matrices
using LinearAlgebra
using Printf

export malfiet_tjon_solve, compute_lambda_eigenvalue, print_convergence_summary

"""
    MalflietTjonResult

Structure to hold results from Malfiet-Tjon eigenvalue calculation.

# Fields
- `energy::Float64`: Converged ground state energy
- `eigenvalue::Float64`: Final λ eigenvalue (should be ≈ 1.0)
- `eigenvector::Vector{ComplexF64}`: Ground state wave function
- `iterations::Int`: Number of secant method iterations
- `convergence_history::Vector{Tuple{Float64, Float64}}`: History of (E, λ) pairs
- `converged::Bool`: Whether the calculation converged
"""
struct MalflietTjonResult
    energy::Float64
    eigenvalue::Float64
    eigenvector::Vector{ComplexF64}
    iterations::Int
    convergence_history::Vector{Tuple{Float64, Float64}}
    converged::Bool
end

"""
    compute_lambda_eigenvalue(E0, α, grid, potname, e2b)

Compute the largest eigenvalue λ(E0) for the equation:
λ(E0) [c] = [E0*B - H0 - V]⁻¹ R [c]

where:
- E0: guessed energy
- H0: kinetic energy matrix
- V: potential energy matrix  
- B: overlap matrix
- R: rearrangement matrix

Returns the largest eigenvalue λ and corresponding eigenvector.
"""
function compute_lambda_eigenvalue(E0::Float64, α, grid, potname, e2b; verbose::Bool=false)
    # Construct the matrices
    T = T_matrix(α, grid)           # Kinetic energy H0
    V = V_matrix(α, grid, potname)  # Potential energy
    B = Bmatrix(α, grid)            # Overlap matrix
    Rxy = Rxy_matrix(α, grid)       # Rearrangement matrix R
    
    # Form the left-hand side operator: E0*B - H0 - V
    # where H0 = T (kinetic energy) and V is the potential
    LHS = E0 * B - T - V
    
    # Check if matrix is singular
    cond_num = cond(LHS)
    if cond_num > 1e12
        if verbose
            @warn "Matrix [E0*B - T - V] is near-singular at E0 = $E0, condition number = $(cond_num)"
        end
        return NaN, nothing
    end
    
    # The Malfiet-Tjon equation from: (E*B - T - V - V*R)[c] = 0
    # Alternative formulation: Try V*R * [E*B - T - V]⁻¹ instead
    VRxy = V * Rxy
    
    # Form the right-hand side operator: [E0*B - T - V]⁻¹ * V*R
    try
        RHS = LHS \ VRxy
        
        # Find the largest eigenvalue and corresponding eigenvector
        eigenvals, eigenvecs = eigen(RHS)
        
        # For debugging: let's look at all eigenvalues
        eigenvals_real = real.(eigenvals)
        eigenvals_sorted = sort(eigenvals_real, rev=true)  # Sort descending
        
        # The ground state should correspond to the eigenvalue closest to 1
        # Find eigenvalue closest to 1
        distances_from_1 = abs.(eigenvals_real .- 1.0)
        closest_idx = argmin(distances_from_1)
        λ_closest = eigenvals[closest_idx]
        eigenvec = eigenvecs[:, closest_idx]
        
        if verbose
            println("  Top 5 eigenvalues: ", eigenvals_sorted[1:min(5, length(eigenvals_sorted))])
            println("  Eigenvalue closest to 1: ", real(λ_closest))
        end
        
        return real(λ_closest), eigenvec
        
    catch e
        @warn "Failed to solve eigenvalue problem at E0 = $E0: $e"
        return NaN, nothing
    end
end

"""
    malfiet_tjon_solve(α, grid, potname, e2b; E0=-8.0, E1=-7.0, 
                      tolerance=1e-6, max_iterations=100, verbose=true)

Solve the Faddeev equation using the Malfiet-Tjon method with secant iteration.

The method solves: λ(E) [c] = [E*B - H0 - V]⁻¹ R [c]
where the ground state energy satisfies λ(E_gs) = 1.

# Arguments
- `α`: Channel structure from channels module
- `grid`: Mesh structure from mesh module  
- `potname`: Nuclear potential name (e.g., "AV18", "MT")
- `e2b`: Two-body threshold energies
- `E0`: First initial energy guess (default: -8.0 MeV)
- `E1`: Second initial energy guess (default: -7.0 MeV)
- `tolerance`: Convergence tolerance |λ-1| < tolerance (default: 1e-6)
- `max_iterations`: Maximum secant method iterations (default: 100)
- `verbose`: Print iteration details (default: true)

# Returns
- `MalflietTjonResult`: Structure containing converged energy, eigenvalue, and diagnostics

# Method
Uses secant method iteration:
E_{n+1} = E_n - [λ(E_n) - 1] * (E_n - E_{n-1}) / [λ(E_n) - λ(E_{n-1})]

Convergence occurs when |λ(E_n) - 1| < tolerance.
"""
function malfiet_tjon_solve(α, grid, potname, e2b; 
                           E0::Float64=-8.0, E1::Float64=-7.0,
                           tolerance::Float64=1e-6, max_iterations::Int=100,
                           verbose::Bool=true)
    
    if verbose
        println("\n" * "="^70)
        println("         MALFIET-TJON EIGENVALUE SOLVER")
        println("="^70)
        println("Potential: $potname")
        println("Two-body threshold: $(round(e2b[1], digits=6)) MeV")
        println("Initial energy guesses: E0 = $E0 MeV, E1 = $E1 MeV")
        println("Convergence tolerance: $tolerance")
        println("Maximum iterations: $max_iterations")
        println("-"^70)
    end
    
    # Initialize secant method
    E_prev = E0
    E_curr = E1
    
    # Compute initial eigenvalues
    λ_prev, eigenvec_prev = compute_lambda_eigenvalue(E_prev, α, grid, potname, e2b; verbose=verbose)
    λ_curr, eigenvec_curr = compute_lambda_eigenvalue(E_curr, α, grid, potname, e2b; verbose=verbose)
    
    if isnan(λ_prev) || isnan(λ_curr)
        error("Failed to compute initial eigenvalues. Check energy guesses and matrix conditions.")
    end
    
    convergence_history = [(E_prev, λ_prev), (E_curr, λ_curr)]
    
    if verbose
        @printf("Initial: E = %8.4f MeV, λ = %10.6f, |λ-1| = %8.2e\n", E_prev, λ_prev, abs(λ_prev - 1))
        @printf("Initial: E = %8.4f MeV, λ = %10.6f, |λ-1| = %8.2e\n", E_curr, λ_curr, abs(λ_curr - 1))
        println("-"^70)
    end
    
    # Check if already converged
    if abs(λ_curr - 1) < tolerance
        if verbose
            println("Already converged at initial guess!")
        end
        return MalflietTjonResult(E_curr, λ_curr, eigenvec_curr, 0, convergence_history, true)
    end
    
    # Secant method iteration
    converged = false
    final_eigenvec = eigenvec_curr
    
    for iteration in 1:max_iterations
        # Secant method update
        if abs(λ_curr - λ_prev) < 1e-15
            @warn "λ values too close, secant method may be unstable"
            break
        end
        
        # Check for divergence - if energies become too extreme, restart with better guesses
        if abs(E_curr) > 100.0  # Energy magnitude > 100 MeV suggests divergence
            @warn "Method appears to be diverging, try different initial energy guesses"
            break
        end
        
        # Secant formula: E_{n+1} = E_n - (λ_n - 1) * (E_n - E_{n-1}) / (λ_n - λ_{n-1})
        E_next = E_curr - (λ_curr - 1) * (E_curr - E_prev) / (λ_curr - λ_prev)
        
        # Compute new eigenvalue
        λ_next, eigenvec_next = compute_lambda_eigenvalue(E_next, α, grid, potname, e2b; verbose=verbose)
        
        if isnan(λ_next)
            @warn "Eigenvalue calculation failed at E = $E_next, stopping iteration"
            break
        end
        
        # Check convergence
        residual = abs(λ_next - 1)
        push!(convergence_history, (E_next, λ_next))
        
        if verbose
            @printf("Iter %2d: E = %8.4f MeV, λ = %10.6f, |λ-1| = %8.2e\n", 
                   iteration, E_next, λ_next, residual)
        end
        
        if residual < tolerance
            converged = true
            final_eigenvec = eigenvec_next
            if verbose
                println("-"^70)
                println("✓ CONVERGED!")
                @printf("Ground state energy: %10.6f MeV\n", E_next)
                @printf("Final eigenvalue λ:   %10.6f\n", λ_next)
                @printf("Binding energy:      %10.6f MeV\n", -E_next)
                println("="^70)
            end
            return MalflietTjonResult(E_next, λ_next, eigenvec_next, iteration, 
                                     convergence_history, true)
        end
        
        # Update for next iteration
        E_prev = E_curr
        E_curr = E_next
        λ_prev = λ_curr
        λ_curr = λ_next
        final_eigenvec = eigenvec_next
        
        # Safety check for energy bounds
        if E_next > 0
            @warn "Energy became positive ($E_next MeV), may have diverged"
        end
    end
    
    if verbose
        println("-"^70)
        if !converged
            println("✗ Did not converge within $max_iterations iterations")
            @printf("Final: E = %8.4f MeV, λ = %10.6f, |λ-1| = %8.2e\n", 
                   E_curr, λ_curr, abs(λ_curr - 1))
        end
        println("="^70)
    end
    
    return MalflietTjonResult(E_curr, λ_curr, final_eigenvec, max_iterations, 
                             convergence_history, converged)
end

"""
    print_convergence_summary(result::MalflietTjonResult)

Print a summary of the Malfiet-Tjon calculation results.
"""
function print_convergence_summary(result::MalflietTjonResult)
    println("\n" * "="^50)
    println("    MALFIET-TJON CALCULATION SUMMARY")
    println("="^50)
    
    if result.converged
        println("Status: ✓ CONVERGED")
        @printf("Ground state energy: %10.6f MeV\n", result.energy)
        @printf("Binding energy:      %10.6f MeV\n", -result.energy)
        @printf("Final λ eigenvalue:  %10.8f\n", result.eigenvalue)
        @printf("Iterations:          %d\n", result.iterations)
    else
        println("Status: ✗ NOT CONVERGED")
        @printf("Last energy:         %10.6f MeV\n", result.energy)
        @printf("Last λ eigenvalue:   %10.8f\n", result.eigenvalue)
        @printf("Residual |λ-1|:      %10.2e\n", abs(result.eigenvalue - 1))
        @printf("Iterations used:     %d\n", result.iterations)
    end
    
    println("="^50)
end

end # module MalflietTjon