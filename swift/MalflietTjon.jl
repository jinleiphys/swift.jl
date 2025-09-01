module MalflietTjon

include("matrices.jl")
using .matrices
using LinearAlgebra
using Printf
using Random

export malfiet_tjon_solve, compute_lambda_eigenvalue, print_convergence_summary, arnoldi_eigenvalue, compute_position_expectation

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
    arnoldi_step!(Q, H, A, j)

Perform one step of the Arnoldi iteration to build an orthonormal basis
for the Krylov subspace span{v, Av, A²v, ...}.

Uses modified Gram-Schmidt for better numerical stability.

# Arguments
- `Q`: Matrix storing orthonormal basis vectors as columns
- `H`: Upper Hessenberg matrix storing projection coefficients
- `A`: Linear operator (function that takes a vector and returns A*vector)
- `j`: Current iteration index (1-based)

# Returns
- `β`: Residual norm for breakdown detection
"""
function arnoldi_step!(Q, H, A, j)
    n = size(Q, 1)
    
    # Apply operator A to current basis vector
    w = A(Q[:, j])
    
    # Modified Gram-Schmidt orthogonalization for better numerical stability
    for i in 1:j
        H[i, j] = dot(Q[:, i], w)
        w = w - H[i, j] * Q[:, i]  # Use explicit subtraction for clarity
    end
    
    # Compute residual norm
    β = norm(w)
    H[j+1, j] = β
    
    # Normalize new basis vector (if not breakdown)
    if β > 1e-12 && j < size(Q, 2)
        Q[:, j+1] = w / β
    end
    
    return β
end

"""
    arnoldi_eigenvalue(A, v0, m; tol=1e-10, maxiter=300)

Compute the largest eigenvalue of operator A using the Arnoldi method.
This is specifically designed for the Faddeev kernel K(E) = [EB - H0 - V]⁻¹ R.

# Arguments
- `A`: Linear operator function A(x) that returns A*x
- `v0`: Initial vector (normalized)
- `m`: Krylov subspace dimension
- `tol`: Convergence tolerance for eigenvalue
- `maxiter`: Maximum number of restart iterations

# Returns
- `λ`: Dominant eigenvalue
- `v`: Corresponding eigenvector
- `converged`: Convergence flag
- `iterations`: Number of iterations used
"""
function arnoldi_eigenvalue(A, v0, m; tol=1e-6, maxiter=300, verbose_arnoldi=false)
    n = length(v0)
    
    # Ensure starting vector is normalized
    v = v0 / norm(v0)
    
    # Test how good our initial guess is and possibly skip Arnoldi entirely
    Av0 = A(v)
    λ_guess = dot(v, Av0) / dot(v, v)  # Rayleigh quotient
    residual = norm(Av0 - λ_guess * v)
    
    # Only show initial guess quality in very verbose mode (disabled by default)
    # if verbose_arnoldi
    #     println("    Initial guess quality: λ₀ = $(real(λ_guess)), residual = $(residual)")
    # end
    
    # If initial guess is excellent, return immediately (micro-optimization)
    # Use a reasonable threshold: if residual is better than what we'd typically achieve in a few Arnoldi steps
    residual_threshold = max(tol * 1000, 1e-4)  # Adaptive threshold, but at least 1e-4
    if residual < residual_threshold
        # Silent return for excellent guesses
        return real(λ_guess), v, true, 0
    end
    
    # For very good guesses (residual < 0.01), use smaller Krylov dimension
    # For moderately good guesses (0.01 < residual < 0.05), use moderate reduction
    # This prevents excessive iterations while ensuring convergence
    original_m = m
    if residual < 0.01 && m > 15
        m = min(15, m)  # Small reduction for very good guesses
    elseif residual < 0.05 && m > 25
        m = min(25, m)  # Moderate reduction for good guesses
    end
    
    λ_old = 0.0
    converged = false
    total_iterations = 0
    
    # Reduce number of restarts only for very good initial guesses
    max_restarts = if residual < 0.01
        min(5, maxiter)  # Reduce restarts only for very good guesses
    else
        maxiter  # Keep full restarts for moderate/poor guesses
    end
    
    for restart in 1:max_restarts
        # Remove restart diagnostics to clean up output
        
        # Initialize Krylov subspace matrices
        Q = zeros(ComplexF64, n, m+1)  # Orthonormal basis
        H = zeros(ComplexF64, m+1, m)   # Upper Hessenberg matrix
        
        Q[:, 1] .= v
        
        # Build Krylov subspace using Arnoldi iteration
        breakdown = false
        actual_m = m
        
        for j in 1:m
            β = arnoldi_step!(Q, H, A, j)
            total_iterations += 1
            
            # Early convergence check - compute Ritz values periodically
            if j >= 3 && j % 2 == 0  # Check every 2 iterations starting from iteration 3
                H_temp = H[1:j, 1:j]
                eigenvals_temp, _ = eigen(H_temp)
                λ_temp = maximum(real.(eigenvals_temp))
                
                if verbose_arnoldi
                    println("    Arnoldi iter $j: λ_temp = $(real(λ_temp)), λ_old = $λ_old, diff = $(abs(real(λ_temp) - λ_old))")
                end
                
                if abs(real(λ_temp) - λ_old) < tol && j >= 5  # Allow early exit after 5 iterations
                    if verbose_arnoldi
                        println("    Early convergence at iteration $j")
                    end
                    actual_m = j
                    break
                end
                λ_old = real(λ_temp)  # Update for next comparison
            end
            
            # Very early convergence check for excellent starting vectors
            if j == 1
                H_temp = H[1:1, 1:1]
                λ_temp = real(H_temp[1,1])
                if verbose_arnoldi
                    println("    Arnoldi iter 1: λ_temp = $λ_temp, λ_old = $λ_old, diff = $(abs(λ_temp - λ_old))")
                end
                
                if abs(λ_temp - λ_old) < tol * 10  # Slightly relaxed for first iteration
                    if verbose_arnoldi
                        println("    Immediate convergence at iteration 1")
                    end
                    actual_m = 1
                    break
                end
                λ_old = λ_temp  # Update for next comparison
            end
            
            # Check for breakdown (happy breakdown)
            if β < 1e-12
                actual_m = j
                breakdown = true
                # Silent breakdown detection
                break
            end
        end
        
        # Extract the actual Hessenberg matrix for eigenvalue computation
        H_reduced = H[1:actual_m, 1:actual_m]
        
        # Solve the reduced eigenvalue problem
        eigenvals, eigenvecs = eigen(H_reduced)
        
        # Find largest eigenvalue (ground state for Malfiet-Tjon method)
        largest_idx = argmax(real.(eigenvals))
        λ = eigenvals[largest_idx]
        y = eigenvecs[:, largest_idx]  # Eigenvector in Krylov basis
        
        # Transform back to original space
        v = Q[:, 1:actual_m] * y
        v = v / norm(v)  # Renormalize
        
        # If we had a breakdown, we found the exact answer - return immediately
        if breakdown
            converged = true
            # Silent return on breakdown
            return real(λ), v, converged, total_iterations
        end
        
        # Check convergence for non-breakdown case
        if abs(λ - λ_old) < tol
            converged = true
            return real(λ), v, converged, total_iterations
        end
        
        λ_old = real(λ)
        
        # Update starting vector for next restart (if no breakdown occurred)
        # v is already updated above
    end
    
    # If we didn't converge and we used reduced parameters, try again with full parameters
    if !converged && (m < original_m || max_restarts < maxiter)
        # Silent retry with full parameters
        
        # Retry with original parameters
        return arnoldi_eigenvalue(A, v0, original_m; tol=tol, maxiter=maxiter, verbose_arnoldi=verbose_arnoldi)
    end
    
    return real(λ_old), v, converged, total_iterations
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

Uses the Arnoldi method for efficient eigenvalue computation of the Faddeev kernel.
Returns the eigenvalue closest to 1 and corresponding eigenvector.
"""
function compute_lambda_eigenvalue(E0::Float64, T, V, B, Rxy; 
                                  verbose::Bool=false, use_arnoldi::Bool=true,
                                  krylov_dim::Int=50, arnoldi_tol::Float64=1e-6,
                                  previous_eigenvector::Union{Nothing, Vector}=nothing)
    
    # Form the left-hand side operator: E0*B - H0 - V
    # where H0 = T (kinetic energy) and V is the potential
    LHS = E0 * B - T - V
    
    # # Check if matrix is singular
    # cond_num = cond(LHS)
    # if cond_num > 1e12
    #     if verbose
    #         @warn "Matrix [E0*B - T - V] is near-singular at E0 = $E0, condition number = $(cond_num)"
    #     end
    #     return NaN, nothing
    # end
    
    # The Faddeev kernel: K(E) = [E*B - T - V]⁻¹ * V*R
    VRxy = V * Rxy
    
    try
        if use_arnoldi
            # Precompute the matrix solve once for efficiency
            # K(E) = [E*B - T - V]⁻¹ * V*R
            if verbose
                print("  Computing RHS = LHS \\ VRxy... ")
                @time RHS = LHS \ VRxy  # Single expensive factorization
            else
                RHS = LHS \ VRxy  # Single expensive factorization
            end
            
            # Define the linear operator K(E) as a function using precomputed matrix
            K = function(x)
                return RHS * x  # Fast matrix-vector multiplication
            end
            
            # Generate initial vector with better conditioning
            n = size(LHS, 1)
            
            # Use multiple strategies for initial vector to improve robustness
            v0_strategies = []
            
            # Strategy 1: Use previous eigenvector if available (best convergence)
            if previous_eigenvector !== nothing && length(previous_eigenvector) == n
                push!(v0_strategies, () -> begin
                    v = ComplexF64.(previous_eigenvector)
                    v / norm(v)
                end)
            end
            
            # Strategy 2: Random Gaussian vector
            push!(v0_strategies, () -> begin
                Random.seed!(42)  # For reproducible results
                v = randn(ComplexF64, n)
                v / norm(v)
            end)
            
            
            λ, eigenvec, converged, iterations = NaN, nothing, false, 0
            
            # Try different initial vectors if first attempt fails
            for (i, strategy) in enumerate(v0_strategies)
                v0 = strategy()
                strategy_name = if i == 1 && previous_eigenvector !== nothing
                    "previous eigenvector"
                else
                    "random Gaussian"
                end
                
                # Adaptive Krylov dimension based on starting vector quality
                adaptive_krylov_dim = if i == 1 && previous_eigenvector !== nothing
                    # For previous eigenvector, use much smaller subspace
                    min(15, krylov_dim)
                else
                    # For random starting vector, use full dimension
                    krylov_dim
                end
                
                try
                    # Use Arnoldi method to find dominant eigenvalue
                    λ, eigenvec, converged, iterations = arnoldi_eigenvalue(K, v0, adaptive_krylov_dim; 
                                                                           tol=arnoldi_tol, maxiter=10,
                                                                           verbose_arnoldi=true)
                    
                    # Check if result is reasonable
                    if !isnan(λ) && isfinite(λ)
                        if verbose
                            if iterations == 0
                                println("  Arnoldi: instant (excellent guess)")
                            elseif iterations <= 5
                                println("  Arnoldi: $iterations iterations (very fast)")
                            elseif iterations <= 15
                                println("  Arnoldi: $iterations iterations (fast)")
                            else
                                println("  Arnoldi: $iterations iterations")
                            end
                        end
                        break
                    end
                catch e
                    if verbose && i == length(v0_strategies)
                        @warn "Arnoldi method failed with all initial vector strategies: $e"
                    end
                    continue
                end
            end
            
            if !converged && verbose
                @warn "Arnoldi method did not converge, result may be inaccurate"
            end
            
            # If Arnoldi failed completely, fall back to direct method
            if isnan(λ) || !isfinite(λ)
                if verbose
                    @warn "Arnoldi method failed, falling back to direct eigenvalue computation"
                end
                use_arnoldi = false  # Force fallback
            else
                return λ, eigenvec
            end
            
        else
            # Fallback to direct eigenvalue computation
            RHS = LHS \ VRxy
            
            # Find the largest eigenvalue and corresponding eigenvector
            eigenvals, eigenvecs = eigen(RHS)
            
            # For debugging: let's look at all eigenvalues
            eigenvals_real = real.(eigenvals)
            eigenvals_sorted = sort(eigenvals_real, rev=true)  # Sort descending
            
            # The ground state should correspond to the largest eigenvalue
            # Find largest eigenvalue
            largest_idx = argmax(eigenvals_real)
            λ_largest = eigenvals[largest_idx]
            eigenvec = eigenvecs[:, largest_idx]
            
            if verbose
                println("  Direct method used")
                println("  Top 5 eigenvalues: ", eigenvals_sorted[1:min(5, length(eigenvals_sorted))])
                println("  Largest eigenvalue: ", real(λ_largest))
            end
            
            return real(λ_largest), eigenvec
        end
        
    catch e
        @warn "Failed to solve eigenvalue problem at E0 = $E0: $e"
        return NaN, nothing
    end
end

"""
    malfiet_tjon_solve(α, grid, potname, e2b; E0=-8.0, E1=-7.0, 
                      tolerance=1e-6, max_iterations=100, verbose=true,
                      use_arnoldi=true, krylov_dim=50, arnoldi_tol=1e-10)

Solve the Faddeev equation using the Malfiet-Tjon method with secant iteration
and efficient Arnoldi eigenvalue computation.

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
- `use_arnoldi`: Use Arnoldi method for eigenvalue computation (default: true)
- `krylov_dim`: Krylov subspace dimension for Arnoldi (default: 50)
- `arnoldi_tol`: Arnoldi convergence tolerance (default: 1e-10)

# Returns
- `MalflietTjonResult`: Structure containing converged energy, eigenvalue, and diagnostics

# Method
Uses secant method iteration:
E_{n+1} = E_n - [λ(E_n) - 1] * (E_n - E_{n-1}) / [λ(E_n) - λ(E_{n-1})]

The Arnoldi method efficiently computes eigenvalues of the Faddeev kernel
K(E) = [EB - H₀ - V]⁻¹ R using Krylov subspace projection, avoiding
expensive full matrix diagonalization.

Convergence occurs when |λ(E_n) - 1| < tolerance.
"""
function malfiet_tjon_solve(α, grid, potname, e2b; 
                           E0::Float64=-8.0, E1::Float64=-7.0,
                           tolerance::Float64=1e-6, max_iterations::Int=100,
                           verbose::Bool=true, use_arnoldi::Bool=true,
                           krylov_dim::Int=50, arnoldi_tol::Float64=1e-6)
    
    if verbose
        println("\n" * "="^70)
        println("         MALFIET-TJON EIGENVALUE SOLVER")
        println("="^70)
        println("Potential: $potname")
        println("Two-body threshold: $(round(e2b[1], digits=6)) MeV")
        println("Initial energy guesses: E0 = $E0 MeV, E1 = $E1 MeV")
        println("Convergence tolerance: $tolerance")
        println("Maximum iterations: $max_iterations")
        method_str = use_arnoldi ? "Arnoldi (Krylov dim: $krylov_dim)" : "Direct diagonalization"
        println("Eigenvalue method: $method_str")
        if use_arnoldi
            println("Arnoldi tolerance: $arnoldi_tol")
        end
        println("-"^70)
    end
    
    # Pre-compute matrices once (they don't change between iterations)
    if verbose
        print("  Building matrices... ")
        @time begin
            T = T_matrix(α, grid)           # Kinetic energy H0
            V = V_matrix(α, grid, potname)  # Potential energy
            B = Bmatrix(α, grid)            # Overlap matrix
            Rxy = Rxy_matrix(α, grid)       # Rearrangement matrix R
        end
    else
        T = T_matrix(α, grid)           # Kinetic energy H0
        V = V_matrix(α, grid, potname)  # Potential energy
        B = Bmatrix(α, grid)            # Overlap matrix
        Rxy = Rxy_matrix(α, grid)       # Rearrangement matrix R
    end
    
    # Initialize secant method
    E_prev = E0
    E_curr = E1
    
    # Compute initial eigenvalues (no previous eigenvector for first iteration)
    λ_prev, eigenvec_prev = compute_lambda_eigenvalue(E_prev, T, V, B, Rxy; 
                                                     verbose=verbose, use_arnoldi=use_arnoldi,
                                                     krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol)
    # Use previous eigenvector for second initial guess  
    λ_curr, eigenvec_curr = compute_lambda_eigenvalue(E_curr, T, V, B, Rxy; 
                                                     verbose=verbose, use_arnoldi=use_arnoldi,
                                                     krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                     previous_eigenvector=eigenvec_prev)
    
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
        
        # Compute new eigenvalue using previous eigenvector as starting point
        λ_next, eigenvec_next = compute_lambda_eigenvalue(E_next, T, V, B, Rxy; 
                                                         verbose=verbose, use_arnoldi=use_arnoldi,
                                                         krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                         previous_eigenvector=eigenvec_curr)
        
        if isnan(λ_next)
            @warn "Eigenvalue calculation failed at E = $E_next, stopping iteration"
            break
        end
        
        # Check convergence
        residual = abs(λ_next - 1)
        push!(convergence_history, (E_next, λ_next))
        
        if verbose
            @printf("Iter %2d: E = %8.4f MeV, λ = %10.6f, |λ-1| = %8.2e", 
                   iteration, E_next, λ_next, residual)
            # Add Arnoldi performance indicator
            if eigenvec_next !== nothing
                # Get the previous eigenvector computation info if available
                println()  # Just the basic iteration info
            else
                println()
            end
        end
        
        if residual < tolerance
            converged = true
            final_eigenvec = eigenvec_next
            if verbose
                # Normalize the eigenvector properly with respect to the overlap matrix B
                ψ = eigenvec_next
                norm_factor = sqrt(real(ψ' * B * ψ))
                ψ_normalized = ψ / norm_factor
                
                # Compute individual expectation values
                T_expectation = real((ψ_normalized' * T * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
                V_expectation = real((ψ_normalized' * V * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
                VRxy_expectation = real((ψ_normalized' * (V * Rxy) * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
                
                # Total Hamiltonian expectation value
                H_expectation = T_expectation + V_expectation + VRxy_expectation
                
                # Compute position expectation values and RMS radii
                r_x_avg, r_y_avg, rho_avg, rms_x, rms_y, rms_rho = compute_position_expectation(ψ_normalized, α, grid, B)
                
                println("-"^70)
                println("✓ CONVERGED!")
                @printf("Ground state energy: %10.6f MeV\n", E_next)
                @printf("Final eigenvalue λ:   %10.6f\n", λ_next)
                @printf("Binding energy:      %10.6f MeV\n", -E_next)
                println()
                println("Expectation values:")
                @printf("  <ψ|T|ψ>/<ψ|B|ψ>      = %10.6f MeV\n", T_expectation)
                @printf("  <ψ|V|ψ>/<ψ|B|ψ>      = %10.6f MeV\n", V_expectation)
                @printf("  <ψ|V*Rxy|ψ>/<ψ|B|ψ>  = %10.6f MeV\n", VRxy_expectation)
                @printf("  <ψ|H|ψ>/<ψ|B|ψ>      = %10.6f MeV\n", H_expectation)
                println()
                println("Position expectation values:")
                @printf("  ⟨x⟩   = %8.4f fm  (Jacobi coordinate for particles 1,2)\n", r_x_avg)
                @printf("  ⟨y⟩   = %8.4f fm  (Jacobi coordinate for particle 3)\n", r_y_avg)
                @printf("  ⟨ρ⟩   = %8.4f fm  (hyperradius √(x²+y²))\n", rho_avg)
                println()
                println("RMS radii:")
                @printf("  √⟨x²⟩ = %8.4f fm  (RMS radius in x-direction)\n", rms_x)
                @printf("  √⟨y²⟩ = %8.4f fm  (RMS radius in y-direction)\n", rms_y)
                @printf("  √⟨ρ²⟩ = %8.4f fm  (RMS hyperradius - overall nuclear size)\n", rms_rho)
                println()
                @printf("Energy difference:   %10.6f MeV\n", abs(H_expectation - E_next))
                if abs(H_expectation - E_next) < 1e-3
                    println("✓ Energy consistency check: PASSED")
                else
                    println("⚠ Energy consistency check: FAILED")
                end
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

"""
    compute_position_expectation(eigenvector, α, grid, B)

Compute position expectation values and RMS radii for the three-body wave function.

Calculates:
- ⟨x⟩, ⟨y⟩, ⟨ρ⟩: expectation values of coordinates
- √⟨x²⟩, √⟨y²⟩, √⟨ρ²⟩: RMS radii

# Arguments
- `eigenvector`: Ground state wave function from Malfiet-Tjon solver
- `α`: Channel structure
- `grid`: Mesh structure containing coordinate grids
- `B`: Overlap matrix

# Returns
- `(r_x_avg, r_y_avg, rho_avg, rms_x, rms_y, rms_rho)`: Expectation values and RMS radii in fm
"""
function compute_position_expectation(eigenvector::Vector, α, grid, B)
    # Normalize the wave function with respect to the overlap matrix B
    ψ = eigenvector
    norm_factor = sqrt(real(ψ' * B * ψ))
    ψ_normalized = ψ / norm_factor
    
    # Create position operator matrices
    n_total = α.nchmax * grid.nx * grid.ny
    
    # First and second moment operator matrices (diagonal in coordinate representation)
    X_matrix = zeros(Float64, n_total, n_total)
    Y_matrix = zeros(Float64, n_total, n_total)
    Rho_matrix = zeros(Float64, n_total, n_total)
    X2_matrix = zeros(Float64, n_total, n_total)
    Y2_matrix = zeros(Float64, n_total, n_total)
    Rho2_matrix = zeros(Float64, n_total, n_total)
    
    for iα in 1:α.nchmax
        for ix in 1:grid.nx
            x_val = grid.xi[ix]  # x-coordinate value
            for iy in 1:grid.ny
                y_val = grid.yi[iy]  # y-coordinate value
                rho_val = sqrt(x_val^2 + y_val^2)  # hyperradius
                
                # Linear index for this state |α,ix,iy⟩
                i = (iα-1) * grid.nx * grid.ny + (ix-1) * grid.ny + iy
                
                # Position operators (first moments)
                X_matrix[i, i] = x_val
                Y_matrix[i, i] = y_val
                Rho_matrix[i, i] = rho_val
                
                # Position squared operators (second moments)
                X2_matrix[i, i] = x_val^2
                Y2_matrix[i, i] = y_val^2
                Rho2_matrix[i, i] = rho_val^2
            end
        end
    end
    
    # Compute expectation values ⟨r⟩
    r_x_avg = real((ψ_normalized' * X_matrix * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
    r_y_avg = real((ψ_normalized' * Y_matrix * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
    rho_avg = real((ψ_normalized' * Rho_matrix * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
    
    # Compute second moments ⟨r²⟩
    r_x2_avg = real((ψ_normalized' * X2_matrix * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
    r_y2_avg = real((ψ_normalized' * Y2_matrix * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
    rho2_avg = real((ψ_normalized' * Rho2_matrix * ψ_normalized) / (ψ_normalized' * B * ψ_normalized))
    
    # Compute RMS radii √⟨r²⟩
    rms_x = sqrt(r_x2_avg)
    rms_y = sqrt(r_y2_avg)
    rms_rho = sqrt(rho2_avg)
    
    return (r_x_avg, r_y_avg, rho_avg, rms_x, rms_y, rms_rho)
end

end # module MalflietTjon