module MalflietTjon

include("matrices.jl")
using .matrices
using LinearAlgebra
using SparseArrays
using Printf
using Random
using Statistics
include("matrices_optimized.jl")
using .matrices_optimized

# Load UIX modules once at module initialization
include("../3Npot/UIX.jl")
include("../3Npot/UIX_optimized.jl")

export malfiet_tjon_solve, malfiet_tjon_solve_optimized, compute_lambda_eigenvalue, compute_lambda_eigenvalue_optimized, print_convergence_summary, arnoldi_eigenvalue, compute_position_expectation, compute_channel_probabilities, RHSCache, precompute_RHS_cache, compute_uix_potential, compute_uix_potential_optimized

"""
    compute_uix_potential(α, grid, Rxy_31, Rxy)

Helper function to compute UIX three-body potential.
This function handles loading the UIX module and computing the potential.
"""
function compute_uix_potential(α, grid, Rxy_31, Rxy)
    # UIX module already loaded at module initialization
    return UIX.full_UIX_potential(α, grid, Rxy_31, Rxy)
end

"""
    compute_uix_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)

Optimized helper function to compute UIX three-body potential.

This function uses the optimized UIX implementation with:
- Cached radial functions (Y, T)
- Cached Wigner symbols and S-matrix elements
- Hybrid sparse/dense matrix operations
- Pre-computed G-coefficients

# Arguments
- `α`: Channel structure
- `grid`: Mesh grid structure
- `Rxy_31`: Rearrangement matrix α₃ → α₁
- `Rxy`: Combined rearrangement matrix (Rxy_31 + Rxy_32)
- `Gαα`: Pre-computed G-coefficients (to avoid recomputation)

# Returns
- UIX three-body potential matrix (optimized, 2-3x faster than standard version)

# Performance
Expected ~2-3x speedup compared to `compute_uix_potential`, with the benefit
increasing for larger grid sizes.

# Example
```julia
# Inside malfiet_tjon_solve_optimized:
Gαα = computeGcoefficient(α, grid)
V_UIX = compute_uix_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)
```
"""
function compute_uix_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)
    # UIX_optimized module already loaded at module initialization
    return UIX_optimized.full_UIX_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)
end

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
    RHSCache

Structure to cache energy-independent RHS matrix components.

This caches the expensive matrix operations that don't depend on energy E:
- V_diag: Block-diagonal potential matrix (V_αα)
- RHS_matrix: Precomputed (V - V_αα) + V*R + UIX

# Fields
- `RHS_matrix`: Precomputed RHS = (V - V_αα) + V*R (or + UIX if included)
                Can be real (Float64) or complex (Complex{Float64}) for complex scaling
- `n_total`: Total dimension (nα × nx × ny)
"""
struct RHSCache
    RHS_matrix::Union{Matrix{Float64}, Matrix{ComplexF64}}
    n_total::Int
end

"""
    precompute_RHS_cache(V, V_x_diag_ch, Rxy, α, grid; V_UIX=nothing)

Precompute energy-independent RHS matrix: RHS = (V - V_αα) + V*R (+ UIX).

This is the single most expensive operation that's currently repeated for every energy,
taking ~1.0-1.5 seconds. By caching it, we eliminate this cost entirely.

# Arguments
- `V`: Full potential matrix (nα*nx*ny × nα*nx*ny)
- `V_x_diag_ch`: Vector of diagonal potential matrices for each channel (from V_matrix)
- `Rxy`: Rearrangement matrix (nα*nx*ny × nα*nx*ny)
- `α`: Channel structure
- `grid`: Mesh structure
- `V_UIX`: Optional UIX three-body force matrix

# Returns
- `RHSCache`: Cached RHS matrix for reuse

# Performance
- Computation time: ~1.0-1.5 seconds (one-time cost)
- Replaces: 1.0-1.5 seconds per energy evaluation
- Speedup: Eliminates 30-40% of per-energy cost

# Example
```julia
# After computing V, V_x_diag_ch, and Rxy:
RHS_cache = precompute_RHS_cache(V, V_x_diag_ch, Rxy, α, grid; V_UIX=V_UIX)

# Then use cached RHS for all energy evaluations
λ, eigenvec = compute_lambda_eigenvalue_optimized(...; RHS_cache=RHS_cache)
```
"""
function precompute_RHS_cache(V, V_x_diag_ch, Rxy, α, grid; V_UIX=nothing)
    nα = α.nchmax
    nx = grid.nx
    ny = grid.ny
    n_total = nα * nx * ny

    # SPARSE OPTIMIZATION: V is 99.7% sparse! Convert to sparse format for massive speedup
    # Check sparsity and convert if beneficial
    n_elements = n_total * n_total
    n_nonzero = count(!iszero, V)
    sparsity = 100 * (1 - n_nonzero / n_elements)

    if sparsity > 90.0
        # Convert to sparse matrix (V is 99.7% sparse → huge speedup!)
        V_sparse = sparse(V)
        println("  V matrix converted to sparse ($(round(sparsity, digits=1))% zeros, $(n_nonzero) nonzeros)")
        V_compute = V_sparse
    else
        V_compute = V
    end

    # Step 1: Compute V*Rxy using sparse matrix multiplication
    # Sparse-Dense multiplication is MUCH faster than Dense-Dense!
    VRxy = V_compute * Rxy

    # Step 2: Start with RHS = V + V*Rxy (single allocation + in-place addition)
    RHS_matrix = V + VRxy  # Allocate once and compute sum

    # Step 3: Add UIX if provided (in-place)
    if V_UIX !== nothing
        RHS_matrix .+= V_UIX
    end

    # Step 4: Subtract V_αα WITHOUT kron (avoid massive temporary allocations!)
    # V_αα[iα, iα] = V_x_diag_ch[iα] ⊗ I_y means:
    # Element (i,j) in channel block iα = V_x[ix, jx] * δ_{iy, jy}
    for iα in 1:nα
        block_start = (iα-1) * nx * ny

        # Subtract V_αα elements directly without building the full block
        for ix in 1:nx, jx in 1:nx
            V_x_val = V_x_diag_ch[iα][ix, jx]
            i_base = block_start + (ix-1)*ny
            j_base = block_start + (jx-1)*ny

            # Kronecker product with I_y: only subtract on y-diagonal
            @simd for iy in 1:ny
                @inbounds RHS_matrix[i_base + iy, j_base + iy] -= V_x_val
            end
        end
    end

    return RHSCache(RHS_matrix, n_total)
end

"""
    GMRESResult

Summary of a GMRES solve.

# Fields
- `converged::Bool`: Whether the residual tolerance was met
- `iterations::Int`: Total Krylov iterations performed
- `residual_norm::Float64`: Final residual ‖A*x - b‖
- `rel_residual::Float64`: Relative residual ‖A*x - b‖ / ‖b‖
"""
struct GMRESResult
    converged::Bool
    iterations::Int
    residual_norm::Float64
    rel_residual::Float64
end

"""
    gmres_matfree(A, b; M=nothing, x0=nothing, abstol=1e-10, reltol=1e-10,
                  maxiter=200, restart=50, verbose=false)

Restarted GMRES for matrix-free operators with optional left preconditioning.

# Arguments
- `A`: Linear operator. Can be a matrix or a function `v -> A*v`.
- `b`: Right-hand side vector.
- `M`: Optional left preconditioner (matrix or function). The solver
        works with the transformed system `M*(A*x) = M*b`.
- `x0`: Optional initial guess. Defaults to the zero vector.

# Keyword arguments
- `abstol`: Absolute residual tolerance.
- `reltol`: Relative residual tolerance (based on ‖b‖).
- `maxiter`: Maximum GMRES iterations (across all restarts).
- `restart`: Krylov subspace dimension for restarts.
- `verbose`: Print basic convergence information when `true`.

# Returns
- `x`: Approximate solution to `A*x = b`.
- `info::GMRESResult`: Convergence summary.
"""
function gmres_matfree(A, b;
                       M=nothing,
                       x0=nothing,
                       abstol::Float64=1e-10,
                       reltol::Float64=1e-10,
                       maxiter::Int=200,
                       restart::Int=50,
                       verbose::Bool=false)
    maxiter <= 0 && throw(ArgumentError("maxiter must be positive"))
    restart <= 0 && throw(ArgumentError("restart must be positive"))

    A_op = A isa AbstractMatrix ? (v -> A * v) : A
    M_op = M === nothing ? nothing : (M isa AbstractMatrix ? (v -> M * v) : M)

    rhs_orig = M_op === nothing ? copy(b) : M_op(b)
    T = promote_type(eltype(rhs_orig), eltype(b))
    rhs = Vector{T}(rhs_orig)
    b_vec = Vector{T}(b)
    n = length(rhs)
    rhs_norm = norm(b)

    x = if x0 === nothing
        zeros(T, n)
    else
        length(x0) == n || throw(DimensionMismatch("x0 has incorrect length"))
        convert(Vector{T}, copy(x0))
    end

    apply_operator = M_op === nothing ? (v -> Vector{T}(A_op(v))) : (v -> Vector{T}(M_op(A_op(v))))
    residual = rhs - apply_operator(x)
    raw_residual = b_vec - Vector{T}(A_op(x))
    β = norm(residual)
    β0 = β

    if β == 0.0
        residual_norm = norm(raw_residual)
        rel_residual = rhs_norm > 0 ? residual_norm / rhs_norm : residual_norm
        return x, GMRESResult(true, 0, residual_norm, rel_residual)
    end

    total_iters = 0
    restart = min(restart, maxiter)
    max_restarts = ceil(Int, maxiter / restart)

    for restart_idx in 1:max_restarts
        m = min(restart, maxiter - total_iters)
        m <= 0 && break

        V = zeros(T, n, m + 1)
        H = zeros(T, m + 1, m)

        β = norm(residual)
        if β < abstol || β / β0 < reltol
            raw_residual = b_vec - Vector{T}(A_op(x))
            residual_norm = norm(raw_residual)
            rel_residual = rhs_norm > 0 ? residual_norm / rhs_norm : residual_norm
            return x, GMRESResult(true, total_iters, residual_norm, rel_residual)
        end

        V[:, 1] .= residual / β
        e1 = zeros(T, m + 1)
        e1[1] = β

        actual_m = m

        for j in 1:m
            vj = view(V, :, j)
            w = apply_operator(vj)

            for i in 1:j
                vi = view(V, :, i)
                H[i, j] = dot(vi, w)
                w .-= H[i, j] .* vi
            end

            H[j + 1, j] = norm(w)

            total_iters += 1

            if abs(H[j + 1, j]) ≤ 1e-14
                actual_m = j
                break
            elseif j < m
                V[:, j + 1] .= w / H[j + 1, j]
            end
        end

        H_reduced = view(H, 1:actual_m + 1, 1:actual_m)
        V_reduced = view(V, :, 1:actual_m)
        e1_reduced = view(e1, 1:actual_m + 1)

        y = H_reduced \ e1_reduced
        x .+= V_reduced * y

        residual = rhs - apply_operator(x)
        β = norm(residual)

        if verbose
            println("    GMRES restart $restart_idx: residual=$(β)")
        end

        if β < abstol || β / β0 < reltol
            raw_residual = b_vec - Vector{T}(A_op(x))
            residual_norm = norm(raw_residual)
            rel_residual = rhs_norm > 0 ? residual_norm / rhs_norm : residual_norm
            return x, GMRESResult(true, total_iters, residual_norm, rel_residual)
        end

        if total_iters >= maxiter
            break
        end
    end

    raw_residual = b_vec - Vector{T}(A_op(x))
    residual_norm = norm(raw_residual)
    rel_residual = rhs_norm > 0 ? residual_norm / rhs_norm : residual_norm
    return x, GMRESResult(false, total_iters, residual_norm, rel_residual)
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
        # IMPORTANT: For complex scaling, only check REAL part convergence
        # The imaginary part can be non-zero and prevents convergence otherwise
        if abs(real(λ) - λ_old) < tol  # λ_old is already real
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
function compute_lambda_eigenvalue(E0::Float64, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                  verbose::Bool=false, use_arnoldi::Bool=true,
                                  krylov_dim::Int=50, arnoldi_tol::Float64=1e-6,
                                  previous_eigenvector::Union{Nothing, Vector}=nothing,
                                  V_UIX=nothing)
    
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

    # The Faddeev kernel: K(E) = [E*B - T - V]⁻¹ * (V*R + UIX)
    VRxy = V * Rxy

    # Add UIX three-body force to the right-hand side if provided
    if V_UIX !== nothing
        VRxy = VRxy + V_UIX  # K(E) = [E*B - T - V]⁻¹ * (V*R + UIX)
    end
    
    try
        if use_arnoldi
            # Precompute the full RHS matrix: K(E) = [E*B - T - V]⁻¹ * (V*R + UIX)
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
    compute_lambda_eigenvalue_optimized(E0, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                       verbose=false, use_arnoldi=true, krylov_dim=50, arnoldi_tol=1e-6,
                                       previous_eigenvector=nothing, V_UIX=nothing)

**OPTIMIZED VERSION** using M⁻¹ preconditioner for faster eigenvalue computation.

This function computes the eigenvalue λ(E0) for the reformulated Faddeev equation:
    λ(E0) [c] = M⁻¹(E0) * (V - V_αα + V*R + UIX) [c]

where:
- M⁻¹ = [E0*B - T - V_αα]⁻¹ is the preconditioner (diagonal potential only)
- V_αα is the diagonal part of the potential (within channels)
- V - V_αα is the off-diagonal part (channel coupling)

# Key Differences from Original
1. **Preconditioner**: Uses M⁻¹ = [E*B - T - V_αα]⁻¹ instead of [E*B - T - V]⁻¹
2. **RHS formulation**: K(E) = M⁻¹ * (V - V_αα + V*R + UIX)
3. **Better conditioning**: M⁻¹ is cheaper to compute (diagonal potential only)
4. **Physical interpretation**: Separates diagonal from off-diagonal channel coupling

# Physics
The reformulation separates the potential into diagonal and off-diagonal parts:
- V_αα: Within-channel interaction (included in preconditioner)
- V - V_αα: Channel-channel coupling (treated as perturbation)

This is similar to the M⁻¹ preconditioner used in GMRES, but applied to eigenvalue problem.

# Arguments
Same as `compute_lambda_eigenvalue()`, see that function for detailed documentation.

# Returns
- `λ`: Dominant eigenvalue
- `eigenvec`: Corresponding eigenvector
- `arnoldi_iters`: Number of Arnoldi iterations performed

# Performance Benefits
- Faster LHS inversion (M⁻¹ has diagonal potential only, not full V)
- Better numerical conditioning
- More efficient when V has strong off-diagonal coupling
"""
function compute_lambda_eigenvalue_optimized(E0::Float64, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                            verbose::Bool=false, use_arnoldi::Bool=true,
                                            krylov_dim::Int=50, arnoldi_tol::Float64=1e-6,
                                            previous_eigenvector::Union{Nothing, Vector}=nothing,
                                            V_UIX=nothing,
                                            M_cache::Union{Nothing, matrices.MInverseCache}=nothing,
                                            RHS_cache::Union{Nothing, RHSCache}=nothing)

    # Compute M⁻¹ = [E0*B - T - V_αα]⁻¹ using the efficient implementation
    # Note: V_x_diag_ch contains the diagonal potential components V_αα
    if M_cache !== nothing
        # Use cached version (FAST!)
        if verbose
            print("  Computing M⁻¹ preconditioner (cached)... ")
            @time M_inv_op = matrices.M_inverse_operator_cached(E0, M_cache)
        else
            M_inv_op = matrices.M_inverse_operator_cached(E0, M_cache)
        end
    else
        # Fallback to non-cached version (SLOW)
        if verbose
            print("  Computing M⁻¹ preconditioner (non-cached)... ")
            @time M_inv_op = M_inverse_operator(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
        else
            M_inv_op = M_inverse_operator(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
        end
    end

    # Compute RHS matrix: (V - V_αα) + V*R + UIX
    # This is energy-independent and can be cached!
    nα = α.nchmax
    nx = grid.nx
    ny = grid.ny
    n_total = nα * nx * ny

    if RHS_cache !== nothing
        # Use cached RHS matrix (FAST!)
        if verbose
            println("  Using cached RHS matrix (energy-independent)")
        end
        RHS_matrix = RHS_cache.RHS_matrix
    else
        # Build RHS matrix from scratch (SLOW - includes expensive V*Rxy multiplication)
        if verbose
            print("  Building RHS matrix from scratch... ")
            build_time = @elapsed begin
                # Build V_αα as block-diagonal matrix
                V_diag = zeros(n_total, n_total)
                for iα in 1:nα
                    idx_start = (iα-1) * nx * ny + 1
                    idx_end = iα * nx * ny

                    # V_αα for this channel is V_x_diag_ch[iα] ⊗ I_y
                    V_diag_block = kron(V_x_diag_ch[iα], Matrix{Float64}(I, ny, ny))
                    V_diag[idx_start:idx_end, idx_start:idx_end] = V_diag_block
                end

                # Compute the RHS: V - V_αα + V*R + UIX
                V_off_diag = V - V_diag  # Off-diagonal channel coupling
                VRxy = V * Rxy           # Expensive 3600×3600 matrix multiplication!

                # Build the full RHS
                RHS_matrix = V_off_diag + VRxy

                # Add UIX three-body force if provided
                if V_UIX !== nothing
                    RHS_matrix = RHS_matrix + V_UIX
                end
            end
            println("$(round(build_time, digits=3))s")
        else
            # Non-verbose version
            V_diag = zeros(n_total, n_total)
            for iα in 1:nα
                idx_start = (iα-1) * nx * ny + 1
                idx_end = iα * nx * ny
                V_diag_block = kron(V_x_diag_ch[iα], Matrix{Float64}(I, ny, ny))
                V_diag[idx_start:idx_end, idx_start:idx_end] = V_diag_block
            end

            V_off_diag = V - V_diag
            VRxy = V * Rxy
            RHS_matrix = V_off_diag + VRxy

            if V_UIX !== nothing
                RHS_matrix = RHS_matrix + V_UIX
            end
        end
    end

    try
        if use_arnoldi
            # MATRIX-FREE APPROACH: Don't precompute M⁻¹ * RHS matrix
            # Instead, define K as a composition of operators for memory efficiency
            # K(x) = M⁻¹ * RHS_matrix * x (apply operations sequentially)

            # Define the linear operator K(E) as a matrix-free function
            K = function(x)
                # Step 1: Apply RHS matrix to vector
                temp = RHS_matrix * x
                # Step 2: Apply M⁻¹ to the result
                return M_inv_op(temp)
            end

            if verbose
                println("  Using matrix-free operator K(E) = M⁻¹ * (V - V_αα + V*R + UIX)")
            end

            # Generate initial vector
            n = n_total

            # Use multiple strategies for initial vector
            v0_strategies = []

            # Strategy 1: Use previous eigenvector if available
            if previous_eigenvector !== nothing && length(previous_eigenvector) == n
                push!(v0_strategies, () -> begin
                    v = ComplexF64.(previous_eigenvector)
                    v / norm(v)
                end)
            end

            # Strategy 2: Random Gaussian vector
            push!(v0_strategies, () -> begin
                Random.seed!(42)
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

                # Adaptive Krylov dimension
                adaptive_krylov_dim = if i == 1 && previous_eigenvector !== nothing
                    min(15, krylov_dim)
                else
                    krylov_dim
                end

                try
                    # Use Arnoldi method
                    λ, eigenvec, converged, iterations = arnoldi_eigenvalue(K, v0, adaptive_krylov_dim;
                                                                           tol=arnoldi_tol, maxiter=10,
                                                                           verbose_arnoldi=false)

                    # Check if result is reasonable
                    if !isnan(λ) && isfinite(λ)
                        if verbose
                            if iterations == 0
                                println("  Arnoldi (optimized): instant convergence")
                            elseif iterations <= 5
                                println("  Arnoldi (optimized): $iterations iterations (very fast)")
                            elseif iterations <= 15
                                println("  Arnoldi (optimized): $iterations iterations (fast)")
                            else
                                println("  Arnoldi (optimized): $iterations iterations")
                            end
                        end
                        break
                    end
                catch e
                    if verbose && i == length(v0_strategies)
                        @warn "Arnoldi method failed with all strategies: $e"
                    end
                    continue
                end
            end

            if !converged && verbose
                @warn "Arnoldi method (optimized) did not converge"
            end

            # Fallback to direct method if Arnoldi fails
            if isnan(λ) || !isfinite(λ)
                if verbose
                    @warn "Arnoldi failed, falling back to direct method"
                end
                use_arnoldi = false
            else
                return λ, eigenvec
            end
        end

        # Fallback: Direct eigenvalue computation
        if !use_arnoldi
            # Compute M⁻¹ * RHS directly
            RHS_preconditioned = zeros(ComplexF64, n_total, n_total)
            for j in 1:n_total
                RHS_preconditioned[:, j] = M_inv_op(RHS_matrix[:, j])
            end

            # Find eigenvalues
            eigenvals, eigenvecs = eigen(RHS_preconditioned)

            # Get largest eigenvalue
            eigenvals_real = real.(eigenvals)
            largest_idx = argmax(eigenvals_real)
            λ_largest = eigenvals[largest_idx]
            eigenvec = eigenvecs[:, largest_idx]

            if verbose
                println("  Direct method (optimized) used")
                eigenvals_sorted = sort(eigenvals_real, rev=true)
                println("  Top 5 eigenvalues: ", eigenvals_sorted[1:min(5, length(eigenvals_sorted))])
                println("  Largest eigenvalue: ", real(λ_largest))
            end

            return real(λ_largest), eigenvec
        end

    catch e
        @warn "Failed to solve optimized eigenvalue problem at E0 = $E0: $e"
        return NaN, nothing
    end
end

"""
    malfiet_tjon_solve(α, grid, potname, e2b; E0=-8.0, E1=-7.0,
                      tolerance=1e-6, max_iterations=100, verbose=true,
                      use_arnoldi=true, krylov_dim=50, arnoldi_tol=1e-10,
                      include_uix=false)

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
- `include_uix`: Include UIX three-body force (default: false)

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
                           krylov_dim::Int=50, arnoldi_tol::Float64=1e-6,
                           include_uix::Bool=false)

    if verbose
        println("\n" * "="^70)
        println("         MALFIET-TJON EIGENVALUE SOLVER")
        println("="^70)
        potential_str = include_uix ? "$potname + UIX" : potname
        println("Potential: $potential_str")
        println("Two-body threshold: $(round(e2b[1], digits=6)) MeV")
        println("Initial energy guesses: E0 = $E0 MeV, E1 = $E1 MeV")
        println("Convergence tolerance: $tolerance")
        println("Maximum iterations: $max_iterations")
        method_str = use_arnoldi ? "Arnoldi (Krylov dim: $krylov_dim)" : "Direct diagonalization"
        println("Eigenvalue method: $method_str")
        if use_arnoldi
            println("Arnoldi tolerance: $arnoldi_tol")
        end
        if include_uix
            println("Three-body force: UIX (Urbana IX)")
        end
        println("-"^70)
    end

    # Initialize wave function variables
    ψtot = ComplexF64[]
    ψ3 = ComplexF64[]

    # Pre-compute matrices once (they don't change between iterations)
    V_UIX = nothing  # Initialize UIX potential

    if verbose
        print("  Building matrices... ")
        @time begin
            T, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)  # Kinetic energy with components
            V, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)   # Potential energy with components
            B = Bmatrix(α, grid)            # Overlap matrix
            Rxy,Rxy_31,Rxy_32 = Rxy_matrix(α, grid)       # Rearrangement matrix R

            # Compute UIX three-body force if requested (separate from V)
            if include_uix
                V_UIX = compute_uix_potential(α, grid, Rxy_31, Rxy)
                if verbose
                    println("    ✓ UIX three-body force computed")
                end
            end
        end
    else
        T, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)  # Kinetic energy with components
        V, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)   # Potential energy with components
        B = Bmatrix(α, grid)            # Overlap matrix
        Rxy,Rxy_31,Rxy_32 = Rxy_matrix(α, grid)       # Rearrangement matrix R

        # Compute UIX three-body force if requested (separate from V)
        if include_uix
            V_UIX = compute_uix_potential(α, grid, Rxy_31, Rxy)
        end
    end

    # Check rearrangement matrix symmetry
    # check_rxy_symmetry(Rxy_31, Rxy_32, α, grid; verbose=verbose)


    # Initialize secant method
    E_prev = E0
    E_curr = E1

    # Compute initial eigenvalues (no previous eigenvector for first iteration)
    λ_prev, eigenvec_prev = compute_lambda_eigenvalue(E_prev, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                                     verbose=verbose, use_arnoldi=use_arnoldi,
                                                     krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                     V_UIX=V_UIX)
    # Use previous eigenvector for second initial guess
    λ_curr, eigenvec_curr = compute_lambda_eigenvalue(E_curr, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                                     verbose=verbose, use_arnoldi=use_arnoldi,
                                                     krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                     previous_eigenvector=eigenvec_prev,
                                                     V_UIX=V_UIX)

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
        ψ = eigenvec_curr
        ψtot, ψ3 = compute_total_wavefunction(ψ, Rxy, B)
        return MalflietTjonResult(E_curr, λ_curr, eigenvec_curr, 0, convergence_history, true), ψtot, ψ3
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
        λ_next, eigenvec_next = compute_lambda_eigenvalue(E_next, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                                         verbose=verbose, use_arnoldi=use_arnoldi,
                                                         krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                         previous_eigenvector=eigenvec_curr,
                                                         V_UIX=V_UIX)

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

            # Normalize the eigenvector properly with respect to the overlap matrix B
            ψ = eigenvec_next
            ψtot, ψ3 = compute_total_wavefunction(ψ, Rxy, B)

            if verbose

                # compute the probabilities in each channel
                channel_probs, channel_info = compute_channel_probabilities(ψtot, B, α, grid)

                # Print channel probability results
                println()
                println("Channel probability contributions:")
                println("-"^60)
                for (i, (prob, info)) in enumerate(zip(channel_probs, channel_info))
                    @printf("  %s: %8.4f%%\n", info, prob * 100.0)
                end
                println("-"^60)


                # Compute individual expectation values
                T_expectation = 3.0*real(ψtot' * T * ψ3)
                V_expectation = 3.0*real(ψtot' * V * ψtot)

                # Include UIX expectation values if present (both formulations)
                UIX_expectation_tot = 0.0
                UIX_expectation_3 = 0.0
                if V_UIX !== nothing
                    UIX_expectation_tot = 3.0*real(ψtot' * V_UIX * ψ3)
                end

                # Total Hamiltonian expectation value: H = T + V + UIX
                H_expectation = T_expectation + V_expectation + UIX_expectation_tot


                println("-"^70)
                println("✓ CONVERGED!")
                @printf("Ground state energy: %10.6f MeV\n", E_next)
                @printf("Final eigenvalue λ:   %10.6f\n", λ_next)
                @printf("Binding energy:      %10.6f MeV\n", -E_next)
                println()
                println("Expectation values:")
                @printf("  <ψ|T|ψ>      = %10.6f MeV\n", T_expectation)
                @printf("  <ψ|V|ψ>      = %10.6f MeV\n", V_expectation)
                if V_UIX !== nothing
                    @printf("  <ψ|UIX|ψ>    = %10.6f MeV  [3⟨ψtot|UIX|ψtot⟩]\n", UIX_expectation_tot)
                end
                @printf("  <ψ|H|ψ>      = %10.6f MeV\n", H_expectation)
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
                                     convergence_history, true), ψtot, ψ3
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

    # For non-converged case, compute wave functions from final eigenvector
    if !converged && final_eigenvec !== nothing
        ψ = final_eigenvec
        ψtot, ψ3 = compute_total_wavefunction(ψ, Rxy, B)
    elseif !converged
        # If no final eigenvector, return empty arrays
        ψtot = ComplexF64[]
        ψ3 = ComplexF64[]
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
                             convergence_history, converged), ψtot, ψ3
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
    compute_total_wavefunction(ψ, Rxy, B)

Normalize the three-body wave function using the proper Faddeev normalization scheme.

This function takes an unnormalized wave function component, constructs the complete 
Faddeev wave function, applies the correct normalization, and verifies the result.

# Arguments
- `ψ`: Unnormalized wave function component ψ₃ (from malfiet_tjon_solve)
- `Rxy`: Rearrangement matrix (from Rxy_matrix function)
- `B`: Overlap matrix (for proper normalization)

# Returns
- `ψ_normalized`: Properly normalized total wave function |Ψ̄⟩

# Physics and Normalization
The total three-body wave function in the Faddeev formalism is given by:
    |Ψ⟩ = (1 + P⁺ + P⁻)|ψ₃⟩ = (1 + Rxy)|ψ₃⟩

The wave function is normalized using the condition:
    ⟨Ψ|Ψ⟩ = ⟨Ψ|(1 + P⁺ + P⁻)|ψ₃⟩ = 3⟨Ψ|ψ₃⟩

where the factor of 3 comes from the sum of identity and permutation operators.
The normalized wave functions are:
    |Ψ̄⟩ = |Ψ⟩/√(3⟨Ψ|ψ₃⟩)
    |ψ̄₃⟩ = |ψ₃⟩/√(3⟨Ψ|ψ₃⟩)

This normalization avoids issues with higher partial wave components outside 
the truncated model space by using ⟨Ψ|ψ₃⟩ instead of ⟨Ψ|Ψ⟩.
"""
function compute_total_wavefunction(ψ::Vector, Rxy, B)
    # Compute the total wave function: |Ψ⟩ = (1 + Rxy)|ψ₃⟩
    # The identity operator (1) is represented implicitly
    ψ_total = ψ + Rxy * ψ

    # Compute the inner product ⟨Ψ|ψ₃⟩ with respect to the overlap matrix B
    psi_psi3_inner = real(ψ_total' * B * ψ)

    # Check if the inner product is positive (required for normalization)
    if psi_psi3_inner <= 0
        error("Inner product ⟨Ψ|ψ₃⟩ = $psi_psi3_inner ≤ 0, cannot normalize wave function")
    end

    # Apply the proper Faddeev normalization: |Ψ̄⟩ = |Ψ⟩/√(3⟨Ψ|ψ₃⟩)
    # This is the CORRECT normalization for truncated model space
    # because it only involves ψ₃ which is converged within the truncation
    normalization_factor = sqrt(3.0 * psi_psi3_inner)
    ψ_normalized = ψ_total / normalization_factor
    ψ3_normalized = ψ / normalization_factor

    # Verify normalization: Check that ⟨Ψ̄|ψ̄₃⟩ = 1/3
    verification_inner = real(ψ_normalized' * B * ψ3_normalized)
    expected_value = 1.0 / 3.0  # Since ⟨Ψ̄|Ψ̄⟩ = 3⟨Ψ̄|ψ̄₃⟩ = 1

    # Check normalization accuracy (tolerance for numerical precision)
    normalization_error = abs(verification_inner - expected_value)
    if normalization_error > 1e-10
        @warn "Normalization verification failed: ⟨Ψ̄|ψ̄₃⟩ = $verification_inner, expected = $expected_value, error = $normalization_error"
    end

    # Also verify that ⟨Ψ̄|B|Ψ̄⟩ = 1 (total norm should be 1)
    total_norm_squared = real(ψ_normalized' * B * ψ_normalized)
    total_norm_error = abs(total_norm_squared - 1.0)
    if total_norm_error > 1e-10
        println("Total normalization check: ⟨Ψ̄|B|Ψ̄⟩ = $total_norm_squared, expected = 1.0")
    end

    return ψ_normalized, ψ3_normalized
end

"""
        check_rxy_symmetry(Rxy_31, Rxy_32, α, grid; tolerance=1e-12, verbose=true)
    
    Check if the rearrangement matrices Rxy_31 and Rxy_32 are equal.
    
    In a three-body system with identical particles (like three nucleons),
    the rearrangement operators R₃₁ and R₃₂ should be equal due to particle
    exchange symmetry: R₃₁ (1↔3) should equal R₃₂ (2↔3) when particles 
    1 and 2 are identical.
    
    # Arguments
    - `Rxy_31`: Rearrangement matrix for 1↔3 particle exchange
    - `Rxy_32`: Rearrangement matrix for 2↔3 particle exchange  
    - `α`: Channel structure with quantum numbers
    - `grid`: Spatial grid information
    - `tolerance`: Numerical tolerance for matrix comparison (default: 1e-12)
    - `verbose`: Print detailed comparison results (default: true)
    
    # Returns
    - `are_equal`: Boolean indicating if matrices are equal within tolerance
    - `max_diff`: Maximum absolute difference between matrix elements
    """
    function check_rxy_symmetry(Rxy_31, Rxy_32, α, grid; tolerance=1e-12, verbose=true)
        # Check that matrices have the same dimensions
        if size(Rxy_31) != size(Rxy_32)
            if verbose
                println("ERROR: Matrix dimensions differ")
                println("  Rxy_31 size: $(size(Rxy_31))")
                println("  Rxy_32 size: $(size(Rxy_32))")
            end
            return false, Inf
        end
        
        # Check each element individually
        n_rows, n_cols = size(Rxy_31)
        
        # Arrays to track which relation each element satisfies
        symmetric_mask = falses(n_rows, n_cols)
        antisymmetric_mask = falses(n_rows, n_cols)
        neither_mask = falses(n_rows, n_cols)
        
        max_diff_overall = 0.0
        
        for i in 1:n_rows, j in 1:n_cols
            # Check symmetric relation: Rxy_31[i,j] = Rxy_32[i,j]
            diff_symmetric = abs(Rxy_31[i,j] - Rxy_32[i,j])
            
            # Check antisymmetric relation: Rxy_31[i,j] = -Rxy_32[i,j]
            diff_antisymmetric = abs(Rxy_31[i,j] + Rxy_32[i,j])
            
            if diff_symmetric < tolerance
                symmetric_mask[i,j] = true
                max_diff_overall = max(max_diff_overall, diff_symmetric)
            elseif diff_antisymmetric < tolerance
                antisymmetric_mask[i,j] = true
                max_diff_overall = max(max_diff_overall, diff_antisymmetric)
            else
                neither_mask[i,j] = true
                max_diff_overall = max(max_diff_overall, min(diff_symmetric, diff_antisymmetric))
            end
        end
        
        # Count elements satisfying each relation
        n_symmetric = sum(symmetric_mask)
        n_antisymmetric = sum(antisymmetric_mask)
        n_neither = sum(neither_mask)
        n_total = n_rows * n_cols
        
        # Overall test passes if no elements fail both relations
        are_equal = n_neither == 0
        max_diff = max_diff_overall
        mean_diff = max_diff_overall  # Simplified for now
        
        if verbose
            println("\n" * "="^60)
            println("         REARRANGEMENT MATRIX SYMMETRY CHECK")
            println("="^60)
            println("Matrix dimensions: $(size(Rxy_31))")
            @printf("Total elements: %d\n", n_total)
            @printf("Elements satisfying Rxy_31 = Rxy_32:  %d (%.1f%%)\n", n_symmetric, 100.0*n_symmetric/n_total)
            @printf("Elements satisfying Rxy_31 = -Rxy_32: %d (%.1f%%)\n", n_antisymmetric, 100.0*n_antisymmetric/n_total)
            @printf("Elements satisfying neither:         %d (%.1f%%)\n", n_neither, 100.0*n_neither/n_total)
            @printf("Maximum difference: %12.6e\n", max_diff)
            @printf("Tolerance: %12.6e\n", tolerance)
            
            # Analyze channel-by-channel symmetry
            println("\nChannel-by-channel analysis:")
            println("────────────────────────────────────────────────────────")
            
            # Debug matrix structure
            println("Matrix size: $(n_rows) × $(n_cols)")
            println("Number of channels: $(α.nchmax)")
            println("Grid points: nx=$(grid.nx), ny=$(grid.ny), nθ=$(grid.nθ)")
            # Rxy matrix uses only nx × ny, not nθ (based on matrices.jl)
            grid_points_per_channel = grid.nx * grid.ny
            println("Grid points per channel (nx×ny): $grid_points_per_channel")
            println("Expected total matrix size: $(α.nchmax * grid_points_per_channel)")
            
            # Check if matrix organization matches expectation
            if n_rows == α.nchmax * grid_points_per_channel
                println("✓ Matrix structure matches expected channel×grid organization")
                
                # Group analysis by channel pairs
                symmetric_channels = Tuple{Int,Int}[]
                antisymmetric_channels = Tuple{Int,Int}[]
                mixed_channels = Tuple{Int,Int}[]
                
                for αout in 1:α.nchmax
                    for αin in 1:α.nchmax
                        # Calculate the matrix block for this channel pair
                        row_start = (αout-1) * grid_points_per_channel + 1
                        row_end = αout * grid_points_per_channel
                        col_start = (αin-1) * grid_points_per_channel + 1
                        col_end = αin * grid_points_per_channel
                        
                        # Extract the block for this channel pair
                        block_symmetric = symmetric_mask[row_start:row_end, col_start:col_end]
                        block_antisymmetric = antisymmetric_mask[row_start:row_end, col_start:col_end]
                        
                        n_sym_block = sum(block_symmetric)
                        n_antisym_block = sum(block_antisymmetric)
                        n_total_block = length(block_symmetric)
                        
                        # Classify this channel pair
                        if n_sym_block == n_total_block && n_sym_block > 0
                            push!(symmetric_channels, (αout, αin))
                        elseif n_antisym_block == n_total_block && n_antisym_block > 0
                            push!(antisymmetric_channels, (αout, αin))
                        elseif n_sym_block > 0 && n_antisym_block > 0
                            push!(mixed_channels, (αout, αin))
                        end
                    end
                end
                
                # Display results by channel type
                if !isempty(symmetric_channels)
                    println("\nChannels satisfying Rxy_31 = Rxy_32 (symmetric): $(length(symmetric_channels)) pairs")
                    for (i, (αout, αin)) in enumerate(symmetric_channels)
                        if i <= 10  # Show first 10
                            @printf("  (αout=%2d, αin=%2d): J₁₂=%.1f, J₃=%.1f, J=%.1f, T₁₂=%.1f, T=%.1f\n", 
                                   αout, αin, α.J12[αout], α.J3[αout], α.J, α.T12[αout], α.T[αout])
                        elseif i == 11
                            println("  ... and $(length(symmetric_channels)-10) more")
                            break
                        end
                    end
                end
                
                if !isempty(antisymmetric_channels)
                    println("\nChannels satisfying Rxy_31 = -Rxy_32 (antisymmetric): $(length(antisymmetric_channels)) pairs")
                    for (i, (αout, αin)) in enumerate(antisymmetric_channels)
                        if i <= 10  # Show first 10
                            @printf("  (αout=%2d, αin=%2d): J₁₂=%.1f, J₃=%.1f, J=%.1f, T₁₂=%.1f, T=%.1f\n", 
                                   αout, αin, α.J12[αout], α.J3[αout], α.J, α.T12[αout], α.T[αout])
                        elseif i == 11
                            println("  ... and $(length(antisymmetric_channels)-10) more")
                            break
                        end
                    end
                end
                
                if !isempty(mixed_channels)
                    println("\nChannels with mixed symmetry: $(length(mixed_channels)) pairs")
                    for (i, (αout, αin)) in enumerate(mixed_channels)
                        if i <= 5  # Show first 5 for mixed channels
                            @printf("  (αout=%2d, αin=%2d): J₁₂=%.1f, J₃=%.1f, J=%.1f, T₁₂=%.1f, T=%.1f\n", 
                                   αout, αin, α.J12[αout], α.J3[αout], α.J, α.T12[αout], α.T[αout])
                        elseif i == 6
                            println("  ... and $(length(mixed_channels)-5) more")
                            break
                        end
                    end
                end
                
                if isempty(symmetric_channels) && isempty(antisymmetric_channels) && isempty(mixed_channels)
                    println("\n⚠ No clear channel patterns found - all blocks may have zero or uniform entries")
                end
            else
                println("⚠ Matrix structure doesn't match expected channel organization")
                println("  Cannot perform channel-by-channel analysis")
            end
            println()
            
            if are_equal
                println("✓ PASSED: All elements satisfy either Rxy_31 = Rxy_32 or Rxy_31 = -Rxy_32")
                if n_symmetric > 0 && n_antisymmetric > 0
                    println("  Mixed symmetric/antisymmetric relations found")
                elseif n_symmetric > n_antisymmetric
                    println("  Predominantly symmetric particle exchange")
                else
                    println("  Predominantly antisymmetric particle exchange (identical fermions)")
                end
            else
                println("✗ FAILED: Some elements satisfy neither relation")
                println("  This may indicate:")
                println("    - Incorrect implementation of rearrangement matrices")
                println("    - Numerical precision issues")
                
                # Show some elements that fail both relations
                if n_neither > 0
                    println("\n  Sample elements failing both relations:")
                    n_show = min(5, n_neither)
                    count = 0
                    for i in 1:n_rows, j in 1:n_cols
                        if neither_mask[i,j] && count < n_show
                            diff_sym = abs(Rxy_31[i,j] - Rxy_32[i,j])
                            diff_antisym = abs(Rxy_31[i,j] + Rxy_32[i,j])
                            @printf("    [%d,%d]: Rxy_31=%12.6e, Rxy_32=%12.6e, diff_sym=%12.6e, diff_antisym=%12.6e\n", 
                                   i, j, Rxy_31[i,j], Rxy_32[i,j], diff_sym, diff_antisym)
                            count += 1
                        end
                    end
                end
            end
            println("="^60)
        end
        
        return are_equal, max_diff
    end

"""
    compute_channel_probabilities(ψ_normalized, B, α, grid)

Compute the probability contribution of each channel in the three-body wave function.

The probability contribution of each channel i is computed as:
    P_i = real(ψ_normalized[i]' * B[i,i] * ψ_normalized[i])

where ψ_normalized is the FULL Faddeev wavefunction Ψ̄ = (1 + Rxy)ψ̄₃ normalized
using the Faddeev scheme: |Ψ̄⟩ = |Ψ⟩/√(3⟨Ψ|ψ₃⟩) so that ⟨Ψ̄|Ψ̄⟩ = 1.

# Arguments
- `ψ_normalized`: Normalized FULL wave function |Ψ̄⟩ where ⟨Ψ̄|B|Ψ̄⟩ = 1
- `B`: Overlap matrix
- `α`: Channel structure (used to get channel information)
- `grid`: Mesh structure (used to get grid dimensions)

# Returns
- `channel_probs`: Vector of probability contributions for each channel (sum to 1)
- `channel_info`: Vector of channel descriptions for labeling

# Physics
In the Faddeev formalism, each channel represents a specific coupling of
angular momentum and isospin quantum numbers. The probability contribution
shows how much each channel contributes to the total three-body bound state.
"""
function compute_channel_probabilities(ψ_normalized, B, α, grid)
    # Channel probabilities from the FULL wavefunction Ψ̄
    # Use ⟨Ψ̄_channel|B|Ψ̄_channel⟩ which automatically sums to 1

    # Get the number of channels and grid points per channel
    nchannels = α.nchmax
    nx, ny = grid.nx, grid.ny
    points_per_channel = nx * ny

    # Initialize arrays for results
    channel_probs = Float64[]
    channel_info = String[]

    # Loop over each channel
    for i in 1:nchannels
        # Calculate index range for this channel
        start_idx = (i-1) * points_per_channel + 1
        end_idx = i * points_per_channel

        # Extract the FULL wave function component for this channel
        ψ_channel = ψ_normalized[start_idx:end_idx]

        # Extract the overlap matrix block for this channel
        B_channel = B[start_idx:end_idx, start_idx:end_idx]

        # Channel probability: ⟨Ψ̄_channel|B|Ψ̄_channel⟩
        # Since ψ_normalized satisfies ⟨Ψ̄|B|Ψ̄⟩ = 1, these sum to 1
        prob = real(ψ_channel' * B_channel * ψ_channel)
        push!(channel_probs, prob)

        # Create channel description using the nch3b structure fields
        # Format: (l₁₂(s₁s₂)s₁₂)J₁₂, (λ₃s₃)J₃, J; (t₁t₂)T₁₂, t₃, T
        channel_desc = @sprintf("Ch %2d: (l₁₂=%d,s₁₂=%.1f)J₁₂=%.1f, (λ₃=%d,s₃=%.1f)J₃=%.1f, J=%.1f; T₁₂=%.1f, t₃=%.1f, T=%.1f",
                               i, α.l[i], α.s12[i], α.J12[i], α.λ[i], α.s3, α.J3[i], α.J, α.T12[i], α.t3, α.T[i])
        push!(channel_info, channel_desc)
    end

    # Verify that probabilities sum to approximately 1
    total_prob = sum(channel_probs)
    if abs(total_prob - 1.0) > 1e-6
        @warn "Channel probabilities do not sum to 1: sum = $total_prob"
    end

    return channel_probs, channel_info
end

"""
    malfiet_tjon_solve_optimized(α, grid, potname, e2b; E0=-8.0, E1=-7.0,
                                 tolerance=1e-6, max_iterations=100, verbose=true,
                                 use_arnoldi=true, krylov_dim=50, arnoldi_tol=1e-6,
                                 include_uix=false)

**OPTIMIZED VERSION** of Malfiet-Tjon solver using optimized T, V, and Rxy matrix functions.

This version uses:
- `T_matrix_optimized()` for ~8× faster kinetic energy matrix computation
- `V_matrix_optimized()` for ~1.5-2× faster potential matrix computation
- `Rxy_matrix_optimized()` for ~2× faster rearrangement matrix computation
- Overall speedup: ~2-3× for matrix construction phase

All other arguments and behavior are identical to `malfiet_tjon_solve()`.

# Performance Benefits
- Faster matrix construction (typically 40-60% of total time)
- Same numerical accuracy as non-optimized version
- 50% memory reduction in Rxy computation (exploits symmetry)

# Arguments
See `malfiet_tjon_solve()` documentation for detailed argument descriptions.

# Returns
- `MalflietTjonResult`: Structure containing converged energy, eigenvalue, and diagnostics
- `ψtot`: Normalized total wave function
- `ψ3`: Normalized component wave function

# Example
```julia
# using .channels, .mesh, .MalflietTjon

α = α3b(true, 0.5, 0.5, 1, 2, 0, 4, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 2.0)
grid = initialmesh(12, 15, 15, 15.0, 15.0, 0.5)
e2b = [-2.2246]

result, ψtot, ψ3 = malfiet_tjon_solve_optimized(α, grid, "AV18", e2b; include_uix=true)
```
"""
function malfiet_tjon_solve_optimized(α, grid, potname, e2b;
                                     E0::Float64=-8.0, E1::Float64=-7.0,
                                     tolerance::Float64=1e-6, max_iterations::Int=100,
                                     verbose::Bool=true, use_arnoldi::Bool=true,
                                     krylov_dim::Int=50, arnoldi_tol::Float64=1e-6,
                                     include_uix::Bool=true,
                                     θ_deg::Float64=0.0, n_gauss=nothing)

    # Timing dictionary to track performance
    timings = Dict{String, Float64}()
    total_time_start = time()

    # ERROR: UIX three-body force cannot be used with complex scaling
    if θ_deg != 0.0 && include_uix
        error("""
        Complex scaling (θ=$(θ_deg)°) with UIX three-body force is not supported!

        There is no general representation for three-body forces under complex scaling.
        This requires special treatment which is currently not implemented.

        Please set include_uix=false to proceed with two-body potential only.
        """)
    end

    if verbose
        println("\n" * "="^70)
        println("    MALFIET-TJON EIGENVALUE SOLVER (OPTIMIZED)")
        println("="^70)
        potential_str = include_uix ? "$potname + UIX" : potname
        println("Potential: $potential_str")
        println("Two-body threshold: $(round(e2b[1], digits=6)) MeV")
        println("Initial energy guesses: E0 = $E0 MeV, E1 = $E1 MeV")
        println("Convergence tolerance: $tolerance")
        println("Maximum iterations: $max_iterations")
        method_str = use_arnoldi ? "Arnoldi (Krylov dim: $krylov_dim)" : "Direct diagonalization"
        println("Eigenvalue method: $method_str")
        if use_arnoldi
            println("Arnoldi tolerance: $arnoldi_tol")
        end
        if include_uix
            println("Three-body force: UIX (Urbana IX)")
        end
        println("Matrix functions: OPTIMIZED (T, V, Rxy)")
        println("-"^70)
    end

    # Initialize wave function variables
    ψtot = ComplexF64[]
    ψ3 = ComplexF64[]

    # Pre-compute matrices once using OPTIMIZED functions
    V_UIX = nothing  # Initialize UIX potential
    Gαα = nothing    # Initialize G-coefficients (needed for optimized UIX)

    # Time matrix construction
    matrix_time_start = time()
    if verbose
        println("  Building matrices (optimized)...")

        print("    - T matrix: ")
        t_start = time()
        T, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true, θ_deg=θ_deg)
        t_time = time() - t_start
        timings["T_matrix"] = t_time
        @printf("%.3f s\n", t_time)

        print("    - V matrix: ")
        v_start = time()
        # V_matrix_optimized_scaled automatically uses V_matrix_optimized when θ=0
        V, V_x_diag_ch = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg, n_gauss=n_gauss, return_components=true)
        v_time = time() - v_start
        timings["V_matrix"] = v_time
        @printf("%.3f s\n", v_time)

        print("    - B matrix: ")
        b_start = time()
        B = Bmatrix(α, grid)
        b_time = time() - b_start
        timings["B_matrix"] = b_time
        @printf("%.3f s\n", b_time)

        print("    - Rxy matrix: ")
        r_start = time()
        Rxy, Rxy_31, Rxy_32 = Rxy_matrix_with_caching(α, grid)
        r_time = time() - r_start
        timings["Rxy_matrix"] = r_time
        @printf("%.3f s\n", r_time)

        # Compute G-coefficients and UIX three-body force if requested (separate from V)
        if include_uix
            print("    - G-coefficients: ")
            g_start = time()
            # Need G-coefficients for optimized UIX
            include("Gcoefficient.jl")
            Gαα = eval(:(Gcoefficient.computeGcoefficient($α, $grid)))
            g_time = time() - g_start
            timings["G_coefficients"] = g_time
            @printf("%.3f s\n", g_time)

            print("    - UIX potential (optimized): ")
            uix_start = time()
            V_UIX = compute_uix_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)
            uix_time = time() - uix_start
            timings["UIX_potential"] = uix_time
            @printf("%.3f s (optimized with caching + hybrid sparse/dense)\n", uix_time)
        end

        matrix_time = time() - matrix_time_start
        timings["total_matrix_construction"] = matrix_time
        @printf("  Total matrix construction: %.3f s\n", matrix_time)
        println()
    else
        # Use optimized matrix functions for T, V, and Rxy
        T, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true, θ_deg=θ_deg)

        # V_matrix_optimized_scaled automatically uses V_matrix_optimized when θ=0
        V, V_x_diag_ch = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg, n_gauss=n_gauss, return_components=true)

        B = Bmatrix(α, grid)
        Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized(α, grid)

        # Compute UIX three-body force if requested (separate from V)
        if include_uix
            # Need G-coefficients for optimized UIX
            include("Gcoefficient.jl")
            Gαα = eval(:(Gcoefficient.computeGcoefficient($α, $grid)))
            V_UIX = compute_uix_potential_optimized(α, grid, Rxy_31, Rxy, Gαα)
        end

        timings["total_matrix_construction"] = time() - matrix_time_start
    end

    # Precompute M⁻¹ cache (ONE-TIME COST, huge speedup!)
    if verbose
        print("  Precomputing M⁻¹ cache (one-time)... ")
        cache_start = time()
        @time M_cache = matrices.precompute_M_inverse_cache(α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
        cache_time = time() - cache_start
        timings["M_cache_precompute"] = cache_time
    else
        M_cache = matrices.precompute_M_inverse_cache(α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
        timings["M_cache_precompute"] = 0.0
    end

    # Precompute RHS matrix cache (ONE-TIME COST, another huge speedup!)
    # RHS = (V - V_αα) + V*Rxy + UIX is completely energy-independent!
    if verbose
        print("  Precomputing RHS matrix cache (one-time)... ")
        rhs_cache_start = time()
        @time RHS_cache = precompute_RHS_cache(V, V_x_diag_ch, Rxy, α, grid; V_UIX=V_UIX)
        rhs_cache_time = time() - rhs_cache_start
        timings["RHS_cache_precompute"] = rhs_cache_time
    else
        RHS_cache = precompute_RHS_cache(V, V_x_diag_ch, Rxy, α, grid; V_UIX=V_UIX)
        timings["RHS_cache_precompute"] = 0.0
    end

    # Initialize secant method
    E_prev = E0
    E_curr = E1

    # Time initial eigenvalue computations
    if verbose
        println("  Computing initial eigenvalues...")
    end

    eigenval_times = Float64[]

    # Compute initial eigenvalues (no previous eigenvector for first iteration) - USE CACHES!
    eigen_start = time()
    λ_prev, eigenvec_prev = compute_lambda_eigenvalue_optimized(E_prev, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                                     verbose=false, use_arnoldi=use_arnoldi,
                                                     krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                     V_UIX=V_UIX, M_cache=M_cache, RHS_cache=RHS_cache)
    eigen1_time = time() - eigen_start
    push!(eigenval_times, eigen1_time)
    if verbose
        @printf("    E = %.4f MeV: %.3f s\n", E_prev, eigen1_time)
    end

    # Use previous eigenvector for second initial guess - USE CACHES!
    eigen_start = time()
    λ_curr, eigenvec_curr = compute_lambda_eigenvalue_optimized(E_curr, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                                     verbose=false, use_arnoldi=use_arnoldi,
                                                     krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                     previous_eigenvector=eigenvec_prev,
                                                     V_UIX=V_UIX, M_cache=M_cache, RHS_cache=RHS_cache)
    eigen2_time = time() - eigen_start
    push!(eigenval_times, eigen2_time)
    if verbose
        @printf("    E = %.4f MeV: %.3f s\n", E_curr, eigen2_time)
        println()
    end

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
        ψ = eigenvec_curr
        ψtot, ψ3 = compute_total_wavefunction(ψ, Rxy, B)
        return MalflietTjonResult(E_curr, λ_curr, eigenvec_curr, 0, convergence_history, true), ψtot, ψ3
    end

    # Secant method iteration
    converged = false
    final_eigenvec = eigenvec_curr
    iteration_time_start = time()

    for iteration in 1:max_iterations
        # Secant method update
        if abs(λ_curr - λ_prev) < 1e-15
            @warn "λ values too close, secant method may be unstable"
            break
        end

        # Check for divergence
        if abs(E_curr) > 100.0
            @warn "Method appears to be diverging, try different initial energy guesses"
            break
        end

        # Secant formula: E_{n+1} = E_n - (λ_n - 1) * (E_n - E_{n-1}) / (λ_n - λ_{n-1})
        E_next = E_curr - (λ_curr - 1) * (E_curr - E_prev) / (λ_curr - λ_prev)

        # Time eigenvalue computation for this iteration - USE CACHES!
        iter_eigen_start = time()
        λ_next, eigenvec_next = compute_lambda_eigenvalue_optimized(E_next, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                                         verbose=false, use_arnoldi=use_arnoldi,
                                                         krylov_dim=krylov_dim, arnoldi_tol=arnoldi_tol,
                                                         previous_eigenvector=eigenvec_curr,
                                                         V_UIX=V_UIX, M_cache=M_cache, RHS_cache=RHS_cache)
        iter_eigen_time = time() - iter_eigen_start
        push!(eigenval_times, iter_eigen_time)

        if isnan(λ_next)
            @warn "Eigenvalue calculation failed at E = $E_next, stopping iteration"
            break
        end

        # Check convergence
        residual = abs(λ_next - 1)
        push!(convergence_history, (E_next, λ_next))

        if verbose
            @printf("Iter %2d: E = %8.4f MeV, λ = %10.6f, |λ-1| = %8.2e  (%.3f s)\n",
                   iteration, E_next, λ_next, residual, iter_eigen_time)
        end

        if residual < tolerance
            converged = true
            final_eigenvec = eigenvec_next

            # Normalize the eigenvector properly with respect to the overlap matrix B
            ψ = eigenvec_next
            ψtot, ψ3 = compute_total_wavefunction(ψ, Rxy, B)

            if verbose
                # Compute the probabilities in each channel
                channel_probs, channel_info = compute_channel_probabilities(ψtot, B, α, grid)

                # Print channel probability results
                println()
                println("Channel probability contributions:")
                println("-"^60)
                for (i, (prob, info)) in enumerate(zip(channel_probs, channel_info))
                    @printf("  %s: %8.4f%%\n", info, prob * 100.0)
                end
                println("-"^60)

                # Compute individual expectation values
                T_expectation = 3.0*real(ψtot' * T * ψ3)
                V_expectation = 3.0*real(ψtot' * V * ψtot)

                # Include UIX expectation values if present
                UIX_expectation_tot = 0.0
                if V_UIX !== nothing
                    UIX_expectation_tot = 3.0*real(ψtot' * V_UIX * ψ3)
                end

                # Total Hamiltonian expectation value: H = T + V + UIX
                H_expectation = T_expectation + V_expectation + UIX_expectation_tot

                println("-"^70)
                println("✓ CONVERGED!")
                @printf("Ground state energy: %10.6f MeV\n", E_next)
                @printf("Final eigenvalue λ:   %10.6f\n", λ_next)
                @printf("Binding energy:      %10.6f MeV\n", -E_next)
                println()
                println("Expectation values:")
                @printf("  <ψ|T|ψ>      = %10.6f MeV\n", T_expectation)
                @printf("  <ψ|V|ψ>      = %10.6f MeV\n", V_expectation)
                if V_UIX !== nothing
                    @printf("  <ψ|UIX|ψ>    = %10.6f MeV\n", UIX_expectation_tot)
                end
                @printf("  <ψ|H|ψ>      = %10.6f MeV\n", H_expectation)
                println()
                @printf("Energy difference:   %10.6f MeV\n", abs(H_expectation - E_next))
                if abs(H_expectation - E_next) < 1e-3
                    println("✓ Energy consistency check: PASSED")
                else
                    println("⚠ Energy consistency check: FAILED")
                end

                # Add timing summary
                iteration_time = time() - iteration_time_start
                total_time = time() - total_time_start
                timings["secant_iterations"] = iteration_time
                timings["total_eigenvalue_time"] = sum(eigenval_times)
                timings["total_runtime"] = total_time

                println()
                println("="^70)
                println("                    TIMING SUMMARY")
                println("="^70)
                @printf("Matrix construction:     %8.3f s  (%5.1f%%)\n",
                       timings["total_matrix_construction"],
                       100*timings["total_matrix_construction"]/total_time)
                @printf("  - T matrix:            %8.3f s\n", timings["T_matrix"])
                @printf("  - V matrix:            %8.3f s\n", timings["V_matrix"])
                @printf("  - B matrix:            %8.3f s\n", timings["B_matrix"])
                @printf("  - Rxy matrix:          %8.3f s\n", timings["Rxy_matrix"])
                if haskey(timings, "UIX_potential")
                    @printf("  - UIX potential:       %8.3f s\n", timings["UIX_potential"])
                end
                println()
                if haskey(timings, "M_cache_precompute") && timings["M_cache_precompute"] > 0
                    @printf("M⁻¹ cache precompute:    %8.3f s  (%5.1f%%) [ONE-TIME]\n",
                           timings["M_cache_precompute"],
                           100*timings["M_cache_precompute"]/total_time)
                    println()
                end
                @printf("Eigenvalue computations: %8.3f s  (%5.1f%%)\n",
                       timings["total_eigenvalue_time"],
                       100*timings["total_eigenvalue_time"]/total_time)
                @printf("  - Initial (2):         %8.3f s\n", eigenval_times[1] + eigenval_times[2])
                if length(eigenval_times) > 2
                    @printf("  - Iterations (%d):      %8.3f s  (avg: %.3f s)\n",
                           length(eigenval_times) - 2,
                           sum(eigenval_times[3:end]),
                           mean(eigenval_times[3:end]))
                    @printf("    Min/Max iteration:   %.3f s / %.3f s\n",
                           minimum(eigenval_times[3:end]),
                           maximum(eigenval_times[3:end]))
                end
                println()
                @printf("Total runtime:           %8.3f s\n", total_time)
                println("="^70)
            end
            return MalflietTjonResult(E_next, λ_next, eigenvec_next, iteration,
                                     convergence_history, true), ψtot, ψ3
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

    # For non-converged case, compute wave functions from final eigenvector
    if !converged && final_eigenvec !== nothing
        ψ = final_eigenvec
        ψtot, ψ3 = compute_total_wavefunction(ψ, Rxy, B)
    elseif !converged
        # If no final eigenvector, return empty arrays
        ψtot = ComplexF64[]
        ψ3 = ComplexF64[]
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
                             convergence_history, converged), ψtot, ψ3
end


end # module MalflietTjon
