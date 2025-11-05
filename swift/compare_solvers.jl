using LinearAlgebra
using IterativeSolvers
using Printf

# PERFORMANCE: Use all available CPU cores for BLAS operations
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS threads set to: $(BLAS.get_num_threads()) (CPU cores: $(Sys.CPU_THREADS))")
println()

# ============================================================================
# Linear operator wrapper for GMRES preconditioner
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

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
include("twobody.jl")
include("matrices_optimized.jl")
include("matrices.jl")
include("scattering.jl")

using .channels
using .mesh
using .twobodybound
using .matrices_optimized
using .matrices
using .Scattering

println("="^70)
println("    SOLVER COMPARISON: GMRES vs LU")
println("    For n+d Scattering Calculation")
println("="^70)
println()

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

# Three-body quantum numbers for ³H system
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
MT = -0.5

# Basis parameters
lmax = 8
lmin = 0
λmax = 20
λmin = 0
j2bmax = 1.0

# Particle spins and isospins
s1 = 0.5
s2 = 0.5
s3 = 0.5
t1 = 0.5
t2 = 0.5
t3 = 0.5

# Mesh parameters
nθ = 12
nx = 20
ny = 20
xmax = 16
ymax = 16
alpha = 1

# Nuclear potential
potname = "AV18"

# Scattering parameters
E_scatt = 3.0   # Scattering energy (MeV)
z1z2 = 0.0      # n+d: both neutral
θ_deg = 0.0     # No complex scaling for comparison

println("System configuration:")
println("  Jtot = $Jtot, T = $T, Parity = $Parity, MT = $MT")
println("  lmax = $lmax, λmax = $λmax, j2bmax = $j2bmax")
println("  Grid: $(nx)×$(ny) points")
println("  Potential: $potname")
println("  Scattering energy: E = $E_scatt MeV")
println()

# ============================================================================
# SETUP: Generate channels, mesh, and initial state
# ============================================================================

println("Setting up calculation...")
println("-"^70)

# Generate channels
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin,
        s1, s2, s3, t1, t2, t3, MT, j2bmax)
println("✓ Generated $(α.nchmax) channels")

# Generate mesh
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("✓ Mesh initialized: $(nx)×$(ny) grid")

# Compute deuteron bound state
e2b, ψ2b = bound2b(grid, potname)
println("✓ Deuteron binding energy: $(round(e2b[1], digits=6)) MeV")

# Extract deuteron wavefunction
φ_d_matrix = ComplexF64.(ψ2b[1])

# Compute initial state vector
φ_θ = compute_initial_state_vector(grid, α, φ_d_matrix, E_scatt, z1z2, θ=θ_deg)
println("✓ Initial state computed: ||φ|| = $(round(norm(φ_θ), digits=6))")

# Compute scattering matrix (shared by both methods)
println("\nComputing scattering matrix A = E*B - T - V*(I + Rxy)...")
A, B, T, V, Rxy, Rxy_31, Rxy_32, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny =
    compute_scattering_matrix(E_scatt, α, grid, potname, θ_deg=θ_deg)

# Compute right-hand side (shared by both methods)
println("Computing source term b = 2*V*Rxy_31*φ...")
b = compute_VRxy_phi(V, Rxy_31, φ_θ)

println("✓ Setup complete")
println("  System size: $(length(b))")
println("  ||b|| = $(round(norm(b), digits=6))")
println()

# ============================================================================
# METHOD 1: LU FACTORIZATION
# ============================================================================

println("="^70)
println("    METHOD 1: LU FACTORIZATION (DIRECT SOLVER)")
println("="^70)
println()

println("Solving A*c = b using LU factorization...")
time_lu = @elapsed begin
    c_lu = A \ b
end

# Compute residual
residual_lu = A * c_lu - b
rel_residual_lu = norm(residual_lu) / norm(b)

println("\n✓ LU solution obtained")
println("  Computation time: $(round(time_lu, digits=3)) seconds")
println("  Solution norm: ||c|| = $(round(norm(c_lu), digits=6))")
println("  Residual: ||A*c - b|| = $(round(norm(residual_lu), digits=10))")
println("  Relative residual: ||A*c - b|| / ||b|| = $(round(rel_residual_lu, digits=12))")
println()

# ============================================================================
# METHOD 2: GMRES WITH PRECONDITIONER
# ============================================================================

println("="^70)
println("    METHOD 2: GMRES (ITERATIVE SOLVER WITH PRECONDITIONER)")
println("="^70)
println()

println("Computing M^{-1} preconditioner...")
println("  M = E*B - T - V_αα (diagonal potential only)")
M_inv_func = matrices.M_inverse_operator(α, grid, E_scatt, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)

# Wrap the function in a PreconditionerOperator for GMRES
M_inv_op = PreconditionerOperator{ComplexF64}(M_inv_func, length(b))
println("✓ Preconditioner constructed and wrapped")
println()

println("Solving M^{-1}*A*c = M^{-1}*b using GMRES...")
println("  Parameters: maxiter=200, reltol=1e-6")
println()

time_gmres = @elapsed begin
    c_gmres, history = gmres(A, b, Pl=M_inv_op, log=true, verbose=true, maxiter=200, reltol=1e-6)
end

# Compute residual
residual_gmres = A * c_gmres - b
rel_residual_gmres = norm(residual_gmres) / norm(b)

println("\n✓ GMRES solution obtained")
println("  Computation time: $(round(time_gmres, digits=3)) seconds")
println("  Iterations: $(history.iters)")
println("  Solution norm: ||c|| = $(round(norm(c_gmres), digits=6))")
println("  Residual: ||A*c - b|| = $(round(norm(residual_gmres), digits=10))")
println("  Relative residual: ||A*c - b|| / ||b|| = $(round(rel_residual_gmres, digits=12))")
println()

# ============================================================================
# COMPARISON
# ============================================================================

println("="^70)
println("    COMPARISON OF METHODS")
println("="^70)
println()

# Solution difference
diff_solution = c_lu - c_gmres
rel_diff = norm(diff_solution) / norm(c_lu)

println("Solution Accuracy:")
println("-"^70)
@printf("  %-30s: %.3e\n", "LU relative residual", rel_residual_lu)
@printf("  %-30s: %.3e\n", "GMRES relative residual", rel_residual_gmres)
@printf("  %-30s: %.3e\n", "||c_LU - c_GMRES||", norm(diff_solution))
@printf("  %-30s: %.3e\n", "Relative difference", rel_diff)
println()

println("Computation Time:")
println("-"^70)
@printf("  %-30s: %.3f seconds\n", "LU time", time_lu)
@printf("  %-30s: %.3f seconds\n", "GMRES time", time_gmres)
@printf("  %-30s: %.2fx\n", "Speedup (LU/GMRES)", time_lu / time_gmres)
println()

println("Memory and Efficiency:")
println("-"^70)
@printf("  %-30s: %d\n", "System size", length(b))
@printf("  %-30s: %d\n", "GMRES iterations", history.iters)
@printf("  %-30s: %.3f sec/iter\n", "Time per iteration", time_gmres / history.iters)
println()

println("Solution Verification:")
println("-"^70)
# Check channel components
Ntot = length(c_lu)
n_channels = α.nchmax
points_per_channel = grid.nx * grid.ny

max_channel_diff = 0.0
max_channel_idx = 0
for iα in 1:n_channels
    idx_start = (iα-1)*points_per_channel + 1
    idx_end = iα*points_per_channel

    c_lu_ch = c_lu[idx_start:idx_end]
    c_gmres_ch = c_gmres[idx_start:idx_end]

    ch_diff = norm(c_lu_ch - c_gmres_ch) / norm(c_lu_ch)
    if ch_diff > max_channel_diff
        global max_channel_diff = ch_diff
        global max_channel_idx = iα
    end
end

@printf("  %-30s: %.3e\n", "Max channel difference", max_channel_diff)
@printf("  %-30s: %d\n", "Channel with max difference", max_channel_idx)
println()

# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

println("="^70)
println("    GMRES CONVERGENCE HISTORY")
println("="^70)
println()

println("Iteration    Residual")
println("-"^40)
for (i, res) in enumerate(history)
    @printf("%9d    %.6e\n", i, res)
    if i >= 10 && history.iters > 15
        println("    ... (showing first 10 iterations)")
        println("    ... (total $(history.iters) iterations)")
        break
    end
end
println()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

println("="^70)
println("    RECOMMENDATIONS")
println("="^70)
println()

if time_gmres < time_lu
    speedup = time_lu / time_gmres
    println("✓ GMRES is FASTER by $(round(speedup, digits=2))x")
    println("  Recommended for large systems")
else
    slowdown = time_gmres / time_lu
    println("✓ LU is FASTER by $(round(slowdown, digits=2))x")
    println("  Recommended for this system size ($(length(b)) unknowns)")
    println("  Note: GMRES becomes advantageous for larger systems")
end
println()

if rel_diff < 1e-6
    println("✓ Both methods agree to high precision (rel. diff < 1e-6)")
    println("  Either method is suitable for production calculations")
elseif rel_diff < 1e-4
    println("⚠ Methods agree to moderate precision (rel. diff < 1e-4)")
    println("  Results are acceptable but consider tighter GMRES tolerance")
else
    println("⚠ WARNING: Methods show significant disagreement!")
    println("  Check GMRES convergence and increase maxiter or tighten reltol")
end
println()

println("System Size Analysis:")
println("  Current system: $(length(b)) unknowns")
if length(b) < 10000
    println("  → LU is typically faster for systems < 10,000")
elseif length(b) < 50000
    println("  → GMRES may be competitive with good preconditioner")
else
    println("  → GMRES strongly recommended for systems > 50,000")
end
println()

println("="^70)
println("    COMPARISON COMPLETE")
println("="^70)
println()

# Save results for further analysis
println("Results stored in variables:")
println("  c_lu        - LU solution")
println("  c_gmres     - GMRES solution")
println("  history     - GMRES convergence history")
println("  A, b        - System matrix and RHS")
println()
