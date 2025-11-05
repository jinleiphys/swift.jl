using LinearAlgebra
using IterativeSolvers  # Required for GMRES method

# PERFORMANCE: Use all available CPU cores for BLAS operations
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS threads set to: $(BLAS.get_num_threads()) (CPU cores: $(Sys.CPU_THREADS))")
println()

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
println("    N+D SCATTERING CALCULATION")
println("    Three-body scattering using Faddeev equations")
println("="^70)
println()

# ============================================================================
# SYSTEM PARAMETERS (same as swift_3H.jl)
# ============================================================================

# Three-body quantum numbers for ³H system
fermion = true
Jtot = 0.5      # Total angular momentum
T = 0.5         # Total isospin
Parity = 1      # Parity
MT = -0.5       # Isospin projection (-0.5 for 3H: n+n+p)

# Basis parameters
lmax = 8        # Maximum two-body orbital angular momentum
lmin = 0
λmax = 20       # Maximum hypermomentum
λmin = 0
j2bmax = 1.0    # Maximum J12 (two-body angular momentum)

# Particle spins and isospins
s1 = 0.5
s2 = 0.5
s3 = 0.5
t1 = 0.5
t2 = 0.5
t3 = 0.5

# Mesh parameters (converged values for j2bmax=2.0)
nθ = 12
nx = 20
ny = 20
xmax = 16
ymax = 16
alpha = 1

# Nuclear potential
potname = "AV18"

# Scattering energy (MeV) - kinetic energy of neutron relative to deuteron
E_scatt = 3.0   # Can be adjusted

# Coulomb parameter (z1*z2 for n+d scattering)
z1z2 = 0.0      # Neutron-deuteron: both neutral

# Complex scaling angle (for resonance calculations)
θ_deg = 10.0     # Standard calculation (no complex scaling)

println("System configuration:")
println("  Jtot = $Jtot, T = $T, Parity = $Parity, MT = $MT")
println("  lmax = $lmax, λmax = $λmax, j2bmax = $j2bmax")
println("  Potential: $potname")
println("  Scattering energy: E = $E_scatt MeV")
println("  Coulomb parameter: z1*z2 = $z1z2")
println("  Complex scaling: θ = $θ_deg degrees")
println()

# ============================================================================
# STEP 1: GENERATE THREE-BODY CHANNELS
# ============================================================================

println("Step 1: Generating three-body channels...")
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin,
        s1, s2, s3, t1, t2, t3, MT, j2bmax)
println("  Generated $(α.nchmax) channels")
println()

# ============================================================================
# STEP 2: GENERATE NUMERICAL MESH
# ============================================================================

println("Step 2: Generating numerical mesh...")
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("  Grid: $(nx)×$(ny) points, xmax=$(xmax) fm, ymax=$(ymax) fm")
println("  Angular points: nθ=$(nθ)")
println()

# ============================================================================
# STEP 3: COMPUTE DEUTERON BOUND STATE (initial state)
# ============================================================================

println("Step 3: Computing deuteron bound state...")
e2b, ψ2b = bound2b(grid, potname)
println("  Deuteron binding energy: $(e2b[1]) MeV")
println("  (Experimental: -2.224575 MeV)")
println()

# Extract deuteron ground state wavefunction (J12=1)
φ_d_matrix = ComplexF64.(ψ2b[1])  # Ground state with ³S₁ + ³D₁ components

# ============================================================================
# STEP 4: COMPUTE INITIAL STATE VECTOR
# ============================================================================

println("Step 4: Computing initial state vector...")
println("  Computing φ(θ) = [φ_d(x) F_λ(ky)] / [ϕx ϕy]...")
φ_θ = compute_initial_state_vector(grid, α, φ_d_matrix, E_scatt, z1z2, θ=θ_deg)

# Verify initial state
println("  Initial state vector size: $(length(φ_θ))")
println("  Initial state norm: $(norm(φ_θ))")
println()

# ============================================================================
# STEP 5: SOLVE SCATTERING EQUATION
# ============================================================================

println("Step 5: Solving inhomogeneous scattering equation...")

# Choose solution method
method = :lu      # Options: :lu (direct) or :gmres (iterative with preconditioner)

println("  Solution method: $method")
println()

# Solve: [E*B - T - V*(I + Rxy)] c = 2*V*Rxy_31*φ
c, A, b = solve_scattering_equation(E_scatt, α, grid, potname, φ_θ,
                                     θ_deg=θ_deg,
                                     method=method)

println("\nSolution obtained:")
println("  Solution vector size: $(length(c))")
println("  Solution norm: $(norm(c))")
println("  RHS norm: $(norm(b))")
println()

# ============================================================================
# STEP 6: ANALYSIS AND OUTPUT
# ============================================================================

println("="^70)
println("    SCATTERING CALCULATION SUMMARY")
println("="^70)
println()

println("System: n + d → n + d")
println("  Scattering energy: E = $E_scatt MeV")
println("  Deuteron binding: E_d = $(e2b[1]) MeV")
println("  Total energy: E_tot = $(E_scatt + e2b[1]) MeV")
println()

println("Three-body configuration:")
println("  Number of channels: $(α.nchmax)")
println("  Grid points: $(nx)×$(ny)")
println("  Matrix size: $(α.nchmax * nx * ny)")
println()

println("Solution convergence:")
println("  ||c|| = $(norm(c))")
println("  ||b|| = $(norm(b))")
println("  ||A*c - b|| / ||b|| = $(norm(A*c - b) / norm(b))")
println()

# Extract channel components of solution
println("Channel analysis:")
Ntot = α.nchmax * grid.nx * grid.ny
for iα in 1:min(5, α.nchmax)  # Show first 5 channels
    idx_start = (iα-1)*grid.nx*grid.ny + 1
    idx_end = iα*grid.nx*grid.ny
    c_channel = c[idx_start:idx_end]
    channel_norm = norm(c_channel)
    println("  Channel $iα: ||c_α|| = $(round(channel_norm, digits=6))")
end
if α.nchmax > 5
    println("  ... ($(α.nchmax - 5) more channels)")
end
println()

println("="^70)
println("    N+D SCATTERING CALCULATION COMPLETE")
println("="^70)
println()

# Save results (optional)
println("Results stored in variables:")
println("  c      - scattering wavefunction")
println("  φ_θ    - initial state vector")
println("  α      - channel structure")
println("  grid   - numerical mesh")
println("  A      - scattering matrix")
println("  b      - source term")
println()
