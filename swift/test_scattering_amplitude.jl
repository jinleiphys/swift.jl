using LinearAlgebra
using Printf

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
include("matrices_optimized.jl")
include("scattering.jl")
include("twobody.jl")

using .channels
using .mesh
using .matrices_optimized
using .Scattering
using .twobodybound

println("="^70)
println("    TEST: Scattering Amplitude Computation")
println("="^70)
println()

# Deuteron scattering system (n+d elastic scattering)
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
MT = -0.5
lmax = 2      # Need l=0,2 for deuteron (³S₁ + ³D₁)
lmin = 0
λmax = 2      # Include higher λ values
λmin = 0
j2bmax = 1.0  # J12=1 for deuteron
s1 = 0.5; s2 = 0.5; s3 = 0.5
t1 = 0.5; t2 = 0.5; t3 = 0.5
nθ = 12; nx = 24; ny = 24; xmax = 20; ymax = 20; alpha = 1

α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

println("System: $(α.nchmax) three-body channels")
println("Grid: $(nx)×$(ny) mesh")
println()

# Scattering energy and charges
E = 1.0   # MeV
z1z2 = 0.0 # Charge product for n+d (no Coulomb interaction)
θ_deg = 0.0    # Complex scaling angle in degrees

println("Computing matrices...")
V = V_matrix_optimized(α, grid, "AV18")
Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)
println("  Matrices computed")
println()

# Compute deuteron bound state using bound2b with same complex scaling angle
println("Computing deuteron bound state with bound2b...")
println("  Complex scaling angle: θ = $(θ_deg)°")
bound_energies, bound_wavefunctions = bound2b(grid, "AV18", θ_deg=θ_deg)

# Extract the ground state (deuteron)
if isempty(bound_energies)
    error("No bound states found! Check potential and mesh parameters.")
end

φ_d_matrix = ComplexF64.(bound_wavefunctions[1])  # Ground state with ³S₁ + ³D₁ components
E_deuteron = real(bound_energies[1])

println("  Deuteron binding energy: $(round(E_deuteron, digits=4)) MeV")
println("  Deuteron wavefunction computed")
println()

# Create initial state vector φ using compute_initial_state_vector
println("Computing initial state vector φ...")
θ_rad = θ_deg * π / 180.0  # Convert to radians
ψ_in = compute_initial_state_vector(grid, α, φ_d_matrix, E, z1z2, θ=θ_rad)

# Check which channels are populated (only deuteron channels should be non-zero)
n_populated = 0
for iα in 1:α.nchmax
    idx_start = (iα-1) * grid.nx * grid.ny + 1
    idx_end = iα * grid.nx * grid.ny
    channel_norm = norm(ψ_in[idx_start:idx_end])
    if channel_norm > 1e-10
        global n_populated += 1
    end
end

println("  Initial state φ created")
println("  Populated channels: $n_populated / $(α.nchmax) (only deuteron channels should be populated)")
println()

# Solve scattering equation to get ψ_sc
# Solves: [E*B - T - V*(I + Rxy)] ψ_sc = 2*V*Rxy_31*φ
ψ_sc, A_matrix, b_vector = solve_scattering_equation(E, α, grid, "AV18", ψ_in,
                                                       θ_deg=θ_deg, method=:lu)
println("  Scattering solution ψ_sc computed via LU factorization")
println("  Residual norm: ||A*ψ_sc - b|| = $(norm(A_matrix * ψ_sc - b_vector))")
println()

# Compute scattering amplitude matrix
println("="^70)
f_matrix, deuteron_channels, channel_labels = compute_scattering_amplitude(
    ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix, z1z2, θ=θ_rad, σ_l=0.0
)
println("="^70)
println()

println("SCATTERING AMPLITUDE MATRIX RESULTS:")
println("="^70)
println("Deuteron channels found: $(length(deuteron_channels))")
for (i, label) in enumerate(channel_labels)
    println("  α₀=$i: $label (three-body channel α=$(deuteron_channels[i]))")
end
println()

println("Scattering amplitude matrix f_{α₀_out, α₀_in} (fm):")
n_d = size(f_matrix, 1)
for i_out in 1:n_d
    for i_in in 1:n_d
        @printf("  f[%d,%d] = %+.4e %+.4e i  (|f| = %.4e)\n",
                i_out, i_in, real(f_matrix[i_out, i_in]),
                imag(f_matrix[i_out, i_in]), abs(f_matrix[i_out, i_in]))
    end
end
println()

# Compute wave number for phase shift analysis
ħ = 197.3269718  # MeV·fm
m = 1.0079713395678829  # amu
amu = 931.49432  # MeV
μ = (2.0 * m) / 3.0
k = sqrt(2.0 * μ * amu * E) / ħ
println("Wave number k = $(k) fm⁻¹")
println()

# Phase shift analysis
println("="^70)
println("PHASE SHIFT ANALYSIS")
println("="^70)
phase_results = compute_phase_shift_analysis(f_matrix, k, α, deuteron_channels, channel_labels)
println()

# Display summary
println("="^70)
println("SUMMARY")
println("="^70)
for ((J_val, parity), result) in phase_results
    parity_symbol = parity == 1 ? "+" : "-"
    println("\nJ^π = $(J_val)^$parity_symbol:")

    eigenphases = result["eigenphases"]
    mixing_params = result["mixing_params"]

    println("  Eigenphase shifts:")
    for (i, δ) in enumerate(eigenphases)
        @printf("    δ_%d = %.4f° (%.6f rad)\n", i, rad2deg(δ), δ)
    end

    if !isnothing(mixing_params)
        println("  Mixing parameters:")
        if mixing_params.ε != 0.0
            @printf("    ε = %.4f° (%.6f rad)\n", rad2deg(mixing_params.ε), mixing_params.ε)
        end
        if mixing_params.ζ != 0.0
            @printf("    ζ = %.4f° (%.6f rad)\n", rad2deg(mixing_params.ζ), mixing_params.ζ)
        end
        if mixing_params.η != 0.0
            @printf("    η = %.4f° (%.6f rad)\n", rad2deg(mixing_params.η), mixing_params.η)
        end
    end
end
println()

println("="^70)
println("TEST COMPLETED")
println("="^70)
