using LinearAlgebra
using Printf

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
include("matrices_optimized.jl")
include("scattering.jl")

using .channels
using .mesh
using .matrices_optimized
using .Scattering

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
nθ = 6; nx = 12; ny = 12; xmax = 10; ymax = 10; alpha = 1

α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

println("System: $(α.nchmax) three-body channels")
println("Grid: $(nx)×$(ny) mesh")
println()

# Scattering energy and charges
E = 10.0   # MeV
z1z2 = 0.0 # Charge product for n+d (no Coulomb interaction)
θ = 0.0    # Complex scaling angle

println("Computing matrices...")
V = V_matrix_optimized(α, grid, "AV18")
Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized(α, grid)
println("  Matrices computed")
println()

# Create mock deuteron bound state wavefunction
# In real calculation, this would come from bound2b()
println("Creating mock deuteron bound state φ_d...")
φ_d_matrix = zeros(ComplexF64, grid.nx, 2)  # 2 channels: ³S₁ and ³D₁

# Mock ³S₁ component (channel 1) - dominant component ~96%
for ix in 1:grid.nx
    x = grid.xi[ix]
    φ_d_matrix[ix, 1] = exp(-0.3*x) * sqrt(0.96)  # ~96% probability
end

# Mock ³D₁ component (channel 2) - small D-wave component ~4%
for ix in 1:grid.nx
    x = grid.xi[ix]
    φ_d_matrix[ix, 2] = exp(-0.3*x) * x * sqrt(0.04)  # ~4% probability
end

println("  Deuteron wavefunction created")
println("  ³S₁ component norm = $(norm(φ_d_matrix[:, 1]))")
println("  ³D₁ component norm = $(norm(φ_d_matrix[:, 2]))")
println()

# Create initial state vector φ using compute_initial_state_vector
println("Computing initial state vector φ...")
ψ_in = compute_initial_state_vector(grid, α, φ_d_matrix, E, z1z2, θ=θ)
println("  Initial state φ created")
println("  Norm ||φ|| = $(norm(ψ_in))")
println()

# Create mock scattering solution ψ_sc
# In real calculation, this would come from solve_scattering_equation
println("Creating mock scattering solution ψ_sc...")
N = α.nchmax * grid.nx * grid.ny
ψ_sc = 0.05 * randn(ComplexF64, N)  # Smaller perturbation
println("  Scattering solution created")
println("  Norm ||ψ_sc|| = $(norm(ψ_sc))")
println()

# Compute scattering amplitude matrix
println("="^70)
f_matrix, deuteron_channels, channel_labels = compute_scattering_amplitude(
    ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix, z1z2, θ=θ, σ_l=0.0
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
