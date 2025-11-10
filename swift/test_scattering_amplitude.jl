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

# Simple fermion system
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
MT = -0.5
lmax = 1
lmin = 0
λmax = 1
λmin = 0
j2bmax = 1.0
s1 = 0.5; s2 = 0.5; s3 = 0.5
t1 = 0.5; t2 = 0.5; t3 = 0.5
nθ = 6; nx = 8; ny = 8; xmax = 8; ymax = 8; alpha = 1

α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

println("System: $(α.nchmax) channels")
println("Grid: $(nx)×$(ny) mesh")
println()

# Scattering energy
E = 10.0  # MeV
z1z2 = 1.0  # Charge product for p+d

println("Computing matrices...")
V = V_matrix_optimized(α, grid, "AV18")
Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized(α, grid)
println()

# Create mock initial state vector ψ_in in α₃ coordinates
# In real calculation, this would be: ψ_in = Rxy_31 * φ (or Rxy_13 * φ)
println("Creating mock initial state ψ_in in α₃ coordinates...")
N = α.nchmax * grid.nx * grid.ny
ψ_in = zeros(ComplexF64, N)

# Populate a few channels with Gaussian-like functions
for iα in 1:min(2, α.nchmax)
    for ix in 1:grid.nx
        for iy in 1:grid.ny
            i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy
            x = grid.xi[ix]
            y = grid.yi[iy]
            # Simple Gaussian-like form
            ψ_in[i] = exp(-0.5*x - 0.5*y) / (grid.ϕx[ix] * grid.ϕy[iy])
        end
    end
end
println("  Initial state ψ_in created, norm = $(norm(ψ_in))")
println()

# Create mock scattering solution ψ_sc
# In real calculation, this would come from solve_scattering_equation
println("Creating mock scattering solution ψ_sc...")
ψ_sc = 0.1 * randn(ComplexF64, N)
println("  Scattering solution created, norm = $(norm(ψ_sc))")
println()

# Compute scattering amplitude
println("="^70)
f_k = compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E)
println("="^70)
println()

println("RESULTS:")
println("--------")
@printf("Scattering amplitude: f(k) = %.6e + %.6e i\n", real(f_k), imag(f_k))
@printf("Magnitude: |f(k)| = %.6e fm\n", abs(f_k))
@printf("Phase: arg(f(k)) = %.6f rad\n", angle(f_k))
println()
@printf("Initial state: ||ψ₃^(in)|| = %.6e\n", norm(ψ_in))
@printf("Scattering state: ||ψ₃^(sc)|| = %.6e\n", norm(ψ_sc))
@printf("Total state: ||ψ₃^(in) + ψ₃^(sc)|| = %.6e\n", norm(ψ_in + ψ_sc))
println()

println("="^70)
println("TEST COMPLETED")
println("="^70)
