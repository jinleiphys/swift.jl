# Test script for M_inverse function
# This demonstrates how to use the new M_inverse function with pre-computed components

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra

println("Testing M_inverse implementation...")

# Example parameters (small for testing) - similar to swift_3H.ipynb but smaller
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
lmax = 2      # Reduced from 4 for faster testing
lmin = 0
λmax = 4      # Reduced from 20 for faster testing
λmin = 0
s1 = 0.5
s2 = 0.5
s3 = 0.5
t1 = 0.5
t2 = 0.5
t3 = 0.5
MT = -0.5     # For ³H (one proton, two neutrons)
j2bmax = 2.0  # Reduced from 3.0 for faster testing

# Create small grid for testing
nθ = 12
nx = 5        # Reduced from 30 for faster testing
ny = 5        # Reduced from 30 for faster testing
xmax = 10.0   # Reduced from 20 for faster testing
ymax = 10.0   # Reduced from 20 for faster testing
alpha = 0.5

println("\nGenerating channels...")
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
println("Number of channels: ", α.nchmax)

println("\nInitializing mesh...")
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("Grid size: $nx × $ny")

# Choose potential
potname = "AV18"

# Energy value for testing
E = -8.0  # MeV (typical binding energy)

println("\nComputing T matrix with components...")
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
println("T matrix computed: size = ", size(Tmat))

println("\nComputing V matrix with components...")
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
println("V matrix computed: size = ", size(Vmat))

println("\nComputing M inverse...")
M_inv = M_inverse(α, grid, E, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
println("M^{-1} computed: size = ", size(M_inv))

println("\nValidation checks:")
println("  M_inv is square: ", size(M_inv, 1) == size(M_inv, 2))
println("  Expected size: ", α.nchmax * nx * ny)
println("  Actual size: ", size(M_inv, 1))
println("  Matrix norm: ", norm(M_inv))

# Check if M_inv is approximately correct by computing M * M_inv
# First we need to construct M
println("\nConstructing M matrix for validation...")
B = Bmatrix(α, grid)
M = E * B - Tmat - Vmat

println("\nComputing M * M_inv (should be ≈ I)...")
prod = M * M_inv
residual = prod - Matrix{Float64}(I, size(prod))
println("  Residual norm: ", norm(residual))
println("  Max residual: ", maximum(abs.(residual)))

if norm(residual) < 1e-10
    println("\n✓ M_inverse implementation validated successfully!")
else
    println("\n⚠ Warning: residual is larger than expected")
    println("  This may be due to numerical precision or matrix conditioning")
end

println("\nTest complete!")
