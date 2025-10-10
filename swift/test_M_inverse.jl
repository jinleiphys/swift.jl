# Test script for M_inverse function
# This demonstrates how to use the new M_inverse function with pre-computed components

include("../general_modules/channels.jl")
using .channels
include("../general_modules/mesh.jl")
using .Mesh
include("matrices.jl")
using .matrices

println("Testing M_inverse implementation...")

# Example parameters (small for testing)
J = 1/2
T = 1/2
P = 1
s1 = 1/2
s2 = 1/2
s3 = 1/2
t3 = 1/2
MT = 1/2

# Create small grid for testing
nx = 5
ny = 5
xmax = 10.0
ymax = 10.0
α_param = 0.5

println("\nGenerating channels...")
α = α3b(J, T, P, s1, s2, s3, t3, MT)
println("Number of channels: ", α.nchmax)

println("\nInitializing mesh...")
grid = initialmesh(nx, ny, xmax, ymax, α_param)
println("Grid size: $nx × $ny")

# Choose potential
potname = "av18"

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
