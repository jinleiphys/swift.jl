# Quick test: M_inverse speed after optimization

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra

println("="^70)
println("  M_INVERSE SPEED TEST (After Block-Diagonal Optimization)")
println("="^70)

# Test with small system first
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
lmax = 4
lmin = 0
λmax = 4
λmin = 0
s1 = 0.5
s2 = 0.5
s3 = 0.5
t1 = 0.5
t2 = 0.5
t3 = 0.5
MT = -0.5
j2bmax = 4.0

nθ = 12
nx = 30
ny = 30
xmax = 15.0
ymax = 15.0
alpha = 0.5

println("\nSmall system test:")
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("  Channels: ", α.nchmax)
println("  Grid: $nx × $ny")
println("  Matrix size: ", α.nchmax * nx * ny)

potname = "AV18"
E0 = -8.0

print("\n  Computing matrices... ")
@time begin
    Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
    Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
end

print("  Computing M_inverse... ")
@time M_inv = M_inverse(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)

println("  ✓ Small system completed")

# Test with larger system (production size)
println("\n" * "="^70)
println("Large system test (production size):")

lmax = 4
λmax = 4
nx = 30
ny = 30

α_large = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid_large = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))
println("  Channels: ", α_large.nchmax)
println("  Grid: $nx × $ny")
println("  Matrix size: ", α_large.nchmax * nx * ny)

print("\n  Computing matrices... ")
@time begin
    Tmat_large, Tx_ch_large, Ty_ch_large, Nx_large, Ny_large = T_matrix(α_large, grid_large, return_components=true)
    Vmat_large, V_x_diag_ch_large = V_matrix(α_large, grid_large, potname, return_components=true)
end

print("  Computing M_inverse... ")
@time M_inv_large = M_inverse(α_large, grid_large, E0, Tx_ch_large, Ty_ch_large, V_x_diag_ch_large, Nx_large, Ny_large)

println("  ✓ Large system completed")
println("\n" * "="^70)
println("  Summary: M_inverse now uses block-diagonal optimization")
println("  Expected: ~10-50× faster than naive implementation")
println("="^70)
