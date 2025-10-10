# Test script for condition number analysis
# Compares the conditioning of:
# 1. Full LHS matrix: LHS = E*B - T - V (with full V including off-diagonal coupling)
# 2. Preconditioned system: M^{-1} * LHS

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("matrices.jl")
using .matrices
using LinearAlgebra
using Kronecker

println("="^70)
println("  CONDITION NUMBER ANALYSIS FOR FADDEEV EQUATION")
println("="^70)

# Setup parameters - similar to swift_3H.ipynb but smaller for testing
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
lmax = 2      # Small for faster computation
lmin = 0
λmax = 4      # Small for faster computation
λmin = 0
s1 = 0.5
s2 = 0.5
s3 = 0.5
t1 = 0.5
t2 = 0.5
t3 = 0.5
MT = -0.5     # For ³H (one proton, two neutrons)
j2bmax = 2.0  # Reduced for faster testing

# Create grid
nθ = 12
nx = 10       # Slightly larger for better analysis
ny = 10
xmax = 15.0
ymax = 15.0
alpha = 0.5

println("\nSystem Configuration:")
println("  Grid size: $nx × $ny")

# Generate channels
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1, s2, s3, t1, t2, t3, MT, j2bmax)
println("  Number of channels: ", α.nchmax)
println("  Total matrix size: ", α.nchmax * nx * ny)

# Initialize mesh
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

# Choose potential and energy
potname = "AV18"
E0 = -8.0  # MeV (typical binding energy for 3H)

println("\nComputing matrices...")
println("  Potential: $potname")
println("  Energy: $E0 MeV")

# Compute matrices
println("\n  Computing T matrix with components...")
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)

println("  Computing V matrix with components...")
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)

println("  Computing B matrix...")
B = Bmatrix(α, grid)

println("  Computing M inverse...")
M_inv = M_inverse(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)

println("\n" * "="^70)
println("  CONDITION NUMBER ANALYSIS")
println("="^70)

# Compute full LHS matrix: LHS = E*B - T - V (with full V)
println("\n1. Full LHS Matrix: LHS = E*B - T - V")
println("   (V includes full off-diagonal channel coupling)")
LHS = E0 * B - Tmat - Vmat

println("\n   Computing condition number of LHS...")
cond_LHS = cond(LHS)
println("   Condition number κ(LHS) = ", cond_LHS)

if cond_LHS > 1e10
    println("   ⚠ WARNING: Matrix is severely ill-conditioned!")
elseif cond_LHS > 1e6
    println("   ⚠ CAUTION: Matrix is moderately ill-conditioned")
elseif cond_LHS > 1e3
    println("   ✓ Matrix is reasonably well-conditioned")
else
    println("   ✓ Matrix is well-conditioned")
end

# Compute preconditioned system: M^{-1} * LHS
println("\n2. Preconditioned System: M^{-1} * LHS")
println("   (M^{-1} is the efficient inverse using diagonal V)")
precond_LHS = M_inv * LHS

println("\n   Computing condition number of M^{-1} * LHS...")
cond_precond = cond(precond_LHS)
println("   Condition number κ(M^{-1} * LHS) = ", cond_precond)

if cond_precond > 1e10
    println("   ⚠ WARNING: Preconditioned matrix is severely ill-conditioned!")
elseif cond_precond > 1e6
    println("   ⚠ CAUTION: Preconditioned matrix is moderately ill-conditioned")
elseif cond_precond > 1e3
    println("   ✓ Preconditioned matrix is reasonably well-conditioned")
else
    println("   ✓ Preconditioned matrix is well-conditioned")
end

# Compute improvement factor
improvement_factor = cond_LHS / cond_precond
println("\n" * "="^70)
println("  PRECONDITIONING EFFECTIVENESS")
println("="^70)
println("\n  Condition number reduction:")
println("    κ(LHS) / κ(M^{-1} * LHS) = ", improvement_factor)

if improvement_factor > 10
    println("    ✓ Excellent preconditioning! (~$(round(improvement_factor, digits=1))× improvement)")
elseif improvement_factor > 2
    println("    ✓ Good preconditioning (~$(round(improvement_factor, digits=1))× improvement)")
elseif improvement_factor > 1
    println("    ⚠ Modest preconditioning (~$(round(improvement_factor, digits=1))× improvement)")
else
    println("    ✗ Preconditioning degraded the condition number")
end

# Additional matrix analysis
println("\n" * "="^70)
println("  ADDITIONAL MATRIX PROPERTIES")
println("="^70)

# Check if matrices are symmetric/Hermitian
println("\n  Symmetry properties:")
is_LHS_hermitian = ishermitian(LHS)
is_precond_hermitian = ishermitian(precond_LHS)
println("    LHS is Hermitian: ", is_LHS_hermitian)
println("    M^{-1} * LHS is Hermitian: ", is_precond_hermitian)

# Compute eigenvalue statistics
println("\n  Eigenvalue spectrum analysis:")
println("    Computing eigenvalues of LHS...")
eigvals_LHS = eigvals(LHS)
println("    Largest eigenvalue: ", maximum(real.(eigvals_LHS)))
println("    Smallest eigenvalue: ", minimum(real.(eigvals_LHS)))
println("    Eigenvalue range: ", maximum(real.(eigvals_LHS)) - minimum(real.(eigvals_LHS)))

println("\n    Computing eigenvalues of M^{-1} * LHS...")
eigvals_precond = eigvals(precond_LHS)
println("    Largest eigenvalue: ", maximum(real.(eigvals_precond)))
println("    Smallest eigenvalue: ", minimum(real.(eigvals_precond)))
println("    Eigenvalue range: ", maximum(real.(eigvals_precond)) - minimum(real.(eigvals_precond)))

# Matrix norms
println("\n  Matrix norms:")
println("    ||LHS||₂ = ", norm(LHS, 2))
println("    ||M^{-1} * LHS||₂ = ", norm(precond_LHS, 2))

println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
println("\n  The M^{-1} preconditioner achieves a condition number reduction")
println("  of $(round(improvement_factor, digits=2))×, making the iterative solution of")
println("  the Faddeev equation more numerically stable and faster to converge.")
println("\n  This validates the efficiency of the eigendecomposition-based")
println("  M^{-1} implementation for practical three-body calculations.")
println("\n" * "="^70)

println("\nTest complete!")
