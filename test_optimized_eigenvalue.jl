# Test script for compute_lambda_eigenvalue_optimized
# Compares the original and optimized versions

println("Loading modules...")
include("general_modules/channels.jl")
include("general_modules/mesh.jl")
include("swift/matrices.jl")
include("swift/MalflietTjon.jl")

using .channels
using .mesh
using .matrices
using .MalflietTjon
using LinearAlgebra
using Printf

println("\n" * "="^70)
println("  Testing compute_lambda_eigenvalue_optimized")
println("="^70)

# Setup: small test case for speed
println("\n1. Setting up test configuration...")
α = α3b(true, 0.5, 0.5, 1, 2, 0, 4, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 2.0)
grid = initialmesh(12, 10, 10, 12.0, 12.0, 0.5)
potname = "MT"  # Malfiet-Tjon potential (fast for testing)

println("  - Channels: $(α.nchmax)")
println("  - Grid: $(grid.nx) × $(grid.ny)")
println("  - Potential: $potname")

# Build matrices
println("\n2. Building matrices...")
@time begin
    T, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
    V, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
    B = Bmatrix(α, grid)
    Rxy, Rxy_31, Rxy_32 = Rxy_matrix(α, grid)
end

# Test energy points
E_test = [-8.0, -7.5, -7.0]

println("\n3. Comparing eigenvalue computations at different energies...")
println("-"^70)

for E0 in E_test
    println("\nTesting at E = $E0 MeV:")
    println("  " * "-"^66)

    # Original method
    print("  Original method: ")
    @time λ_orig, eigvec_orig = compute_lambda_eigenvalue(
        E0, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
        verbose=false, use_arnoldi=true, krylov_dim=30
    )

    # Optimized method
    print("  Optimized method: ")
    @time λ_opt, eigvec_opt = compute_lambda_eigenvalue_optimized(
        E0, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
        verbose=false, use_arnoldi=true, krylov_dim=30
    )

    # Compare results
    λ_diff = abs(λ_orig - λ_opt)
    λ_rel_diff = λ_diff / abs(λ_orig)

    println()
    @printf("  λ (original):  %.10f\n", λ_orig)
    @printf("  λ (optimized): %.10f\n", λ_opt)
    @printf("  Difference:    %.2e (absolute), %.2e (relative)\n", λ_diff, λ_rel_diff)

    # Eigenvector comparison (inner product should be close to 1)
    if eigvec_orig !== nothing && eigvec_opt !== nothing
        eigvec_overlap = abs(eigvec_orig' * eigvec_opt)
        @printf("  Eigenvector overlap: %.10f (should be ≈ 1.0)\n", eigvec_overlap)

        if eigvec_overlap < 0.99
            @warn "Eigenvectors differ significantly!"
        end
    end

    # Check if results agree within tolerance
    tolerance = 1e-4
    if λ_rel_diff < tolerance
        println("  ✓ Results agree within tolerance ($tolerance)")
    else
        println("  ✗ Results differ by more than tolerance!")
    end
end

println("\n" * "="^70)
println("Test completed!")
println("="^70)

# Additional test: with UIX three-body force
println("\n4. Testing with UIX three-body force...")
println("-"^70)

# Compute UIX potential
uix_path = joinpath(dirname(@__FILE__), "3Npot", "UIX.jl")
include(uix_path)
V_UIX = UIX.full_UIX_potential(α, grid, Rxy_31, Rxy)

E0_uix = -8.5
println("\nTesting at E = $E0_uix MeV with UIX:")
println("  " * "-"^66)

# Original with UIX
print("  Original (UIX): ")
@time λ_orig_uix, eigvec_orig_uix = compute_lambda_eigenvalue(
    E0_uix, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
    verbose=false, use_arnoldi=true, krylov_dim=30, V_UIX=V_UIX
)

# Optimized with UIX
print("  Optimized (UIX): ")
@time λ_opt_uix, eigvec_opt_uix = compute_lambda_eigenvalue_optimized(
    E0_uix, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
    verbose=false, use_arnoldi=true, krylov_dim=30, V_UIX=V_UIX
)

# Compare
λ_diff_uix = abs(λ_orig_uix - λ_opt_uix)
λ_rel_diff_uix = λ_diff_uix / abs(λ_orig_uix)

println()
@printf("  λ (original):  %.10f\n", λ_orig_uix)
@printf("  λ (optimized): %.10f\n", λ_opt_uix)
@printf("  Difference:    %.2e (absolute), %.2e (relative)\n", λ_diff_uix, λ_rel_diff_uix)

if eigvec_orig_uix !== nothing && eigvec_opt_uix !== nothing
    eigvec_overlap_uix = abs(eigvec_orig_uix' * eigvec_opt_uix)
    @printf("  Eigenvector overlap: %.10f\n", eigvec_overlap_uix)
end

if λ_rel_diff_uix < 1e-4
    println("  ✓ UIX results agree within tolerance")
else
    println("  ✗ UIX results differ!")
end

println("\n" * "="^70)
println("All tests completed!")
println("="^70)
