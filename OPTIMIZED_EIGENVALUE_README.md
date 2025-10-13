# Optimized Lambda Eigenvalue Function

## Overview

The `compute_lambda_eigenvalue_optimized` function provides a reformulated approach to computing eigenvalues in the Malfiet-Tjon iterative solver for the Faddeev equations. It uses a preconditioner-based formulation for better numerical conditioning and performance.

## Mathematical Formulation

### Original Method
The original `compute_lambda_eigenvalue` solves:
```
λ(E) [c] = [E*B - T - V]⁻¹ * (V*R + UIX) [c]
```

### Optimized Method
The optimized version reformulates this as:
```
λ(E) [c] = M⁻¹(E) * (V - V_αα + V*R + UIX) [c]
```

where:
- **M⁻¹ = [E*B - T - V_αα]⁻¹** is the preconditioner
- **V_αα** is the diagonal (within-channel) part of the potential
- **V - V_αα** is the off-diagonal (channel-coupling) part

## Key Differences

### 1. Preconditioner Formulation
- **Original**: Inverts `[E*B - T - V]` where V includes all channel couplings
- **Optimized**: Inverts `[E*B - T - V_αα]` where V_αα only includes diagonal potential

### 2. Physical Interpretation
The reformulation separates:
- **V_αα**: Within-channel interaction (included in preconditioner M⁻¹)
- **V - V_αα**: Channel-to-channel coupling (treated as perturbation in RHS)

This is similar to the M⁻¹ preconditioner already used in GMRES solvers, but applied to the eigenvalue problem.

### 3. Computational Efficiency
- **M⁻¹ computation**: Faster because V_αα is diagonal (block-diagonal structure)
- **Matrix-free operator**: Uses composition K(x) = M⁻¹(RHS*x) without forming full matrix
- **Memory efficient**: Avoids storing the full preconditioned matrix

## Implementation Details

### Function Signature
```julia
compute_lambda_eigenvalue_optimized(E0, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                   verbose=false, use_arnoldi=true, krylov_dim=50, arnoldi_tol=1e-6,
                                   previous_eigenvector=nothing, V_UIX=nothing)
```

### Key Steps

1. **Compute M⁻¹ preconditioner** using `M_inverse_operator()`
   - Efficient eigendecomposition-based implementation
   - Returns a function that applies M⁻¹ to vectors

2. **Build V_αα** (diagonal potential)
   - Extracted from pre-computed `V_x_diag_ch` components
   - Block-diagonal structure: V_αα[α] = V_x_diag_ch[α] ⊗ I_y

3. **Form RHS matrix**: `(V - V_αα) + V*R + UIX`
   - Separates diagonal and off-diagonal potential contributions
   - Includes three-body force if provided

4. **Define matrix-free operator**: K(x) = M⁻¹(RHS * x)
   - Applies operations sequentially without forming full matrix
   - Memory efficient for large problems

5. **Arnoldi eigenvalue computation**
   - Uses iterative Arnoldi method with adaptive Krylov dimension
   - Multiple initial vector strategies for robustness
   - Falls back to direct method if needed

## Performance Comparison

### Test Results (10×10 grid, 16 channels, Malfiet-Tjon potential)

| Energy (MeV) | Original Time | Optimized Time | Speedup |
|--------------|---------------|----------------|---------|
| -8.0         | 4.29 s        | 4.94 s         | 0.87×   |
| -7.5         | 0.69 s        | 3.47 s         | 0.20×   |
| -7.0         | 0.67 s        | 0.45 s         | **1.49×** |

**With UIX three-body force**:
| Energy (MeV) | Original Time | Optimized Time | Speedup |
|--------------|---------------|----------------|---------|
| -8.5         | 0.72 s        | 0.61 s         | **1.18×** |

### Numerical Accuracy
- Eigenvalue agreement: < 10⁻⁵ relative difference
- Eigenvector overlap: > 0.999999
- Fully compatible with UIX three-body forces

## When to Use

### Advantages of Optimized Version
1. **Better numerical conditioning** when channel coupling is strong
2. **Consistent with GMRES preconditioning** used elsewhere in the code
3. **Matrix-free implementation** saves memory for large problems
4. **Potentially faster** for problems where M⁻¹ is much cheaper than [E*B - T - V]⁻¹

### Advantages of Original Version
1. **Simpler conceptual model** - one operator to invert
2. **Better for first iteration** when no previous eigenvector available
3. **May be faster** for small problems with cheap factorizations

## Usage Example

```julia
using .channels, .mesh, .matrices, .MalflietTjon

# Setup
α = α3b(true, 0.5, 0.5, 1, 2, 0, 4, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 2.0)
grid = initialmesh(12, 10, 10, 12.0, 12.0, 0.5)

# Build matrices with components
T, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
V, V_x_diag_ch = V_matrix(α, grid, "AV18", return_components=true)
B = Bmatrix(α, grid)
Rxy, Rxy_31, Rxy_32 = Rxy_matrix(α, grid)

# Compute eigenvalue at energy E0
E0 = -8.0
λ, eigenvec = compute_lambda_eigenvalue_optimized(
    E0, T, V, B, Rxy, α, grid, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
    verbose=true, use_arnoldi=true
)

println("λ(E=$E0) = $λ")
```

## Integration with Malfiet-Tjon Solver

The optimized function can be integrated into the existing Malfiet-Tjon solver by:

1. Replacing `compute_lambda_eigenvalue` calls with `compute_lambda_eigenvalue_optimized`
2. Ensuring `return_components=true` when building T and V matrices
3. Passing the component arrays (Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny) to the function

This would enable direct comparison between the two formulations and allow choosing the best approach for specific problems.

## Technical Notes

### Preconditioner Structure
The M⁻¹ operator has the form:
```
M⁻¹ = [E*B - T - V_αα]⁻¹
```

This is block-diagonal with respect to channels, and within each channel block:
```
M_α⁻¹ = U_α * D_α⁻¹ * U_α^T * N_α⁻¹
```

where:
- U_α are eigenvectors of the diagonal Hamiltonian H_α = T_α + V_αα[α]
- D_α⁻¹ has diagonal elements 1/(E - eigenvalue_α)
- N_α⁻¹ is the overlap matrix inverse

### Matrix-Free Implementation
The operator K(E) is implemented as a function composition:
```julia
K = x -> M_inv_op(RHS_matrix * x)
```

This avoids forming the full matrix M⁻¹ * RHS_matrix, saving memory and potentially improving performance.

### Convergence Properties
The reformulation can affect convergence behavior:
- **Better conditioning**: Separating diagonal/off-diagonal parts can improve spectral properties
- **Different eigenvalue spectrum**: The spectrum of M⁻¹ * (V - V_αα + V*R) differs from [E*B - T - V]⁻¹ * (V*R)
- **Same physical solution**: Both formulations converge to the same ground state energy

## Future Improvements

1. **Adaptive method selection**: Automatically choose between original and optimized based on problem properties
2. **Hybrid approach**: Use optimized method for some iterations, original for others
3. **Improved V_αα extraction**: More efficient construction of diagonal potential blocks
4. **Performance profiling**: Detailed analysis of where time is spent in each formulation
5. **Integration with malfiet_tjon_solve**: Create `malfiet_tjon_solve_with_preconditioner` wrapper

## References

- Original Malfiet-Tjon method: MalflietTjon.jl
- M⁻¹ preconditioner: matrices.jl, M_inverse_operator()
- GMRES with preconditioning: gmres_matfree() in MalflietTjon.jl
- Test script: test_optimized_eigenvalue.jl
