# M⁻¹ Implementation Documentation

## Overview

This document describes the implementation of the efficient M⁻¹ (M-inverse) matrix computation for the Faddeev equation solver in `matrices.jl`.

## Mathematical Formulation

The M matrix is defined as:
```
M = E[I_α ⊗ N_x ⊗ N_y] - ∑_α [δ_αα ⊗ H_x^α ⊗ N_y] - ∑_α [δ_αα ⊗ N_x ⊗ T_y^α]
```

where:
- `H_x^α = T_x^α + V_x^α` (diagonal part of potential)
- `E` is the energy parameter
- `N_x, N_y` are overlap matrices
- `I_α` is the channel identity matrix

## Implementation Strategy

Instead of directly inverting the large M matrix, the implementation uses eigendecomposition:

### Step 1: Factorization
M can be factored as:
```
M = [I_α ⊗ N_x ⊗ N_y] · U · D · U⁻¹ · [I_α ⊗ N_x⁻¹ ⊗ N_y⁻¹]
```

where:
- `U = ⊕_α [U_x^α ⊗ U_y^α]` (block diagonal transformation)
- `D` is a diagonal matrix with elements `E - d_x^α[i] - d_y^α[j]`

### Step 2: Per-Channel Eigendecomposition
For each channel α:
```
N_x⁻¹ · H_x^α = U_x^α · diag(d_x^α) · (U_x^α)⁻¹
N_y⁻¹ · T_y^α = U_y^α · diag(d_y^α) · (U_y^α)⁻¹
```

### Step 3: Diagonal Inversion
The diagonal matrix D is trivially inverted element-wise:
```
D_inv[idx] = 1 / (E - d_x^α[ix] - d_y^α[iy])
```

### Step 4: Final Assembly
```
M⁻¹ = U · D⁻¹ · U⁻¹ · [I_α ⊗ N_x⁻¹ ⊗ N_y⁻¹]
```

## Code Structure

### Modified Functions

#### 1. `T_matrix(α, grid; return_components=false)`
**Enhanced to return per-channel components when `return_components=true`:**
- Returns: `(Tmatrix, Tx_channels, Ty_channels, Nx, Ny)`
- `Tx_channels[iα]`: Kinetic energy matrix in x-direction for channel α
- `Ty_channels[iα]`: Kinetic energy matrix in y-direction for channel α
- `Nx, Ny`: Overlap matrices

#### 2. `V_matrix(α, grid, potname; return_components=false)`
**Enhanced to return diagonal potential components when `return_components=true`:**
- Returns: `(Vmatrix, V_x_diag_channels)`
- `V_x_diag_channels[iα]`: Diagonal potential matrix V_x^α for channel α

#### 3. `M_inverse(α, grid, E, Tx_channels, Ty_channels, V_x_diag_channels, Nx, Ny)`
**New function that computes M⁻¹ efficiently:**
- Takes pre-computed components from T_matrix and V_matrix
- Returns the inverse matrix M⁻¹

## Usage Example

```julia
# Step 1: Compute T and V matrices with components
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)

# Step 2: Compute M inverse at energy E
E = -8.0  # MeV
M_inv = M_inverse(α, grid, E, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)

# Step 3: Use M_inv for Faddeev equation solving
# For example: solution = M_inv * source_term
```

## Performance Benefits

1. **Reuse of computed matrices**: T_x^α, T_y^α, and V_x^α are computed once and reused
2. **Efficient eigendecomposition**: Small per-channel eigenvalue problems instead of one large problem
3. **Diagonal inversion**: O(N) operation instead of O(N³) for full matrix inversion
4. **Memory efficiency**: Stores only per-channel eigenvectors, not the full transformation

## Computational Complexity

- **Per-channel eigendecomposition**: O(n_α × n_x³) + O(n_α × n_y³)
- **Full matrix construction**: O(n_α² × n_x² × n_y²)
- **Overall**: Much more efficient than direct inversion O((n_α × n_x × n_y)³)

For typical problem sizes:
- Direct inversion: ~O(N³) where N = n_α × n_x × n_y (e.g., N=1000 → 10⁹ operations)
- Eigendecomposition approach: ~O(n_α × n_x³) (e.g., n_α=10, n_x=20 → 80,000 operations)

## Backward Compatibility

The modifications maintain backward compatibility:
- `T_matrix(α, grid)` still works as before (returns only Tmatrix)
- `V_matrix(α, grid, potname)` still works as before (returns only Vmatrix)
- New functionality only activated with `return_components=true`

## Testing

Run the test script to validate the implementation:
```bash
julia swift/test_M_inverse.jl
```

The test verifies that M × M⁻¹ ≈ I within numerical precision.

## Integration with Faddeev Solvers

The M⁻¹ matrix can be integrated into:
1. **Iterative solvers**: Malfiet-Tjon method (replace direct matrix inversions)
2. **Preconditioners**: For conjugate gradient or GMRES methods
3. **Energy-dependent calculations**: Compute M⁻¹(E) for different energies efficiently

## Notes

- The implementation assumes diagonal potential approximation (V_x^α diagonal in channel space)
- For full off-diagonal coupling, a more general approach would be needed
- Numerical stability depends on the condition number of N_x⁻¹·H_x^α and N_y⁻¹·T_y^α
- Consider using more robust eigensolvers for ill-conditioned systems
