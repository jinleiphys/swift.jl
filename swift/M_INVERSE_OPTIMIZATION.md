# M^{-1} Implementation and Optimization Guide

## Mathematical Foundation

The M matrix for the Faddeev equation has the form:

```
M = E*I - ∑_α [δ_αα ⊗ H_x^α ⊗ N_y] - ∑_α [δ_αα ⊗ N_x ⊗ T_y^α]
```

where `H_x^α = T_x^α + V_x^α` (using **diagonal** potential only).

### Key Insight: Diagonal Structure in Eigenvalue Basis

After eigendecomposition, M becomes **diagonal** with elements:

```
M_diagonal[α, kx, ky] = E - d_x^α[kx] - d_y^α[ky]
```

Therefore:

```
M^{-1} = U * D^{-1} * U^{-1} * N^{-1}
```

where:
- `U = block_diag(U_x^α ⊗ U_y^α)` - block diagonal transformation
- `D^{-1}` - **diagonal** matrix with elements `1/(E - d_x^α[kx] - d_y^α[ky])`
- `N^{-1} = I_α ⊗ N_x^{-1} ⊗ N_y^{-1}` - overlap matrix inverse

## Implementation Approaches

### 1. M_inverse (Full Matrix)

**File**: `matrices.jl:604-678`

**Pros**:
- Returns explicit matrix (easy to use with existing code)
- Can compute `M_inv * LHS` and `M_inv * VRxy` directly

**Cons**:
- Stores full 16200×16200 matrix (~2.1 GB for 30×30 grid)
- Slower construction time

**Use for**:
- Testing and validation
- When you need `M_inv * (large matrix)`

**Example**:
```julia
M_inv = M_inverse(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
precond_LHS = M_inv * LHS
```

### 2. M_inverse_operator (Function)

**File**: `matrices.jl:526-602`

**Pros**:
- Returns lightweight function (minimal memory)
- Faster construction (~2-3× speedup)
- Efficient matrix-vector products

**Cons**:
- Cannot multiply with large matrices directly
- Must apply column-by-column for matrix operations

**Use for**:
- GMRES preconditioning (only needs `M_inv * vector`)
- Iterative methods
- Memory-constrained systems

**Example**:
```julia
M_inv_op = M_inverse_operator(α, grid, E0, Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny)
precond_vec = M_inv_op(rhs_vector)  # Apply to single vector
```

## Block-Diagonal Optimization

Both implementations exploit the block-diagonal structure:

**Before** (naive approach):
```julia
# Creates full 16200×16200 matrices (2.1 GB each!)
U_full = zeros(16200, 16200)
N_inv_full = kron(I(nα), kron(Nx_inv, Ny_inv))  # 16200×16200
M_inv = U_full * Diagonal(D_inv) * U_inv_full * N_inv_full
```

**After** (block-diagonal):
```julia
# Work with 18 channels × 900×900 blocks
for iα in 1:nα
    U_block = kron(Ux[iα], Uy[iα])  # Only 900×900
    N_inv_block = kron(Nx_inv, Ny_inv)  # Only 900×900
    M_inv_block = U_block * Diagonal(D_inv_block) * U_inv_block * N_inv_block
end
```

**Performance**:
- Memory: ~100× reduction (117 MB vs 2.1 GB)
- Speed: ~10-50× faster construction

## Integration with GMRES

### Wrong Approach (Column-by-Column)
```julia
# This takes forever! (29 iterations × 16200 columns = 470,000 GMRES solves!)
for col in 1:16200
    x_col = gmres(precond_LHS, precond_RHS[:, col])
    RHS[:, col] = x_col
end
```

### Correct Approach (On-the-Fly)
```julia
# Solve only when needed for Arnoldi iteration (~50 times total)
K = function(x)
    rhs = VRxy * x                  # Compute RHS for this vector
    precond_rhs = M_inv_op(rhs)     # Precondition (fast!)
    y = gmres(precond_LHS, precond_rhs; maxiter=100)
    return y
end

# Use K(x) in Arnoldi method (only ~50 calls to K)
λ, eigenvec = arnoldi_eigenvalue(K, v0, krylov_dim)
```

## Usage in MalflietTjon.jl

The Malfiet-Tjon solver has been updated to support GMRES with M^{-1} preconditioning:

```julia
# In malfiet_tjon_solve():
T, Tx_ch, Ty_ch, Nx, Ny = T_matrix(α, grid, return_components=true)
V, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)

# Pass components to compute_lambda_eigenvalue
λ, eigenvec = compute_lambda_eigenvalue(E, T, V, B, Rxy, α, grid,
                                       Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny;
                                       use_gmres=true)  # Enable GMRES
```

**Parameters**:
- `use_gmres=false` (default): Use direct solve `LHS \ VRxy`
- `use_gmres=true`: Use GMRES with M^{-1} preconditioning

## Performance Summary

### M^{-1} Construction (30×30 grid, 18 channels, 16200×16200 system)

| Method | Time | Memory | Notes |
|--------|------|--------|-------|
| Naive implementation | ~60s | 2.1 GB | Creates full dense matrices |
| Block-diagonal (M_inverse) | ~3-5s | 2.1 GB | Optimized but stores full result |
| Operator (M_inverse_operator) | ~1-2s | ~10 MB | Function-based, minimal storage |

### GMRES Performance

| Approach | Time | Notes |
|----------|------|-------|
| Direct solve `LHS \ VRxy` | 20s | Single factorization, full matrix |
| GMRES column-by-column | >1000s | Impractical! 16200 GMRES solves |
| GMRES on-the-fly with M^{-1}| ~30-50s | ~50 GMRES solves during Arnoldi |

## Testing

### Validate M_inverse Implementation
```bash
julia test_M_inverse.jl
```
Expected: Residual norm < 1e-12

### Check Condition Number Improvement
```bash
julia test_condition_number.jl
```
Expected: ~1000× reduction in condition number

### Compare Matrix vs Operator
```bash
julia test_M_operator_vs_matrix.jl
```
Expected: 2-3× speedup for operator construction

### GMRES Performance Test
```bash
julia test_gmres_vs_direct.jl
```
Note: Direct solve faster for small systems, GMRES advantages appear for larger systems

## Recommendations

1. **For production calculations**:
   - Use `use_gmres=false` (direct solve) for now
   - Direct solve is faster for current system sizes
   - GMRES becomes advantageous for very large systems (50×50 grids or more)

2. **For memory-constrained systems**:
   - Consider using `M_inverse_operator` in custom iterative solvers
   - Avoids storing large preconditioner matrices

3. **Future optimizations**:
   - Implement sparse matrix storage for M_inv (it's block-diagonal!)
   - Use Julia's LinearMaps.jl for matrix-free operators
   - Parallelize channel-by-channel operations

## Implementation Details

### Why Only Diagonal Potential?

The M matrix uses **only the diagonal part** of the potential (no off-diagonal channel coupling):

```julia
M = E*B - T - V_diagonal
```

The full potential (with off-diagonal coupling) appears in the LHS:

```julia
LHS = E*B - T - V_full
```

This approximation makes M^{-1} efficient to compute while still providing excellent preconditioning:
- Condition number: κ(LHS) ~ 60,000 → κ(M^{-1} * LHS) ~ 48
- ~1,239× improvement!

### Eigendecomposition Strategy

For each channel α:
1. Compute `H_x^α = T_x^α + V_x^α` (diagonal potential)
2. Solve `eigen(N_x^{-1} * H_x^α)` → eigenvalues `d_x^α`, eigenvectors `U_x^α`
3. Solve `eigen(N_y^{-1} * T_y^α)` → eigenvalues `d_y^α`, eigenvectors `U_y^α`
4. Diagonal matrix: `D_inv[kx, ky] = 1/(E - d_x^α[kx] - d_y^α[ky])`

**Cost**:
- Per channel: O(n_x^3) + O(n_y^3) eigenvalue problems
- Total: n_α * [O(n_x^3) + O(n_y^3)] ≈ 18 * [O(30^3) + O(30^3)]
- Much cheaper than O((n_α * n_x * n_y)^3) = O(16200^3) direct inversion!

## References

- Original implementation: `matrices.jl:604-678` (M_inverse)
- Optimized operator: `matrices.jl:526-602` (M_inverse_operator)
- GMRES integration: `MalflietTjon.jl:309-360` (compute_lambda_eigenvalue)
- Test scripts: `test_M_inverse.jl`, `test_condition_number.jl`, `test_gmres_vs_direct.jl`
