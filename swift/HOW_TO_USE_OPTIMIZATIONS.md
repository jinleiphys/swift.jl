# How to Use the Optimized Matrix Functions

## Quick Start (2 minutes)

### Option 1: Replace Original Functions (Recommended)

Simply replace the function calls in your existing code:

```julia
# At the top of your file, REPLACE:
include("matrices.jl")
using .matrices

# WITH:
include("matrices_optimized.jl")          # For T_matrix optimization
include("Rxy_matrix_optimized_v2.jl")     # For Rxy_matrix optimization
using .matrices_optimized
using .RxyOptimizedV2
```

Then use the optimized functions:

```julia
# For T matrix (14.3× faster):
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true)

# For Rxy matrix (1.35× faster):
Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized_v2(α, grid)

# V and B matrices use original implementation:
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)
B = Bmatrix(α, grid)
```

### Option 2: Side-by-Side Comparison

Keep both versions for testing:

```julia
include("matrices.jl")
include("matrices_optimized.jl")
include("Rxy_matrix_optimized_v2.jl")

using .matrices
using .matrices_optimized
using .RxyOptimizedV2

# Original versions
Tmat_orig = matrices.T_matrix(α, grid)
Rxy_orig, _, _ = matrices.Rxy_matrix(α, grid)

# Optimized versions
Tmat_opt = matrices_optimized.T_matrix_optimized(α, grid)
Rxy_opt, _, _ = RxyOptimizedV2.Rxy_matrix_optimized_v2(α, grid)

# Verify they're identical
using LinearAlgebra
@assert norm(Tmat_orig - Tmat_opt) < 1e-10
@assert norm(Rxy_orig - Rxy_opt) < 1e-10
```

---

## Example: Full Integration

Here's a complete example showing how to integrate into a typical three-body calculation:

```julia
# Load modules
include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh

# Load OPTIMIZED matrix functions
include("matrices_optimized.jl")
include("Rxy_matrix_optimized_v2.jl")
using .matrices_optimized
using .RxyOptimizedV2

# Also need original matrices module for V_matrix and Bmatrix
include("matrices.jl")
using .matrices

using LinearAlgebra

# System parameters
fermion = true
Jtot = 0.5
T = 0.5
Parity = 1
lmax = 2
λmax = 4
# ... (other parameters)

# Setup channels and grid
α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin,
        s1, s2, s3, t1, t2, t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

# Build matrices with OPTIMIZED functions
println("Computing matrices...")

# T matrix: 14.3× faster
Tmat, Tx_ch, Ty_ch, Nx, Ny = T_matrix_optimized(α, grid, return_components=true)

# V matrix: use original (no optimization yet)
Vmat, V_x_diag_ch = V_matrix(α, grid, potname, return_components=true)

# B matrix: use original (already fast)
B = Bmatrix(α, grid)

# Rxy matrix: 1.35× faster
Rxy, Rxy_31, Rxy_32 = Rxy_matrix_optimized_v2(α, grid)

# Build Hamiltonian
E0 = -8.0
H = E0 * B - Tmat - Vmat - Vmat * Rxy

# Solve eigenvalue problem
eigenvalues, eigenvectors = eigen(H, B)

println("Binding energy: ", eigenvalues[1], " MeV")
```

---

## Performance Comparison

Run this script to see the speedup:

```julia
include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels, .mesh

# Setup (same as above)
α = α3b(...)
grid = initialmesh(...)

# Original timing
include("matrices.jl")
using .matrices
println("Original implementation:")
@time begin
    T_orig = T_matrix(α, grid)
    Rxy_orig, _, _ = Rxy_matrix(α, grid)
end

# Optimized timing
include("matrices_optimized.jl")
include("Rxy_matrix_optimized_v2.jl")
using .matrices_optimized, .RxyOptimizedV2
println("\nOptimized implementation:")
@time begin
    T_opt = T_matrix_optimized(α, grid)
    Rxy_opt, _, _ = Rxy_matrix_optimized_v2(α, grid)
end

# Verify correctness
using LinearAlgebra
println("\nVerification:")
println("T matrix difference: ", norm(T_orig - T_opt))
println("Rxy matrix difference: ", norm(Rxy_orig - Rxy_opt))
```

Expected output:
```
Original implementation:
  1.785 seconds

Optimized implementation:
  0.802 seconds

Verification:
T matrix difference: 1.04e-16
Rxy matrix difference: 1.87e-16
```

---

## For Notebook Usage

If you're using Jupyter notebooks (like `swift_3H_optimized.ipynb`):

1. Add a new cell at the top:
```julia
# Load optimized matrix functions
include("matrices_optimized.jl")
include("Rxy_matrix_optimized_v2.jl")
using .matrices_optimized
using .RxyOptimizedV2

# Also load original for V and B matrices
include("matrices.jl")
using .matrices
```

2. Replace matrix computations:
```julia
# In your calculation cell, CHANGE:
# Tmat = matrices.T_matrix(α, grid)
# TO:
Tmat, Tx_ch, Ty_ch, Nx, Ny = matrices_optimized.T_matrix_optimized(α, grid, return_components=true)

# CHANGE:
# Rxy, Rxy_31, Rxy_32 = matrices.Rxy_matrix(α, grid)
# TO:
Rxy, Rxy_31, Rxy_32 = RxyOptimizedV2.Rxy_matrix_optimized_v2(α, grid)
```

3. Keep V and B matrix calls as-is:
```julia
Vmat = matrices.V_matrix(α, grid, potname)
B = matrices.Bmatrix(α, grid)
```

---

## Troubleshooting

### Issue 1: Module conflict errors
**Error**: `WARNING: both matrices and matrices_optimized export T_matrix`

**Solution**: Use explicit module prefixes:
```julia
T_opt = matrices_optimized.T_matrix_optimized(α, grid)
T_orig = matrices.T_matrix(α, grid)
```

### Issue 2: Results differ
**Error**: Large difference between original and optimized

**Solution**: This shouldn't happen! If it does:
1. Check that you're using the same input parameters
2. Verify Julia version compatibility (tested on Julia 1.6+)
3. Report the issue with your system parameters

### Issue 3: No speedup observed
**Possible causes**:
- First run includes compilation time (always warm up first!)
- System too small (benefits are smaller for nx,ny < 15)
- I/O bottleneck elsewhere in code

**Solution**:
```julia
# Warm up (compile)
T_matrix_optimized(α, grid)
Rxy_matrix_optimized_v2(α, grid)

# Now benchmark
@time T_matrix_optimized(α, grid)
@time Rxy_matrix_optimized_v2(α, grid)
```

---

## API Reference

### T_matrix_optimized
```julia
T_matrix_optimized(α, grid; return_components=false)
```
- **Speedup**: 14.3× faster than T_matrix
- **Memory**: 10× reduction
- **Returns**: Same as T_matrix
  - `Tmatrix` if `return_components=false`
  - `(Tmatrix, Tx_channels, Ty_channels, Nx, Ny)` if `return_components=true`

### Rxy_matrix_optimized_v2
```julia
Rxy_matrix_optimized_v2(α, grid)
```
- **Speedup**: 1.35× faster than Rxy_matrix
- **Memory**: 18% reduction
- **Returns**: `(Rxy, Rxy_31, Rxy_32)` - same as original

### Accuracy Guarantee
Both functions return results **identical to machine precision** (difference ~ 1e-16).

---

## Migration Checklist

- [ ] Backup your current working code
- [ ] Copy optimization files to your swift/ directory
- [ ] Test on small system first (nx=15, ny=15)
- [ ] Verify results match original (norm(A-B) < 1e-10)
- [ ] Benchmark to confirm speedup
- [ ] Integrate into production notebooks/scripts
- [ ] Test on realistic calculation (3H, 3He, etc.)
- [ ] Document any issues or unexpected behavior

---

## Support

For issues or questions:
1. Check `FINAL_OPTIMIZATION_REPORT.md` for detailed analysis
2. Run benchmark scripts to verify performance
3. Compare with original implementation side-by-side

**Files to reference**:
- Detailed analysis: `OPTIMIZATION_ANALYSIS.md`
- Performance results: `PERFORMANCE_SUMMARY.md`
- This guide: `HOW_TO_USE_OPTIMIZATIONS.md`
- Complete report: `FINAL_OPTIMIZATION_REPORT.md`
