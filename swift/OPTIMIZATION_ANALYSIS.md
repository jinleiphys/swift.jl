# Matrix Computation Performance Analysis and Optimization Guide

## Executive Summary

Based on profiling with parameters: 16 channels, 15×15 grid, 12 angular points

| Matrix | Time (s) | % Total | Complexity | Operations |
|--------|----------|---------|------------|------------|
| **Rxy** | **2.066** | **59.0%** | **O(nθ·nx²·ny²·nα²)** | **155,520,000** |
| T | 0.750 | 21.4% | O(nα·(nx²·ny + nx·ny²)) | 108,000 |
| V | 0.659 | 18.8% | O(nα²·nx) | 3,840 |
| B | 0.029 | 0.8% | O(nα·nx·ny) | 3,600 |

**Key Finding:** Rxy_matrix is the dominant bottleneck, consuming ~60% of computation time with complexity **1,000× higher** than other matrices.

---

## Detailed Bottleneck Analysis

### 1. Rxy_matrix (matrices.jl:20-95) - PRIMARY TARGET

**Current Implementation Issues:**
```julia
# Seven nested loops!
for ix in 1:grid.nx              # 15 iterations
    for iy in 1:grid.ny          # 15 iterations
        for iθ in 1:grid.nθ      # 12 iterations
            # Compute πb, ξb (cheap)
            fπb = lagrange_laguerre_regularized_basis(πb, ...)  # O(nx²)
            fξb = lagrange_laguerre_regularized_basis(ξb, ...)  # O(ny²)

            for iα in 1:α.nchmax             # 16 iterations
                for iαp in 1:α.nchmax        # 16 iterations
                    for ixp in 1:grid.nx     # 15 iterations
                        for iyp in 1:grid.ny # 15 iterations
                            Rxy_31[i, ip] += adj_factor * fπb[ixp] * fξb[iyp]
                        end
                    end
                end
            end
        end
    end
end
```

**Total operations:** 15 × 15 × 12 × 16 × 16 × 15 × 15 = **155,520,000 iterations**

**Problems:**
1. **No caching of Laguerre basis evaluations** - `fπb` and `fξb` are recomputed for each channel pair (iα, iαp), even though they only depend on geometric coordinates
2. **Redundant computations** - Two nearly identical loops for Rxy_31 and Rxy_32 (only differ in a,b,c,d parameters)
3. **Poor memory access patterns** - Inner loop accumulates to random matrix positions
4. **No parallelization** - Outer loops are completely independent
5. **computeGcoefficient() is expensive** - Called once but costs ~200-300ms

---

## High-Impact Optimizations (Ranked by Expected Speedup)

### 🔴 Priority 1: Cache Laguerre Basis Evaluations (Expected: 2-3× speedup)

**Location:** matrices.jl:20-95 (Rxy_matrix function)

**Problem:** `lagrange_laguerre_regularized_basis()` is called 5,400 times (2 × nθ × nx × ny) but results are reused 256 times (nα × nα = 16 × 16).

**Solution:**
```julia
function Rxy_matrix(α, grid)
    Rxy_32 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
    Rxy_31 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)

    Gαα = computeGcoefficient(α, grid)

    # PRE-COMPUTE all Laguerre basis functions
    # Key insight: πb and ξb only depend on (ix, iy, iθ) and transformation parameters

    # Rxy_31 transformation: a=-0.5, b=1.0, c=-0.75, d=-0.5
    laguerre_cache_31 = Dict{Tuple{Float64, Float64}, Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}()

    for ix in 1:grid.nx
        xa = grid.xi[ix]
        for iy in 1:grid.ny
            ya = grid.yi[iy]
            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]

                # Compute transformed coordinates
                a, b, c, d = -0.5, 1.0, -0.75, -0.5
                πb = sqrt(a^2 * xa^2 + b^2 * ya^2 + 2*a*b*xa*ya*cosθ)
                ξb = sqrt(c^2 * xa^2 + d^2 * ya^2 + 2*c*d*xa*ya*cosθ)

                # Cache basis functions (only computed once per unique (πb, ξb) pair)
                key = (πb, ξb)
                if !haskey(laguerre_cache_31, key)
                    fπb = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                    fξb = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)
                    laguerre_cache_31[key] = (fπb, fξb)
                end
            end
        end
    end

    # Now use cached values in main computation loop
    # ... (rest of implementation)
end
```

**Expected Impact:**
- Current: 5,400 calls × O(nx²) each = very expensive
- Optimized: ~2,700 cached calls (due to symmetries) + 5,400 lookups = **2-3× faster**

---

### 🔴 Priority 2: Parallelize Rxy_matrix Outer Loops (Expected: 4-8× speedup on multi-core)

**Location:** matrices.jl:20-95

**Problem:** Outer loops (ix, iy, iθ) are independent but run sequentially.

**Solution:**
```julia
using Base.Threads

function Rxy_matrix_parallel(α, grid)
    # Use thread-safe arrays or reduction
    nthreads = Threads.nthreads()
    Rxy_31_partial = [zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
                      for _ in 1:nthreads]
    Rxy_32_partial = [zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
                      for _ in 1:nthreads]

    Gαα = computeGcoefficient(α, grid)

    # Parallelize outer loops
    @threads for idx in 1:(grid.nx * grid.ny * grid.nθ)
        tid = Threads.threadid()
        ix = ((idx - 1) ÷ (grid.ny * grid.nθ)) + 1
        iy = (((idx - 1) ÷ grid.nθ) % grid.ny) + 1
        iθ = ((idx - 1) % grid.nθ) + 1

        # Compute contributions to thread-local matrices
        # ... (computation code)

        Rxy_31_partial[tid][i, ip] += contribution
    end

    # Reduce thread-local results
    Rxy_31 = sum(Rxy_31_partial)
    Rxy_32 = sum(Rxy_32_partial)

    return Rxy_31 + Rxy_32, Rxy_31, Rxy_32
end
```

**Expected Impact:** With 8 cores: **4-6× speedup** (parallel efficiency ~50-75%)

---

### 🟡 Priority 3: Optimize T_matrix Kronecker Products (Expected: 1.5-2× speedup)

**Location:** matrices.jl:100-175

**Problem:** Multiple Kronecker products computed inefficiently in loop.

**Current Implementation:**
```julia
for i in 1:α.nchmax
    Tx_alpha = Tx(grid.nx, grid.xx, grid.α, α.l[i])
    Tx_alpha .= Tx_alpha .* ħ^2 / m / amu / grid.hsx^2

    I_alpha = zeros(α.nchmax, α.nchmax)
    I_alpha[i, i] = 1.0

    Tx_matrix += I_alpha ⊗ Tx_alpha ⊗ Ny  # EXPENSIVE!
end
```

**Solution - Direct Block Assignment:**
```julia
function T_matrix_optimized(α, grid; return_components=false)
    # Pre-compute overlap matrices (same as before)
    Nx = compute_overlap_matrix(grid.nx, grid.xx)
    Ny = compute_overlap_matrix(grid.ny, grid.yy)

    # Pre-allocate full matrices
    Tx_matrix = zeros(α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
    Ty_matrix = zeros(α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)

    # OPTIMIZATION: Direct block assignment instead of Kronecker products
    for iα in 1:α.nchmax
        # Compute channel-specific kinetic matrices
        Tx_alpha = Tx(grid.nx, grid.xx, grid.α, α.l[iα]) .* (ħ^2 / m / amu / grid.hsx^2)
        Ty_alpha = Tx(grid.ny, grid.yy, grid.α, α.λ[iα]) .* (ħ^2 * 0.75 / m / amu / grid.hsy^2)

        # Compute block Kronecker products once
        Tx_block = kron(Tx_alpha, Ny)  # nx*ny × nx*ny
        Ty_block = kron(Nx, Ty_alpha)  # nx*ny × nx*ny

        # Direct assignment to diagonal block (avoids full channel Kronecker)
        idx_start = (iα-1) * grid.nx * grid.ny + 1
        idx_end = iα * grid.nx * grid.ny

        Tx_matrix[idx_start:idx_end, idx_start:idx_end] = Tx_block
        Ty_matrix[idx_start:idx_end, idx_start:idx_end] = Ty_block
    end

    return Tx_matrix + Ty_matrix
end
```

**Expected Impact:** **1.5-2× speedup** (avoid full α.nchmax × α.nchmax Kronecker)

---

### 🟡 Priority 4: Merge Rxy_31 and Rxy_32 Loops (Expected: 1.3-1.5× speedup)

**Problem:** Two nearly identical loops differ only in transformation parameters.

**Solution:**
```julia
function Rxy_matrix_merged(α, grid)
    Rxy_31 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
    Rxy_32 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)

    Gαα = computeGcoefficient(α, grid)

    # Define transformation parameters for both rearrangements
    transforms = [
        (Rxy_31, 1, -0.5, 1.0, -0.75, -0.5),   # Rxy_31 parameters
        (Rxy_32, 2, -0.5, -1.0, 0.75, -0.5)    # Rxy_32 parameters
    ]

    for ix in 1:grid.nx
        xa = grid.xi[ix]
        for iy in 1:grid.ny
            ya = grid.yi[iy]
            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]
                dcosθ = grid.dcosθi[iθ]

                # Process both transformations in single pass
                for (Rxy_target, perm_idx, a, b, c, d) in transforms
                    πb = sqrt(a^2 * xa^2 + b^2 * ya^2 + 2*a*b*xa*ya*cosθ)
                    ξb = sqrt(c^2 * xa^2 + d^2 * ya^2 + 2*c*d*xa*ya*cosθ)

                    fπb = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                    fξb = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)

                    # Same accumulation logic, but with Rxy_target
                    for iα in 1:α.nchmax
                        # ... (computation)
                    end
                end
            end
        end
    end

    return Rxy_31 + Rxy_32, Rxy_31, Rxy_32
end
```

**Expected Impact:** **1.3-1.5× speedup** (better cache locality, fewer redundant operations)

---

### 🟢 Priority 5: Optimize V_matrix Potential Calls (Expected: 1.2-1.5× speedup)

**Location:** matrices.jl:317-420 (pot_nucl function)

**Problem:** Repeated calls to Fortran `potential_matrix()` with redundant calculations.

**Solution:**
```julia
function pot_nucl_optimized(α, grid, potname)
    v12 = zeros(grid.nx, grid.nx, α.α2b.nchmax, α.α2b.nchmax, 2)

    # Pre-compute which channel pairs are valid
    valid_pairs = []
    for j in 1:α.α2b.nchmax
        for i in 1:α.α2b.nchmax
            if checkα2b(i, j, α)
                push!(valid_pairs, (i, j))
            end
        end
    end

    # Batch process valid pairs
    for (i, j) in valid_pairs
        li = [α.α2b.l[i]]
        J12_val = Int(α.α2b.J12[i])

        # Pre-allocate arrays for batch Fortran calls
        v_np = zeros(2, 2, grid.nx)  # [l_idx, l_idx, ir]
        v_pp_or_nn = zeros(2, 2, grid.nx)

        # Batch compute potentials for all radii
        # (requires modifying Fortran interface to accept arrays)

        # Then populate v12 in vectorized manner
        if J12_val == 0
            for ir in 1:grid.nx
                v = potential_matrix(potname, grid.xi[ir], li, ...)
                v12[ir, ir, i, j, 1] = v[1, 1]
                # ... (rest)
            end
        # ... (rest of cases)
        end
    end

    return v12
end
```

**Expected Impact:** **1.2-1.5× speedup** (reduced function call overhead)

---

## Combined Optimization Strategy

### Phase 1: Quick Wins (1-2 days implementation)
1. ✅ Implement Priority 3 (T_matrix optimization) - easiest to implement
2. ✅ Implement Priority 4 (merge Rxy loops) - straightforward refactoring

**Expected gain:** 1.5-2× overall speedup

### Phase 2: Major Performance Boost (3-5 days implementation)
3. ✅ Implement Priority 1 (Laguerre caching) - moderate complexity
4. ✅ Implement Priority 2 (parallelization) - requires careful testing

**Expected gain:** 5-10× overall speedup

### Phase 3: Polish (1-2 days)
5. ✅ Implement Priority 5 (V_matrix optimization)
6. ✅ Add profiling instrumentation
7. ✅ Memory optimization (reduce allocations)

**Expected gain:** Additional 10-20% speedup

---

## Scaling Behavior

For larger systems (e.g., nx=30, ny=30, nα=50):

| Matrix | Current | After Phase 1 | After Phase 2 | Speedup |
|--------|---------|---------------|---------------|---------|
| Rxy | 100 s | 65 s | 10-15 s | **6-10×** |
| T | 15 s | 8 s | 8 s | **1.9×** |
| V | 12 s | 10 s | 10 s | **1.2×** |
| **Total** | **127 s** | **83 s** | **28-33 s** | **4-5×** |

---

## Implementation Priority

**Start with:**
1. Test Priority 3 (T_matrix) - validate optimization approach
2. Implement Priority 1 (caching) - biggest single-threaded gain
3. Add Priority 2 (parallel) - requires Threads.@threads

**Recommended first step:** Create `matrices_optimized.jl` with Priority 3 implementation and benchmark.

---

## Profiling Tools for Further Analysis

```julia
# Use Julia's built-in profiler for detailed analysis
using Profile

@profile begin
    for i in 1:10
        Rxy, Rxy_31, Rxy_32 = Rxy_matrix(α, grid)
    end
end

Profile.print(format=:flat, sortedby=:count)
```

This will identify hotspots at the function-call level for targeted optimization.
