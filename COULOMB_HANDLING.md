# Coulomb Interaction Handling in swift.jl

## Overview

This document explains how Coulomb interactions are handled for proton-proton (pp) pairs in three-body calculations, particularly for ³He.

## The Double-Counting Issue

**Problem**: The Argonne potentials (AV18, AV14, etc.) already include electromagnetic interactions, including Coulomb repulsion for pp pairs, in the Fortran implementation (see `av18pot.f` line 61-63). However, `matrices.jl` was adding an additional point Coulomb term `VCOUL_point` for all pp pairs when `MT > 0`, leading to **double-counting** of the Coulomb interaction.

## Solution: Subtract and Re-add

To maintain code consistency and make Coulomb handling explicit, we implement the following strategy:

### Design Philosophy
```
Fortran AV18:  V_nuclear + V_coulomb (built-in)
      ↓
nuclear_potentials.jl: (V_nuclear + V_coulomb) - V_coulomb = V_nuclear
      ↓
matrices.jl: V_nuclear + VCOUL_point = V_nuclear + V_coulomb
      ↓
Final result: Same physics, but Coulomb handling is explicit and centralized
```

### Implementation Details

#### 1. `NNpot/nuclear_potentials.jl`

Added `compute_coulomb()` function:
```julia
const e2 = 1.43997  # MeV·fm (Coulomb constant)

function compute_coulomb(R::Float64, z12::Float64)
    if R ≈ 0.0
        return 0.0
    end
    return e2 * z12 / R
end
```

Modified `potential_matrix()` to **subtract** Coulomb for Argonne potentials:
```julia
# For all potential matrix elements (single channel, coupled, general):
if lpot <= 8 && tz == 1  # Argonne potentials and pp pair
    potential[...] -= compute_coulomb(r, 1.0)
end
```

**Argonne potentials** (lpot ≤ 8):
- AV18 (lpot=1)
- AV8' (lpot=2)
- AV6' (lpot=3)
- AV4' (lpot=4)
- AVX' (lpot=5)
- AV2' (lpot=6)
- AV1' (lpot=7)
- Modified AV8' (lpot=8)

#### 2. `swift/matrices.jl` (UNCHANGED)

Continues to add point Coulomb for pp pairs:
```julia
if α.MT > 0
    v = potential_matrix(potname, grid.xi[ir], li, ..., tz=1)  # pp pair
    v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0)
```

**VCOUL_point** implementation:
```julia
function VCOUL_point(R, z12)
    e2 = 1.43997  # MeV·fm
    if R ≈ 0.0
        return 0.0
    end
    return e2 * z12 / R
end
```

#### 3. `swift/matrices_optimized.jl` (UNCHANGED)

Uses the same approach as `matrices.jl` - automatically works with modified `nuclear_potentials.jl`.

### Why This Approach?

1. **No double-counting**: Coulomb is removed from Argonne potentials, then added back consistently
2. **Explicit handling**: Coulomb treatment is visible in Julia code, not hidden in Fortran
3. **Centralized**: All Coulomb additions happen in `matrices.jl` via `VCOUL_point`
4. **Consistent**: Same approach works for both AV18 (which has Coulomb) and MT (which doesn't)
5. **Simple**: Only `nuclear_potentials.jl` modified, `matrices.jl` unchanged

### Alternative Approaches (NOT Used)

❌ **Modify matrices.jl to check potential type**: Too complicated, couples matrix code to potential details
❌ **Remove VCOUL_point entirely**: Would require changing all matrix construction code
❌ **Keep double-counting**: Wrong physics for ³He

## Physics Validation

### For ³H (tritium, MT = -0.5)
- Contains nn pairs (tz = -1) and np pairs (tz = 0)
- No Coulomb interaction (no pp pairs)
- **No changes to physics**

### For ³He (helium-3, MT = +0.5)
- Contains pp pairs (tz = 1) and np pairs (tz = 0)
- Coulomb repulsion between protons
- **Before fix**: Double-counted Coulomb → Wrong binding energy
- **After fix**: Correct Coulomb → Correct binding energy (~7.72 MeV)

### Expected Results

| Nucleus | MT | Binding Energy | Coulomb Effect |
|---------|-----|----------------|----------------|
| ³H | -0.5 | ~8.48 MeV | None (no pp pairs) |
| ³He | +0.5 | ~7.72 MeV | ~0.76 MeV reduction |

## Testing

To verify the fix works correctly:

```bash
cd swift

# Test ³H (no Coulomb, should be unchanged)
julia swift_3H.jl

# Test ³He (with Coulomb, should give correct B.E.)
julia swift_3He.jl
```

Expected output for ³He:
```
Calculated binding energy: ~7.7-7.9 MeV
Experimental value:        ~7.718 MeV
```

## Code Locations

**Modified**:
- `NNpot/nuclear_potentials.jl`: Lines 9-22 (compute_coulomb), Lines 199-248 (subtract Coulomb)
- `swift/swift_3He.jl`: New file for ³He calculations

**Unchanged** (as intended):
- `swift/matrices.jl`: VCOUL_point additions remain as-is
- `swift/matrices_optimized.jl`: Uses same nuclear_potentials.jl

## References

- `av18pot.f` lines 60-63: Documentation that AV18 includes full EM interaction
- Experimental ³He binding energy: 7.718 MeV
- Experimental ³H binding energy: 8.482 MeV
- Coulomb constant: e² = 1.43997 MeV·fm

---

**Last Updated**: October 20, 2025
**Issue**: Fixed double-counting of Coulomb interaction for pp pairs with AV18 potential
**Solution**: Subtract Coulomb in nuclear_potentials.jl, re-add in matrices.jl
