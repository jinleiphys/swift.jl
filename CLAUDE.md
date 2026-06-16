# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **⚠️ 2026-06-16 UPDATE — read [TODO.md](TODO.md) for the current state.** The code was cleaned to a
> single fastest path; several descriptions below are now stale. Key changes:
> - **M⁻¹ preconditioner is V-sector-only.** `matrices.jl` no longer has `M_inverse_operator`,
>   `M_inverse_operator_cached`, `precompute_M_inverse_cache`, `MInverseCache` — only the V-sector trio
>   (`MInverseCacheVSector`, `precompute_M_inverse_cache_vsector`, `M_inverse_operator_cached_vsector`)
>   + `group_channels_by_v_sector`. The `use_vsector` and `use_arnoldi` toggles are gone (always V-sector
>   + Arnoldi). `malfiet_tjon_solve_optimized` / `compute_lambda_eigenvalue_optimized` signatures slimmed.
> - **Scattering `solve_scattering_equation` is GMRES-only** with the V-sector M⁻¹ preconditioner; the
>   dense `:lu` method and the `method=` kwarg were REMOVED (LU is intractable at production size).
> - **ħ²/m_N = 41.471 MeV·fm² uniformly** (`twobody.jl` now uses equal nucleon masses; matches rimas).
> - **Deleted files**: `compare_solvers.jl`, `threebodybound.jl`, `Rxy_matrix_cached.jl`, `spline.jl`,
>   `3Npot/UIX.jl` (non-optimized UIX; `UIX_optimized.jl` is the only UIX now).
> - **New files**: `benchmark_rimas.jl` (AV18 3H/3He jx scan vs rimas), `benchmark_nd_scatt.jl` (n-d MT
>   vs Lazauskas PRC 84 Tab.III), `test_scatt_diag.jl` / `test_scatt_diag2.jl` (scattering diagnostics),
>   `test_cnorm_extraction.jl` (c-product / C_n extraction diagnostic, 2026-06-16).
> - **Open bug (2026-06-16, two layers)**: n-d elastic η was extracted ≈1 (unphysical) instead of the
>   benchmark η=0.4649. LAYER 1 (understood): the CS amplitude must use the **bilinear c-product**
>   (`transpose`, not Hermitian `dot`) + the **deuteron c-norm 1/C_n** (C_n = φ_dᵀBφ_d removes the
>   eigenvector's arbitrary global phase); this turns η≥1 into a physical η<1 (0.27) and is
>   gauge-invariant. LAYER 2 (localized, open): a residual (η 0.27 vs 0.46, δ 77° vs 105°) is
>   independent of mesh AND normalization → the `⟨φ_d F|V·Rxy_31|ψ⟩` projection OPERATOR differs from
>   Lazauskas Eq.16/17. `compute_scattering_amplitude` gained a `conj_bra` kwarg (default `true` = old
>   `dot`, since the bilinear path is not yet benchmark-complete). See TODO.md "▶ NEXT". The earlier
>   note "line 350 should be `dot` not `transpose`" was BACKWARDS.

## Project Overview

This is a Julia-based nuclear physics framework implementing the Faddeev method in coordinate (R-) space for three-body quantum mechanical bound-state and scattering calculations. Specializes in nuclear three-body problems (³H, ³He, n+d) on a Laguerre + Gauss-Legendre basis with multi-channel coupling, realistic NN potentials (AV18, AV14, Nijmegen, Minnesota), the Urbana IX three-nucleon force, and complex scaling for scattering above breakup threshold.

**Method family**: configuration-space Faddeev / Faddeev-Yakubovsky on a Laguerre + Gauss-Legendre basis with complex scaling. **swift.jl is independently developed by Jin Lei (Tongji)**, following the methodological line of Rimantas Lazauskas (IPHC Strasbourg) — the code is Jin's, the approach (R-space Faddeev + Lagrange-mesh + complex scaling) is the Lazauskas line of thinking. Lazauskas also acts as the external cross-validation reference for the numerical results (see project memory). For the field-level method survey see [FADDEEV_R_SPACE_NOTES.md](FADDEEV_R_SPACE_NOTES.md) (local formalism reference notes) and the Carbonell-Deltuva-Fonseca-Lazauskas 2013 review (arXiv:1310.6631, archived at `~/research-wiki/sources/2013-arxiv-carbonell-bound-state-techniques-multiparticle-scattering.md`).

**Capabilities as of 2026-05**:
- Bound states: direct generalized eigenvalue solver + Malfiet-Tjon iteration (with Arnoldi).
- Three-body forces: Urbana IX integrated into the Malfiet-Tjon path.
- Scattering: inhomogeneous CC equation solver (LU / GMRES) with complex scaling, COULCC Coulomb wavefunctions, partial-wave scattering amplitude matrix, Blatt-Biedenharn eigenphase + mixing-parameter analysis.

## Development Commands

### Quick Setup (Automated)
For first-time setup or to update the environment, run the automated setup script:
```bash
./setup.sh
```
This script will:
1. Check Julia installation and install/update if needed (requires Julia >= 1.9.0)
2. Install all required Julia packages via `setup.jl`
3. Compile Fortran nuclear potential libraries in `NNpot/`

### Manual Setup (Alternative)
If you prefer manual setup or need to rebuild specific components:

**Julia Environment Setup:**
```bash
julia setup.jl
```

**Building Fortran Libraries:**
```bash
cd NNpot
make clean && make
```
This creates `libpotentials.dylib` (macOS), `libpotentials.so` (Linux), or `libpotentials.dll` (Windows).

### Build System Details
- **Fortran compiler**: `gfortran` with `-O2 -fPIC -Wall -Wextra` optimization
- **Platform detection**: Automatic selection of shared library format
- **F77/F90 compatibility**: Separate compilation flags for legacy and modern Fortran code
- **COULCC library**: Compiled automatically by `setup.sh` via `Makefile_coulcc` in `swift/`
  - Creates `libcoulcc.dylib` (macOS), `libcoulcc.so` (Linux), or `libcoulcc.dll` (Windows)
  - Provides Coulomb wavefunctions for scattering calculations

### Running Calculations
- **³H (AV18, Malfiet-Tjon)**: `cd swift && julia swift_3H.jl` — the canonical tritium driver.
- **³H (Minnesota, exact NCM VMC benchmark)**: `cd swift && julia swift_3H_MN.jl` — MN u=1.0 (Serber) potential; target E(³H) = −7.889 ± 0.047 MeV (NCM VMC reference).
- **³He (with Coulomb)**: `cd swift && julia swift_3He.jl` — mirror of ³H with proton-proton Coulomb; MT = +0.5; explicit point-Coulomb handling in `matrices.jl`.
- **n+d scattering**: `cd swift && julia ndscatt.jl` — three-body scattering driver using the same Faddeev infrastructure.
- **Interactive**: `swift/swift_3H.ipynb` is the working notebook for exploration / method comparison.

### Testing
- **NNpot Julia interface**: `julia NNpot/test.jl` — basic AV18 / AV14 / Nijmegen interface validation.
- **Minnesota potential**: `julia NNpot/test_mn.jl` — pure-Julia MN potential cross-check.
- **Scattering amplitude pipeline**: `julia swift/test_scattering_amplitude.jl` — end-to-end n+d (`z1z2 = 0`) scattering at E = 1 MeV with MT potential and θ_deg = 8° complex scaling; computes deuteron bound state via `bound2b`, builds `ψ_in` via `compute_initial_state_vector`, solves [E·B − T − V·(I+Rxy)] ψ_sc = 2·V·Rxy_31·φ via LU, extracts partial-wave amplitude matrix and runs the Blatt-Biedenharn phase-shift analysis. **Tests use n+d (neutral); do not change to p+d (charged) without updating the Coulomb-wavefunction path.**
- **Solver comparison**: `julia swift/compare_solvers.jl` — LU vs preconditioned GMRES head-to-head on the same scattering system.

### Development Workflow
1. **Initial setup**: Run `./setup.sh` for automated environment setup (Julia installation, packages, Fortran libraries).
2. **Bound-state development**: edit, run `cd swift && julia swift_3H.jl` (or `swift_3H_MN.jl` for the MN benchmark).
3. **Scattering development**: edit, run `julia swift/test_scattering_amplitude.jl` (fast smoke test) → `julia swift/ndscatt.jl` (production-size n+d).
4. **Interactive exploration**: open `swift/swift_3H.ipynb` for method comparison / debugging.
5. **Deeper-dive notes**: [OPTIMIZATION_ANALYSIS.md](OPTIMIZATION_ANALYSIS.md) (performance + memory analysis), [FADDEEV_R_SPACE_NOTES.md](FADDEEV_R_SPACE_NOTES.md) (formalism reference), [slides_presentation.md](slides_presentation.md) (talk-ready slides).

## Core Architecture

### Module Structure
The codebase is organized into four module directories:

1. **NNpot/**: Nuclear potential interface
   - Fortran libraries (AV18, AV14, Nijmegen) with Julia wrappers via `Libdl`
   - `nuclear_potentials.jl`: dispatch by `potname` ∈ {"AV18", "AV14", "Nijmegen", "MN"}; Minnesota potential is pure-Julia (no Fortran).
   - `makefile`: compiles `libpotentials.dylib` / `.so` / `.dll`.
   - `test.jl`, `test_mn.jl`: interface smoke tests.

2. **general_modules/**: Foundation components
   - `channels.jl`: three-body channel coupling with angular-momentum algebra; entry point `α3b(...)`.
   - `mesh.jl`: Laguerre + Gauss-Legendre mesh generation; entry point `initialmesh(...)`.

3. **swift/**: Core Faddeev implementation
   - `matrices.jl`: T / V / Rxy matrix elements; explicit point-Coulomb branch for MT > 0 systems (e.g., ³He).
   - `matrices_optimized.jl`: caching + complex-scaling-aware variants (`V_matrix_optimized`, `V_matrix_optimized_scaled`, `T_matrix_optimized`, `Rxy_matrix_optimized`).
   - `Rxy_matrix_cached.jl`: per-channel cached Rxy with custom G coefficients (includes `Rxy_13`).
   - `Gcoefficient.jl`: angular-momentum coupling coefficients (Wigner symbols).
   - `laguerre.jl`: Laguerre basis function implementations.
   - `spline.jl`: spline utilities.
   - `twobody.jl`: two-body reference solver (`bound2b` → deuteron with ³S₁ + ³D₁).
   - `threebodybound.jl`: direct generalized-eigenvalue bound-state solver (`ThreeBody_Bound`).
   - `MalflietTjon.jl`: iterative bound-state solver (`malfiet_tjon_solve`, `malfiet_tjon_solve_optimized`) with Arnoldi eigenvalue routine (`arnoldi_eigenvalue`), RHS-cache precomputation (`RHSCache`, `precompute_RHS_cache`), channel-probability analysis (`compute_channel_probabilities`), and integrated UIX evaluation (`compute_uix_potential`, `compute_uix_potential_optimized`).
   - `scattering.jl` (Module `Scattering`): inhomogeneous CC solver `solve_scattering_equation` (LU / GMRES with M⁻¹ preconditioner), `compute_initial_state_vector` (deuteron × Coulomb F_λ source), `compute_scattering_amplitude` (partial-wave amplitude matrix), `compute_collision_matrix` (U = 2ik·f + I), `compute_eigenphase_shifts`, `compute_phase_shift_analysis` (Blatt-Biedenharn eigenphase + mixing ε, ζ, η).
   - `coulcc.jl` + `coulcc.f` + `Makefile_coulcc`: Julia wrapper for the COULCC Fortran library (Coulomb wavefunctions F_λ, G_λ).
   - `ndscatt.jl`: production-size n+d scattering driver.
   - `compare_solvers.jl`: LU vs GMRES diagnostics on the scattering equation.
   - Drivers: `swift_3H.jl` (AV18), `swift_3H_MN.jl` (Minnesota), `swift_3He.jl` (with Coulomb).
   - Tests: `test_scattering_amplitude.jl` (n+d MT smoke test with θ_deg = 8°).
   - Notebook: `swift_3H.ipynb`.

4. **3Npot/**: Three-body nuclear force models
   - `UIX.jl`: Urbana IX three-body force with Y(r) and T(r) radial functions.
   - `UIX_optimized.jl`: cached / vectorised UIX matrix element evaluation.

### Key Physics Concepts
- **Faddeev equations in R-space**: three-body quantum mechanics on Jacobi coordinates (x, y) with coordinate-transformation matrices Rxy_31, Rxy_32.
- **Channel coupling**: |(l₁₂(s₁s₂)s₁₂)J₁₂, (λ₃s₃)J₃, J; (t₁t₂)T₁₂, t₃, T MT⟩ enumerated by `α3b(...)`.
- **NN potentials**: realistic AV18 / AV14 / Nijmegen (Fortran) + Minnesota (Julia); switched by `potname`.
- **Three-body forces**: Urbana IX (UIX) with Y(r) and T(r); evaluated in the Lagrange-function basis inside the `MalflietTjon` module.
- **Complex scaling**: r → r·e^{iθ} for scattering above breakup threshold and resonance calculations; supported via `*_scaled` variants of the matrix builders.
- **Scattering amplitude → phase shifts**: partial-wave f-matrix → collision matrix U → Blatt-Biedenharn eigenphase shifts + mixing parameters (ε, ζ, η) on the (λ, 𝕊) channel-spin basis.

### Computational Workflow

**Bound State Calculations:**
1. **Channel generation**: `α3b()` creates allowed quantum states based on conservation laws
2. **Mesh initialization**: `initialmesh()` sets up hyperspherical coordinate grids
3. **Matrix construction**: Build Hamiltonian H = T + V + V*Rxy using Faddeev rearrangement
   - Optional: Include three-body forces H₃ = T + V + V*Rxy + X₁₂ (UIX model)
4. **Eigenvalue solution**: Two approaches available:
   - **Direct method**: `ThreeBody_Bound()` solves generalized eigenvalue problem `eigen(H, B)`
   - **Iterative method**: `malfiet_tjon_solve()` uses secant iteration to find λ(E) = 1

**Scattering Calculations:**
1. **Initial state**: `compute_initial_state_vector(grid, α, φ_d_matrix, E, z1z2; θ=0.0)` builds ψ_in = φ_d(x) · F_λ(ky) / (φx φy) populating only deuteron channels (J₁₂ = 1, ³S₁ + ³D₁). Uses COULCC for F_λ. Complex scaling via θ (radians).
2. **Matrix assembly**: `compute_scattering_matrix(E, α, grid, potname; θ_deg=0.0)` returns A = E·B − T − V·(I + Rxy) plus component matrices (Tx_ch, Ty_ch, V_x_diag_ch, Nx, Ny) used to build the M⁻¹ preconditioner.
3. **Source term**: 2·V·Rxy_31·φ — factor of 2 from Faddeev symmetry. Multiplication order V·(Rxy_31·φ) for efficiency.
4. **Linear solve**: `solve_scattering_equation(E, α, grid, potname, ψ_in; θ_deg=0.0, method=:lu)` solves [A] ψ_sc = b.
   - `method=:lu`: dense LU, for smoke tests and benchmarking.
   - `method=:gmres`: preconditioned GMRES with M⁻¹ = [E·B − T − V_αα]⁻¹ (within-channel V only) — same preconditioner used by Malfiet-Tjon; required for larger meshes.
   - Both branches accept complex scaling via θ_deg.
5. **Amplitude + phase shifts**: `compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix, z1z2; θ=0.0, σ_l=0.0)` returns the partial-wave amplitude matrix f and the deuteron-channel labels. Pipe into `compute_phase_shift_analysis(f, k, α, deuteron_channels, channel_labels)` to get eigenphase shifts and Blatt-Biedenharn mixing parameters on the channel-spin basis.

### Data Flow
- **Bound states**: System parameters → Channel coupling → Matrix elements → Eigenvalue problem → Binding energies and wavefunctions
- **Scattering**: System parameters + Initial state → Matrix assembly → Inhomogeneous equation → Scattering wavefunction and observables

## Important Implementation Details

### Fortran Integration
- Nuclear potentials implemented in Fortran for performance (AV18, AV14, Nijmegen models)
- Julia-Fortran interface via `ccall` and `Libdl.dlopen()` for dynamic library loading
- Automatic symbol resolution with name-mangling fallback patterns (`find_symbol()` function)
- Platform-specific dynamic library handling (`.dylib` on macOS, `.so` on Linux, `.dll` on Windows)

### Numerical Methods
- **Basis functions**: Laguerre polynomials for radial coordinates with regularization parameter
- **Integration**: Gauss-Legendre quadrature for angular components
- **Eigenvalue methods**: Two complementary approaches:
  - **Direct diagonalization**: Generalized eigenvalue problem `eigen(H, B)` finds all states
  - **Malfiet-Tjon iteration**: Reformulates as `λ(E)[c] = [E*B - T - V]⁻¹ * V*R * [c]`
- **Scattering solver**: Inhomogeneous equation solver with LU factorization or GMRES iterative method
- **Convergence**: Secant method iteration until `|λ(E) - 1| < tolerance`

### Faddeev Normalization (Critical for Truncated Model Space)
The framework uses **Faddeev normalization** which is essential for truncated model spaces:

**Two possible normalization schemes:**
1. **⟨Ψ|Ψ⟩ = 1**: Direct normalization of full wavefunction Ψ = (1 + Rxy)ψ₃
   - ❌ INCORRECT for truncated spaces
   - Rxy maps between Jacobi coordinates and is incomplete with finite lmax/λmax
   - Results in unconverged normalization

2. **3⟨Ψ|ψ₃⟩ = 1**: Faddeev normalization scheme
   - ✅ CORRECT for truncated spaces
   - Only involves ψ₃ which is fully defined within the truncated space
   - Converged even with finite basis size
   - Implementation: `|Ψ̄⟩ = |Ψ⟩/√(3⟨Ψ|ψ₃⟩)` ensures ⟨Ψ̄|Ψ̄⟩ = 1

**Channel Probability Calculation:**
- Computed from the full wavefunction Ψ̄: `P_channel = ⟨Ψ̄_ch|B|Ψ̄_ch⟩`
- Probabilities sum to ~98-99% (1-2% missing due to truncation)
- As lmax, λmax → ∞, sum approaches 100%
- All probabilities are positive (cross-terms can be negative, avoid using them)

### Channel Indexing
The framework uses sophisticated indexing schemes:
- Three-body channels: `|(l₁₂(s₁s₂)s₁₂)J₁₂, (λ₃s₃)J₃, J; (t₁t₂)T₁₂, t₃, T MT⟩`
- Matrix elements indexed by channel and coordinate grid points
- Pauli principle and parity constraints automatically enforced

## Working with the Code

### Adding New Potentials
1. Add Fortran implementation to `NNpot/` directory
2. Update makefile to include new source files
3. Add Julia wrapper function in `nuclear_potentials.jl`
4. Update `potential_type_to_lpot()` mapping

### Working with Three-Body Forces
- **UIX model**: Use `UIX.X12_matrix(α, grid)` to compute three-body force matrix
- **Matrix indexing**: Same as V and T matrices: `i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy`
- **Angular momentum basis**: UIX functions implemented in Lagrange function basis, not Jacobi coordinate basis
- **Physical constants**: Uses PDG pion mass values with proper averaging formula
- **Delta functions**: Matrix elements include channel selection rules and coordinate constraints
- **Isospin phase convention**: The isospin phase factor is `(-1)^(T12_prime + 2*t1 + t2 + t3)` where T12_prime is from the **ket** (incoming channel), not the bra (outgoing channel). This must match the phase convention in `Gcoefficient.jl` line 91 for consistent angular momentum recoupling.

### Scattering Calculations
The framework supports n+d (neutral) scattering with full machinery for Coulomb (p+d) and complex scaling. The reference end-to-end pipeline is `swift/test_scattering_amplitude.jl` (smoke test) and `swift/ndscatt.jl` (production).

**Reference pipeline** (matches `test_scattering_amplitude.jl`):
```julia
# 1. Channels + mesh
α    = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin, s1,s2,s3, t1,t2,t3, MT, j2bmax)
grid = initialmesh(nθ, nx, ny, Float64(xmax), Float64(ymax), Float64(alpha))

# 2. Deuteron bound state (³S₁ + ³D₁)
bound_E, bound_ψ = bound2b(grid, "MT", θ_deg=θ_deg)
φ_d_matrix       = ComplexF64.(bound_ψ[1])

# 3. Three-body matrices
V              = V_matrix_optimized(α, grid, "MT")
Rxy, Rxy_31    = Rxy_matrix_optimized(α, grid)

# 4. Initial state ψ_in = φ_d(x) F_λ(ky)
θ_rad = θ_deg * π / 180.0
ψ_in  = compute_initial_state_vector(grid, α, φ_d_matrix, E, z1z2, θ=θ_rad)

# 5. Solve [E·B − T − V·(I+Rxy)] ψ_sc = 2·V·Rxy_31·φ
ψ_sc, A, b = solve_scattering_equation(E, α, grid, "MT", ψ_in, θ_deg=θ_deg, method=:lu)

# 6. Partial-wave amplitude matrix
f, deuteron_channels, channel_labels = compute_scattering_amplitude(
    ψ_in, V, Rxy_31, ψ_sc, E, grid, α, φ_d_matrix, z1z2, θ=θ_rad, σ_l=0.0)

# 7. Blatt-Biedenharn phase-shift analysis
k             = sqrt(2.0 * (2m/3) * 931.49432 * E) / 197.3269718
phase_results = compute_phase_shift_analysis(f, k, α, deuteron_channels, channel_labels)
```

**Key physics**:
- Deuteron (J₁₂ = 1) contains ³S₁ (~96%) + ³D₁ (~4%) components — both populated in ψ_in.
- Initial state ψ_in lives only in deuteron-coupling channels; J₁₂ ≠ 1 channels are zero.
- Each channel uses its own λ for F_λ(ky); COULCC returns all λ in a single call via `coulcc(x, η, lmin; lmax=λmax, mode=4)`.
- For n+d use `z1z2 = 0.0`; for p+d (or any charged channel) use `z1z2 = 1.0` and the COULCC path activates the Coulomb wavefunctions.

**Phase-shift output structure** (`compute_phase_shift_analysis` return):
- `Dict{(J, π) → Dict("eigenphases", "mixing_params", "U_matrix", "u_matrix", "labels")}`.
- `mixing_params` is a named tuple `(ε, ζ, η)` for 3×3 (J, π) blocks, `(ε, ζ=0, η)` or `(ε=0, ζ=0, η)` for 2×2 blocks.

**COULCC library**:
- Provides regular F_λ and irregular G_λ Coulomb wavefunctions.
- Built from `coulcc.f` by `Makefile_coulcc` into `libcoulcc.dylib` / `.so` / `.dll`.
- Julia wrapper in `coulcc.jl` accepts complex arguments → complex-scaling-ready.

**Complex scaling**:
- Coordinates rotate r → r·e^{iθ}; matrices support this via the `*_scaled` variants:
  - `V_matrix_optimized_scaled(α, grid, potname; θ_deg=10.0)`
  - `T_matrix_optimized(α, grid; θ_deg=10.0)`
- θ = 0 paths are zero-overhead (they short-circuit back to the real builders).
- Validity bound: tan θ < η/k (η = inverse range of the short-range potential).

### Modifying Calculations
- Channel quantum numbers: edit the header of the relevant driver (`swift_3H.jl`, `swift_3H_MN.jl`, `swift_3He.jl`, `ndscatt.jl`, `test_scattering_amplitude.jl`).
- Mesh parameters: adjust `nx`, `ny`, `xmax`, `ymax`, `nθ`, `alpha` for convergence.
  - **For j2bmax = 2.0, bound state**: `nθ=12, nx=20, ny=20, xmax=16, ymax=16` converges within 0.2 keV.
  - For tighter convergence: `nx=25, ny=25` reaches ~1 keV.
  - Scattering convergence (test_scattering_amplitude default) uses `nθ=12, nx=30, ny=70, xmax=30, ymax=60` for n+d at 1 MeV.
- Potential: change `potname` ∈ {"AV18", "AV14", "Nijmegen", "MN", "MT"} (MT is the Malfliet-Tjon toy potential used in scattering tests).
- Three-body forces: pass `include_uix=true` to `malfiet_tjon_solve_optimized(...)`; the module handles `compute_uix_potential_optimized` internally.

### Solver Selection and Performance
- **Direct method**: Use `ThreeBody_Bound()` when you need all eigenvalues or for initial exploration
- **Malfiet-Tjon method**: Use `malfiet_tjon_solve()` for ground state targeting, can be faster than direct method
- **Arnoldi optimization**: Recent performance improvements use optimized Arnoldi eigenvalue solver with adaptive convergence
- **Memory optimization**: Use smaller mesh sizes (20×20 instead of 30×30) and reduced λmax/lmax for memory-constrained systems
- **Performance consideration**: Malfiet-Tjon avoids expensive generalized eigenvalue problems
- **Initial guesses**: Use direct method results to inform Malfiet-Tjon starting energies for optimal convergence
- **Module conflicts**: Import MalflietTjon functions explicitly: `import .MalflietTjon: malfiet_tjon_solve`

### Debugging and Troubleshooting
- **Library loading**: Use `list_symbols(libpot)` to inspect available Fortran symbols
- **Symbol resolution**: `find_symbol()` handles platform-specific name mangling
- **Channel validation**: Check `α3b()` output for quantum number consistency and conservation laws
- **Matrix conditioning**: Monitor eigenvalue convergence and matrix conditioning in both solvers
- **Malfiet-Tjon convergence**: Use `verbose=true` to track λ eigenvalue behavior during iteration
- **Energy guesses**: For Malfiet-Tjon, start within ±1 MeV of expected ground state energy
- **Performance analysis**: Built-in timing analysis in main calculation routines
- **Notebook debugging**: Use `.ipynb` files for interactive problem investigation and method comparison
- **Symmetry checks**: Built-in rearrangement matrix transpose relationship validation (`Rxy_32 = Rxy_31^T`) with tolerance checking
- **Wave function analysis**: Channel probability contributions and normalization verification
  - Check that ⟨Ψ̄|ψ̄₃⟩ = 1/3 (Faddeev normalization)
  - Channel probabilities should sum to 98-99% (1-2% missing due to truncation is normal)
  - Negative channel probabilities indicate incorrect normalization scheme
- **Energy consistency**: Automated checks that ⟨ψ|H|ψ⟩ matches eigenvalue within tolerance
- **Truncation effects**: Monitor total probability sum - as lmax/λmax increase, sum approaches 100%

### Required Julia Packages
- **SphericalHarmonics**: spherical harmonic evaluation.
- **WignerSymbols**: angular-momentum coupling coefficients.
- **FastGaussQuadrature**: Gauss-Legendre quadrature for the angular mesh.
- **Kronecker**: tensor-product operations.
- **IterativeSolvers**: GMRES path for `solve_scattering_equation` and for Arnoldi-based eigensolves in MalflietTjon.
- **LinearAlgebra** + **SparseArrays**: standard library; BLAS thread count set to `Sys.CPU_THREADS` in the drivers.
- **Printf**, **JSON**: I/O.
- **Revise** (optional): hot reloading during development.

### Platform-Specific Notes
- **Dynamic libraries**: Build system automatically detects platform and uses appropriate extensions
- **macOS**: Creates `.dylib` files with `-dynamiclib -single_module -undefined dynamic_lookup`
- **Linux**: Creates `.so` files with `-shared` flag
- **Windows**: Creates `.dll` files with `-shared -Wl,--export-all-symbols`
- **Cross-platform compatibility**: Consistent interface across all platforms via Julia's `Libdl` module

### Error Handling and Common Issues
- **Library not found**: Ensure `make` completed successfully in `NNpot/` directory
- **Symbol resolution failures**: Check Fortran compilation and library loading with `list_symbols()`
- **Julia package issues**: Re-run `julia setup.jl` if missing dependencies
- **Module import conflicts**: Use explicit imports `import .MalflietTjon: function_name` instead of `using`
- **Malfiet-Tjon divergence**: Method fails with poor initial guesses - use direct method results as starting point
- **Numerical instabilities**: Adjust mesh parameters and check channel coupling convergence
- **Insufficient basis**: Small mesh sizes (< 20×20) may not capture three-body bound states

### Method Comparison Guidelines
- **Cross-validation**: Always compare direct and Malfiet-Tjon results for the same system
- **Energy convergence**: Both methods should agree to at least 6 decimal places for ground state
- **Computational trade-offs**: Malfiet-Tjon faster for single state, direct method gives complete spectrum
- **Physical insight**: Malfiet-Tjon iteration demonstrates bound state formation mechanism via λ → 1

### Advanced Features and Optimizations
- **Arnoldi eigenvalue solver** (`arnoldi_eigenvalue` in MalflietTjon.jl): adaptive convergence with early termination, used by `malfiet_tjon_solve_optimized`.
- **RHSCache** (`precompute_RHS_cache` in MalflietTjon.jl): caches the V·Rxy product structure across Malfiet-Tjon iterations.
- **Caching of Wigner coefficients**: angular-momentum coupling cached across V, T, Rxy, UIX builds.
- **Cross-platform symbol resolution**: `find_symbol()` in `nuclear_potentials.jl` handles gfortran name-mangling on macOS / Linux / Windows.
- **Validation framework**: `Rxy_32 = Rxy_31ᵀ` transpose check; energy consistency ⟨ψ|H|ψ⟩ vs eigenvalue; channel-probability sum diagnostic.
- **Integrated timing**: `@elapsed` blocks in the drivers report per-stage cost; `compare_solvers.jl` is the dedicated LU vs GMRES benchmark.

## Methodological references

- **[FADDEEV_R_SPACE_NOTES.md](FADDEEV_R_SPACE_NOTES.md)** (local file in this repo): formalism reference notes covering the (x, y) Jacobi coordinates, channel-coupling algebra, Rxy rearrangement matrices, Faddeev normalization, and the bound-state eigenvalue problem (Sec. "Eigenvalue problem of the Faddeev Equations"). **Primary reference when modifying matrix-element code.**
- **[OPTIMIZATION_ANALYSIS.md](OPTIMIZATION_ANALYSIS.md)** (local): performance + memory analysis notes for the Malfiet-Tjon / Arnoldi path.
- **[slides_presentation.md](slides_presentation.md)** (local): talk-ready summary slides of the framework.
- **Carbonell-Deltuva-Fonseca-Lazauskas 2013, arXiv:1310.6631** (review, 36 pp, 159 refs): comprehensive survey of bound-state-basis techniques for multiparticle scattering. Archived in the literature wiki at `~/research-wiki/sources/2013-arxiv-carbonell-bound-state-techniques-multiparticle-scattering.md`. **The Lazauskas-Carbonell configuration-space complex-scaling Faddeev-Yakubovsky line (refs [41, 42, 44, 50, 51] within that paper) is the methodological line Jin followed when independently designing swift.jl.** Read it when extending swift.jl to 4N or to scattering above breakup.