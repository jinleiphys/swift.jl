# swift.jl TODO

Snapshot **2026-06-15 (end of day)**. This session: (1) cleaned the code to a single fastest
path (V-sector M⁻¹ only, GMRES-only scattering, ħ²/m_N=41.471 everywhere); (2) debugged the n-d
scattering phase-shift extraction and **localized the remaining bug to the S-matrix extraction**.

> ⚠️ **CLOUD-SYNC HAZARD**: this repo lives in a cloud-synced folder. A `scattering (conflicted copy …).jl`
> appeared mid-session and the sync silently reverted the `inner_product` line (dot ↔ transpose) at least
> once. Before trusting any run tomorrow, confirm `swift/scattering.jl:350` is `inner_product = dot(...)`
> (NOT `transpose(...)`). Consider moving the repo out of the synced folder, or commit early/often.

---

## ✅ Done this session

### Code cleanup → only the fastest path remains
- **matrices.jl** (653→425 lines): deleted `M_inverse_operator`, `M_inverse_operator_cached`,
  `precompute_M_inverse_cache`, `MInverseCache`. Only the **V-sector** trio + `group_channels_by_v_sector`.
- **MalflietTjon.jl**: collapsed to V-sector M⁻¹ + Arnoldi only. Removed: strict / non-cached M⁻¹ branches,
  the `use_vsector` and `use_arnoldi` toggles, the scratch RHS build, and the direct-eigen fallback
  (intractable at production size). `compute_lambda_eigenvalue_optimized` signature stripped of 9 vestigial
  positional args + `V_UIX`/`use_arnoldi` kwargs (carried by `M_cache`/`RHS_cache`).
- **scattering.jl**: `solve_scattering_equation` is **GMRES-only** with the V-sector M⁻¹ preconditioner
  (dense `:lu` removed — intractable at production size). `V_matrix_optimized_scaled` can return
  per-pair `V_x_full` so the CS preconditioner uses the same complex-scaled blocks as the operator.
- **Deleted** `swift/compare_solvers.jl` (broken LU-vs-GMRES diagnostic).
- **Convention**: ħ²/m_N = **41.471** MeV·fm² uniformly. 3-body kinetic was already 41.471; `twobody.jl`
  switched to equal nucleon masses so the deuteron is also exactly 41.471 (= rimas). The old
  benchmark_rimas comment claiming 41.468 was WRONG; the residual ~5 keV vs rimas is **nx=20 mesh
  truncation**, not the m_N convention.
- Commits: `c647f1c` (remove non-opt paths) → `71c6bf8` (V-sector-only + 41.471) → `7138d37`
  (scattering recoupling + CS amplitude fixes). Plus uncommitted scattering-debug WIP (see below).

### Bound-state verification (unchanged physics — cleanup is split-independent)
- swift_3H.jl AV18: **E(³H) = −7.191400 MeV** (was −7.191389 strict; Δ=1.1e-5, within λ tol).
- swift_3He.jl: −6.909; swift_3H_MN.jl: −8.302; all converge.
- **benchmark_rimas.jl full jx=1..6 scan reproduces rimas AV18 3H/3He to ~5 keV** (m_N-convention level),
  exact channel counts, conv=true everywhere. Cross-validation PASSED.

### Scattering debug — bug LOCALIZED (not yet fixed)
Target: Lazauskas-Carbonell PRC 84,034002 Tab.III doublet n-d, E_lab=14.1 MeV → **Re δ=105.49°, η=0.4649**.
Fixes applied + verified this session:
1. **wigner6j doubled-j** in `recouple_to_channel_spin`: WignerSymbols wants PHYSICAL j; code passed 2j →
   recoupling coeff²=0.444 instead of 1.0. Fixed (physical args + phase (-1)^(λ+s₃+J₁₂+J)). Identity test
   `T·I·T†=I` now holds for a single deuteron component.
2. **³S₁/³D₁ double-count**: deuteron S/D components were treated as independent channels (identity test
   gave 2.0). Fixed by keeping only deuteron components the bound state occupies (MT → pure ³S₁).
3. Operator energy → total 3-body **E_total = E_cm + E_d**; amplitude uses **complex-scaled V**;
   amplitude inner product = conjugated `dot` (matches bound-state ψ'·V·ψ convention, NO B metric).

**Current result**: Re δ ≈ 78.9°, **η ≈ 1.16** (full mesh) / η≈1.01 (reduced). δ in the ballpark; η wrong.

**Routing diagnostic (`swift/test_scatt_diag2.jl`, one reduced solve) → bug is in EXTRACTION (H1):**
- |ψ_in| GROWS ~exp(+q sinθ y) (correct for regular F_λ under CS) → A6 "rotated source" hypothesis REFUTED.
- source b is LOCALIZED (peaks y~2, decays) → correct, as Lazauskas requires.
- |ψ_sc| DECAYS ~exp(−q sinθ y) → CS working on the solution.
- **ψ_sc breakup-channel weight W_brk/(W_brk+W_d) = 0.21** → the CS solution DOES carry breakup flux.
- ⟹ generation/channels/solve are FINE; the elastic S-matrix EXTRACTION fails to convert the 21% breakup
  flux into η<1. (Channels ruled out earlier: 10→18 channels gave identical δ,η. Solve ruled out:
  relative residual 2.1e-5.)

**idea-pk panel** (dual-AI, groupthink-resolved) saved at
`idea-pk-output/20260615-161222/decision-panel.md`. It routes exactly to this H1 conclusion.

---

## ▶ NEXT (start here tomorrow) — fix the S-matrix extraction (H1)

The amplitude `f = -4μ/(ħ²k²)·⟨φ_d|V·Rxy_31|ψ_total⟩` is missing factors from the wiki's authoritative
benchmark formula (Lazauskas-Carbonell 2011, archived at
`~/research-wiki/sources/2011-prc-lazauskas-carbonell-cs-few-body-scattering.md`, Eqs. 16/17):

1. [ ] **Compute the deuteron CS c-norm** C_n = ∫ φ_d(x e^{−iθ}) φ_d(x e^{iθ}) e^{3iθ} dx and see how far
       it is from 1 (it is complex under CS). Current ψ_in uses the conjugate-normalized ⟨φ_d|B|φ_d⟩=1
       deuteron, but the amplitude should carry **C_n⁻¹**. ~1 h, no solve.
2. [ ] **Implement Eq.16 y-asymptotic projection** as an independent extraction (idea-pk candidate B3):
       `f_n = C_n⁻¹ · y · e^{−i q y e^{iθ}} · ∫ φ_d(x e^{−iθ}) ψ_sc(x,y) e^{3iθ} dx`, take the plateau over
       the last 10–15 y nodes. Plateau η≈0.46 ⟹ old V·Rxy integral was the wrong projection/normalization;
       plateau η≈1 ⟹ escalate to Green-theorem Eq.17 (with [V_j+V_k], e^{6iθ}). ~half day, reuses ψ_sc.
3. [ ] **2-body isolation check** (idea-pk candidate B1): build a 1-coordinate CS MT ¹S₀ two-body amplitude,
       θ=10°, r_max=100, verify δ=63.512° (Lazauskas Table I, exact to 5 digits). Isolates the scalar
       amplitude / S=1+2ikf convention from all 3-body machinery. ~1 day.
- After η is fixed: re-run benchmark_nd_scatt doublet/quartet at 14.1 & 42 MeV vs Tab.III, then θ-plateau
  + mesh (ymax→100) convergence to close the residual few-degree gap.

## Housekeeping
- [ ] Decide whether to keep `idea-pk-output/` and `test_scatt_diag*.jl` in the repo or gitignore them.
- [ ] Move repo out of the cloud-synced folder (see hazard banner) OR add the conflicted-copy glob to
      `.gitignore` and commit after every edit.

---

## Lessons (project memory at `~/.claude/projects/-Users-jinlei-Desktop-code-swift-jl/memory/`)
1. Arnoldi eltype trap — buffers via `eltype(v)`, not the cache type param.
2. swift.jl authorship framing — Jin's independent code following rimas's methodological line.
3. Lazauskas benchmark provenance — rimas personal comm., T=1/2, ħ²/m_N=41.471 (now matched exactly).
4. **NEW**: V-sector M⁻¹ preconditioner becomes near-singular ON-SHELL in scattering → GMRES converges in
   the M⁻¹ norm (1e-6) while the TRUE residual is ~2e-2 absolute but only **2.1e-5 relative** (‖b‖~1000).
   The relative residual is the one to trust; the solve is fine.
5. **NEW**: in CS scattering, the elastic inelasticity η<1 is NOT automatic from a breakup-carrying ψ_sc —
   it must be put in by the correct amplitude extraction (C_n c-norm + CS Jacobians, Lazauskas Eq.16/17).
   A wavefunction that contains breakup flux can still extract to η≈1 if the projection is wrong.

*End-of-day 2026-06-15. Resume at "▶ NEXT": fix the S-matrix extraction (compute C_n, implement Eq.16).*
