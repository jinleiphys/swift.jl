# swift.jl TODO

Snapshot **2026-06-18**. ✅ **BREAKTHROUGH: δ now hits ~103-104° (benchmark 105.49°) after Rimas replied.**
Rimas's answer: I was using too large a CS angle (θ=10°). swift uses PHYSICAL (non-mass-scaled) Jacobi
coords so θ_max is smaller than my √3 estimate; dropping to **θ=3-4°** makes the scattered term
mesh-stable (his Table 5.1: θ=3° → δ=105.0/η=0.456). The amplitude is his HDR **Eq.2.118** (arXiv:1904.04675):
`f = −(1/Ecm)[ e^{2iθ}⟨Ω_in|V·Rxy|ψ_sc⟩_CS + ⟨Ω_in|V·Rxy|Ω_in⟩_{no CS, Born} ]`, bra = REGULAR incoming
(not the outgoing exp I wrongly switched to), Born term on the real axis, CS Jacobian = e^{2iθ} (x-contour
cancels V's e^{−iθ}, y-contour supplies the factor missing from Ny). δ within ~2° now; **remaining: η is
~30% low (0.30-0.34 vs 0.465) and drifts with ymax** → channel-spin recoupling (only first λ group, no
Blatt-Biedenharn weights) and/or y-mesh resolution. NOTE: the earlier "Lagrange-Laguerre overlap tail"
root-cause was a misread; the real issue was the CS angle (back-rotation + Lagrange-Laguerre is correct,
confirmed by Rimas + COLOSS). Full record in memory `kmatrix-integral-relations-method.md`.

> ⚠️ **CLOUD-SYNC HAZARD**: this repo lives in a cloud-synced folder; a `scattering (conflicted copy …).jl`
> appeared mid-session before and the sync silently reverted an `inner_product` line at least once. Commit early/often.

---

## ⏸ BLOCKED 2026-06-17 — root cause found, waiting for Rimas

**Target unchanged**: Lazauskas-Carbonell PRC 84,034002 Tab.III doublet n-d, E_lab=14.1 MeV →
**Re δ=105.49°, η=0.4649** (K[1,1]=−1.143+1.870i). Extraction back-end = Glöckle Physics Reports 274
(1996) Eq.209-214 (channel-spin Blatt-Biedenharn S=U†ΛU), already in `compute_phase_shift_analysis`.

**What this session settled (all in memory `kmatrix-integral-relations-method.md`):**
- ✅ **Rimas's actual method = his 2011 Eq.17 (Green theorem), NOT the Duerinck-thesis K-matrix.** The
  whole K-matrix A-matrix linear-system detour was the overcomplication. Eq.17:
  `f = −C_n⁻¹(2µ_y/ħ²)e^{6iθ}∫∫ φ_n*(xe^{−iθ})[e^{−iqy e^{iθ}}/y][V_j+V_k]ψ_m d³x d³y`, then S=1+2iq·f.
- ✅ **Formula/coefficient OK**: coordinate-space S=1+2iq·f (Rimas Eq.6) vs momentum-space Glöckle U→S
  differ ONLY in the amplitude→S step; Eq.209-214 is convention-free on S. swift's `compute_collision_matrix`
  already uses U=1+2ik·f (coordinate-space). Don't mix in momentum-space µq/(2π)³ factors. Coefficient = 31.10·q
  (=ℏ²q/2µ_y, swift physical Jacobi coords; the earlier "ℏ²q/m0=4/3" was wrong — Codex-confirmed).
- ✅ **Coordinate rotation OK**: verified V uses back-rotation (basis at re^{−iθ}, V at real r, jacobian e^{−iθ})
  = exactly V(re^{+iθ}), same r→re^{+iθ} as the kinetic's e^{−2iθ}. Outgoing decays, no sign error.
- ✅ **off-diagonal=0 is physical** (MT central force, no deuteron D-state → doublet decouples from λ=2 at
  the V vertex). Doublet is a scalar K[1,1].
- ❌ **THE BUG (root cause, decisive per-y diagnostics)**: `compute_overlap_matrix` (matrices_optimized.jl:49)
  builds the non-orthonormal Lagrange-Laguerre overlap `N[i,j]=δ + (-1)^{i-j}/√(yᵢyⱼ)`. The 1/√(yᵢyⱼ)
  off-diagonal is LONG-RANGE. Since `V_block=kron(V_x,Ny)`, this tail couples the incoming F(qy·e^{iθ})
  — which grows to ~1e5 at y=120 under CS — from small y up to large y. The source `V·Rxy·ΩR` then does
  not localize (decays only ~1/√y instead of exponentially), and Eq.17 (growing bra × algebraic-tail source)
  diverges with the box: η = 0.57 (ymax60) → 25 (ymax90) → 5549 (ymax120). Proof: Rxy·ΩR(y_large)=1e-21 but
  src(y_large)=18 ⟹ src at large y comes non-locally from small y via the overlap tail.
- **Rimas's 2011 paper used local Hermite splines (banded overlap, no tail) — but he says he later switched
  to Lagrange-Laguerre, so he handles this. That is the question in the email.**

**[ ] WAIT for Rimas's reply** (`~/Downloads/email-to-lazauskas-scattering-extraction.txt`). Key questions:
how to keep source+integral convergent in Lagrange-Laguerre given the growing incoming wave; what θ and
y_max for n-d @14.1 MeV; Eq.17 vs Eq.16 and the G̃ regularization.

**[ ] After reply — candidate fixes (don't start before hearing back):**
  (a) limit y_max so F doesn't overflow precision (e^{q·ymax·sinθ}≲1e2-3 → ymax≲40-50 @θ=10°), trading
      asymptotic reach; cheapest, verify Eq.17 converges + hits benchmark in that window.
  (b) switch the y-direction basis to local splines (matches Rimas 2011) — large change, root cure.
  (c) handle the overlap consistently in the extraction (B-weighted inner product / de-project), not
      treating reduced-rep coefficients as function values.
  (d) reduce θ (slower F growth, lighter tail contamination); pair with (a).

---

## ✅ Done 2026-06-16 — scattering η bug traced from the physics outward

Target: Lazauskas-Carbonell PRC 84,034002 Tab.III doublet n-d, E_lab=14.1 MeV → **Re δ=105.49°, η=0.4649**.

### Layer 1 — the inner-product convention (UNDERSTOOD, physically confirmed)
- **Bug**: `compute_scattering_amplitude` (`swift/scattering.jl:~350`) projected the amplitude with
  Julia `dot` (Hermitian, conjugated bra). Under complex scaling the rotated H is **complex-SYMMETRIC,
  not Hermitian**, so the projection must be the **bilinear c-product** (`transpose`, NO conjugation).
  A conjugated projection restores an artificial Hermiticity → single-channel elastic unitarity |S|=1
  → η pinned ≈1, blind to the breakup flux that ψ_sc demonstrably carries (21%, verified 2026-06-15).
- **Deeper layer — CS eigenvector global phase**: `eigen` returns the deuteron eigenvector with an
  arbitrary global complex phase e^{iγ}; `bound2b` fixes only |evec| (Hermitian B-norm), not the phase.
  The bilinear extraction `transpose(φ)…φ ∝ e^{2iγ}` drags this unphysical phase into f (→ η=1.59, worse),
  whereas Hermitian `dot` cancels it (e^{-iγ}e^{iγ}) — which is *why* `dot` looked "stable" at η≈1.
  Dividing by the **deuteron c-norm** `C_n = φ_d^T B φ_d = e^{2iγ}·|C_n|` removes exactly this phase.
  → **This is the missing C_n⁻¹ flagged on 2026-06-15; it is a GAUGE factor, not a free knob.**
- **Confirmation that |C_n| is physical, not numerical**: |C_n| = **0.9626** (reduced mesh) vs **0.9624**
  (full mesh) — identical; only arg(C_n) jumps run-to-run (+113° vs −70.7°) = the random eigenvector phase.
  Bound-state c-norm should be θ-independent ≈1; |C_n|≈0.962 is consistent.
- **Result** (θ=10°, 14.1 MeV doublet, elastic λ=0,𝕊=1/2):

  | extraction | Re δ | η |
  |---|---|---|
  | benchmark | 105.49° | 0.4649 |
  | old `dot` (Hermitian) | 78.9° | 1.165 (unphysical, η>1) |
  | `transpose` only | 29.2° | 1.372 (e^{2iγ}-contaminated) |
  | **`transpose × 1/C_n`** | 76.3° | **0.273** (physical, η<1) |

### Layer 2 — residual LOCALIZED to the projection operator (NOT mesh, NOT normalization)
- **Mesh ruled out**: reduced (nx12,ny30,ymax50) → full (nx30,ny70,ymax60) moves η 0.300→0.273 and
  δ 54°→76° (NOT converging toward 0.46/105). rel_res=9.3e-5, GMRES converged. So 0.27 is not undercon-
  vergence.
- **Normalization ruled out for δ**: δ sits at ~77° for BOTH `dot` (78.9°) and `transpose×1/C_n` (76.3°),
  a ~28° deficit vs 105° that is invariant under the normalization change. → **δ is fixed by the
  V·Rxy_31 projection structure itself.** This is the fingerprint that the extraction OPERATOR
  `⟨φ_d F|V·Rxy_31|ψ_total⟩` (a hybrid) is not Lazauskas' authoritative Eq.16/17 form.

### Separate confirmed bug — FIXED 2026-06-16 (found by Codex cross-validation)
- `ndscatt.jl` passed `θ=θ_deg` (degrees, =10.0) to `compute_initial_state_vector`, which uses
  `exp(im*θ)` (radians) at `matrices_optimized.jl:1186` → the production driver ran at θ≈10 rad (≈573°),
  not 10°, corrupting all `ndscatt.jl` output. `benchmark_nd_scatt.jl` and the test scripts pass `θ_rad`
  correctly, so the η diagnosis above is unaffected. **Fixed**: `ndscatt.jl` now defines `θ_rad` and
  passes it to `compute_initial_state_vector` (the `θ_deg=` kwarg of `solve_scattering_equation` is
  correct as-is). The old "Standard calculation (no complex scaling)" comment on θ_deg=10.0 was also wrong.

### Code state after this session
- `swift/scattering.jl`: `compute_scattering_amplitude` gained a `conj_bra::Bool=true` kwarg.
  **Default = true = old Hermitian `dot`** (production behaviour unchanged on purpose, since the
  bilinear path is not benchmark-complete). `conj_bra=false` selects the bilinear c-product.
- `swift/twobody.jl`: **UNCHANGED** — the bound-state B-normalization is benchmarked and correct; the
  1/C_n factor belongs in the scattering extraction, not in the bound-state code. (A transient edit was
  made and reverted this session.)
- **New file** `swift/test_cnorm_extraction.jl`: one reduced/full-mesh doublet solve, extracts the
  elastic S-matrix four ways (`dot`; `transpose`; `transpose×1/C_n`; `transpose×1/C_n²`) and computes
  C_n from the deuteron wavefunction. This is the layer-1/layer-2 diagnostic of record.

---

## ✅ Done 2026-06-16 (cont.) — Step A 2-body isolation PASSED: the missing factor is a CS Jacobian

`swift/test_2body_cs_1S0.jl`: self-contained single-coordinate CS MT ¹S₀ two-body scattering
(inhomogeneous Lazauskas method), reusing the proven 2-body builders (`T_matrix_scaled`, `Bmatrix`,
the Lagrange basis); V built inline forcing s=0 (¹S₀). Direct solve, residual 1e-14.

- **First found**: even in the 2-body case η≠1 (bilinear 1.16, Hermitian 1.49) — but η MUST be 1
  (single open channel, real potential). So the bug is in the SCALAR amplitude machinery, NOT in any
  3-body recoupling. Isolation succeeded.
- **Decisive clue**: with `f = -2μ/(ħ²k²)·⟨F_0|V|u_tot⟩` the **magnitude |f|=5.766 matches the exact
  |f_true|=5.764 to 4 digits**; only the PHASE was off by ~10° ≈ θ.
- **Fix identified**: `f = -2μ/(ħ²k²)·e^{+iθ}·⟨F_0|V|u_tot⟩_bilinear`. The `e^{+iθ}` is the **CS contour
  Jacobian** (one factor per rotated radial integration; the backward-rotation V matrix carries the
  Hamiltonian's e^{-iθ}, but the amplitude integral over the rotated contour `d(re^{iθ})` needs the
  forward e^{+iθ}). With it: δ=63.22°, **η=0.999** (nx=100) → unitarity restored, δ within CS
  convergence of the exact 63.512° (single θ=10°, no θ-plateau yet).
- **Confirms**: (i) bilinear c-product is required; (ii) the prefactor / F_0 normalization / S=1+2ikf
  are all correct; (iii) the layer-2 residual is the **missing CS Jacobian**, exactly the e^{3iθ}/e^{6iθ}
  factors in Lazauskas Eq.16/17. The "wrong projection operator" hypothesis is now refined: the operator
  may be fine; it was missing its CS Jacobian (and 1/C_n).

## ✅ Done 2026-06-16 (cont.) — global CS-Jacobian RULED OUT for the 3-body V·Rxy amplitude

Scanned `f = bilinear × (1/C_n) × e^{inθ}`, n=0..4, on the full-mesh ψ_sc (no new solve). With the
δ↔δ−180 ambiguity unwrapped: n=0 (76.3°,0.273), n=1 (100.3°,0.262), n=2 (117.8°,0.372),
n=3 (127.4°,0.531). **δ and η cannot be hit simultaneously**: δ≈105° lands near n≈1.2 (η≈0.27, too low),
η≈0.465 lands near n≈2.7 (δ≈123°, too high). The two never cross.

**Proof this is not a Jacobian problem**: `f` is linear in `temp2 = V·Rxy·ψ`, so any *node-independent*
factor (e^{inθ}) on V/Rxy equals a global phase on f = exactly this scan. CS measure jacobians
(x²dx → x²e^{3iθ}dx) contribute only the *constant* e^{3iθ} → also global. A constant Jacobian
therefore cannot fix it. Moreover the magnitude is wrong too (at the δ-matching n, η is off), and a
phase cannot fix magnitude. ⟹ **the V·Rxy projection operator is structurally ≠ the correct extraction.**
(The 2-body succeeded because it is a single-integral, single-channel scalar where |f| was already exact
and a global e^{iθ} sufficed; that does not carry over to the 3-body V·Rxy volume integral.)

## ✅ Done 2026-06-16 (cont.) — Eq.16 asymptotic RULED OUT; got the exact Eq.17 from the paper + thesis

- **Eq.16 (asymptotic y-projection) tried and FAILED**: prototyped in `test_cnorm_extraction.jl` on the
  full-mesh ψ_sc; f_λ(y) does NOT plateau, grows to the boundary (|f|~50, noise). This independently
  confirms what Lazauskas told Jin directly AND what the paper says (§II.A: Green's-theorem Eq.7 is
  ~1 digit more accurate; "differential relations ... do not lead to very convincing results"). The
  HDR (2019) §5.5: "the Green's-theorem integral relation is systematically more accurate." **Use the
  Green's-theorem extraction (Eq.17), NOT the asymptotic projection.** (memory: lazauskas-greens-theorem-not-asymptotic)
- **Read the actual paper Eqs.16/17 (PDF p3-4) and Rimas' thesis (`~/Desktop/TALENT-Trento-2015/LAZAUSKAS R/
  Rimas-thesis.pdf`, Appendix H + §1.1.4 + §2.3).** The TALENT folder has only teaching codes (bound
  state, ordinary CC phase shifts via `atan(K)`, resonances) — NO 3-body CS amplitude code.

## ★★★ K-matrix integral-relations method (2026-06-16) — ⚠️ SUPERSEDED, see 2026-06-17 top section ★★★

> **2026-06-17 verdict**: this whole K-matrix route (from the Duerinck STUDENT thesis) was the
> overcomplication. Rimas himself uses his 2011 Eq.17 (Green theorem). The 3-body K-matrix below
> never converged (δ stuck) for the same root cause now identified: the Lagrange-Laguerre overlap
> tail + growing CS incoming wave. Kept below for the record only. Do NOT resume the K-matrix path.

Lazauskas sent his student P.Y. Duerinck's PhD thesis (`~/Downloads/PhD_Thesis_PYDuerinck_REV.pdf`,
§1.1.6 / §1.3.5 / Appendix A). NOTE (per Jin): Eq.16/17 Green's theorem and the K-matrix integral
relations are BOTH the group's methods — equivalent routes to the S-matrix, NOT one superseding the
other. Eq.17 is not "wrong"; my IMPLEMENTATION of it was incomplete. The thesis's value: it spells out
the K-matrix route clearly + multichannel, and exposes the 3 things my Eq.17 attempts lacked (which
BOTH routes need): (1) regularized irregular G̃=G(1−e^{−r/b})^{2l+1} (un-regularized G → my divergence);
(2) the correct multichannel structure; (3) regular-F-class bra + correct Wronskian/operator. The
K-matrix route is just EASIER to land in swift (K linear, one n_c×n_c linear system, reuses existing A).
Full recipe in memory `kmatrix-integral-relations-method.md`.

**3-body recipe (Eq.1.342-1.357), reuses swift's existing matrices verbatim:**
1. Channels c={l_y,j_y}: (0,½) and (2,3/2) for J=½⁺ (matches B5).
2. `Ω_c^(R)=(1/y)F_{l_y}(η,qy)[φ_d⊗(l_y s_k)j_y]` (regular F bra) and `Ω̃_c^(I)` with the **REGULARIZED
   irregular `G̃_{l_y}=G_{l_y}(1−e^{−y/b})^{2l+1}`** (this regularization tames the G-divergence/CS-growth).
3. Per entrance channel c0, build sources `b_c^(R)∝u_n(x)F_{l_y}(qy)`, `b_c^(I)∝u_n(x)G̃_{l_y}(qy)` and solve
   `[EN−H0−Vx−P]v_c^(λ) = −[EN−H0−Vx−P]b_c^(λ)` (swift's existing A + GMRES) → v_c^(R), v_c^(I).
4. K-matrix from the Wronskian linear system (1.357 / App A.2 n_c×n_c): `⟨Ω_c^(R)|H0y+VCy−Erel|Φ_α⟩ =
   (ℏ²/m0)q·K_cc0` (neutral: H0y−Erel). Equivalent 2-body forms App A.12/A.16.
5. `S=(1+iK)/(1−iK)` (matrix), diagonalize → doublet eigenphase. Target δ=105.49°, η=0.4649.
KEY: "the matrices involved in both bound and scattering states are identical" — only NEW code is G̃, the
R/I sources, the driven solves, and the K-matrix linear system. CS: same integral relations, rotated ops.

**[x] 2-body K-matrix DONE + VALIDATED 2026-06-16 (`swift/test_2body_kmatrix.jl`):** CS MT ¹S₀, b=12 →
K=2.006 (real), δ=63.506°, η=0.9999 = Lazauskas Table I (63.512°, η=1), more accurate than amplitude route.
Method locked (see memory `kmatrix-integral-relations-method.md` for the validated recipe). Critical fixes:
(1) COULCC must be mode=1 not mode=4 (mode=4 gives gc=0 — hidden bug, also in compute_initial_state_vector
but harmless there since source only needs F); (2) φ^(I) source needs the localized regularization defect
d=(H̄0^θ−E)G̃ = −E[2G₀'(z)reg_z'+G₀(z)reg_z''] (analytic, from COULCC G₀,G₀'); (3) K=−I_R/(ℏ²k/2µ+I_I),
S=(1+iK)/(1−iK); b≳8 converges. Consistency self-check I_I→−i·I_R.

**[~] 3-body multichannel IMPLEMENTED, η ballpark-right, δ NOT yet pinned** (`swift/test_3body_kmatrix.jl`,
2026-06-16). 2 channels (l_y=0,j_y=½)=doublet + (l_y=2,j_y=3/2). Per-incident-λ: (R) solve A·v^(R)=V·Rxy·Ω^(R)
(=masked existing solve); (I) solve A·v^(I)=B·d3+V·Rxy·Ω̃^(I) with d3=φ_d⊗d_y, d_y=−E_cm[2G_{l_y}'(z)reg_z'+
G_{l_y}(z)reg_z''] (same form all l_y, centrifugal cancels). K from [(ℏ²q/2µ_y)δ+M^(I)]K=−M^(R),
M^(λ)=(jac/C_n)Σ_{iα∈c}⟨Ω_c^(R)|V·Rxy|U^(λ)⟩. S=(I+iK)(I−iK)⁻¹, diagonalize.
STATUS: at jac e^{1θ}, b=8: doublet δ=62.85°, **η=0.4117 (close to benchmark 0.4649!)**, but δ off (vs 105.49°).
Doublet K[1,1]=0.97+1.21i vs benchmark −1.14+1.87i (Im ballpark-right, Re wrong sign → phase issue).
FIXED this session: λ=2 reg must use power (1−e)^{2l+1}=^5 (was ^1 → MI[2,2]=−32021 killed λ=2; now sane).
**OPEN (next session):** (1) doublet δ phase calibration — K[1,1] Re sign wrong, jac/C_n phase power for the
3-body (x,y) double integral not pinned (jac e^{nθ} scan moves δ 53→63→79 for n=0,1,2; η tracks 0.57→0.41→0.30).
(2) off-diagonal MR[1,2]=MR[2,1]=0 EXACTLY — verify whether λ=0/λ=2 coupling is physically ~0 for central MT
or a projection bug. (3) "other" (λ=2) eigenphase η>1 (1.24-1.40, unphysical) — λ=2 self-K needs checking.
Likely lever: the jac/C_n convention for the 3-body matrix element (x-integral=deuteron→C_n, y-integral→jac);
anchor against the validated 2-body (test_2body_kmatrix.jl) where jac=e^{iθ}, one open coordinate. Target δ=105.49°, η=0.4649.

---

## ▶ (Eq.17 Green's-theorem route — equivalent to K-matrix above; my earlier incomplete attempt)

Target (PRC 84,034002 Eq.17, neutral n+d):
```
f_{nm}(ŷ) = -C_n^{-1} (m/ħ²) ∫∫ φ_n*(x e^{-iθ}) [e^{-i q_n y e^{iθ}}/y] [V_j(x_j e^{iθ})+V_k(x_k e^{iθ})] Ψ̄_m(x,y) e^{6iθ} d³x d³y
C_n = ∫ φ_n*(x e^{-iθ}) φ_n(x e^{iθ}) e^{3iθ} d³x      (complex; |C_n|≈0.96, arg = eigenvector gauge phase)
```
**★ BREAKTHROUGH 2026-06-16 (`test_eq17_magnitude.jl` operator sweep + COLOSS source compare):**
The amplitude operator was WRONG. It is the **post-form** `⟨ψ_in| V·Rxy |ψ_total⟩`, NOT `Rxy·V·Ψ̄`.
Swept all operator choices against benchmark η=0.4649:
  - `Rxy·V·Ψ̄` (the previously-documented "derived" operator) → η=82.9  (×58 too big)
  - **`V·Rxy·ψ_total` (= 2·V·Rxy_31·ψ_total, the SAME operator that builds the source b) → η=0.471** ✓ (1.3%)
ψ_total = ψ_in+ψ_sc is the Faddeev component. The `Rxy·V·Ψ̄` identity (still in `test_V23V31_operator.jl`)
is the BOUND-STATE expectation operator, NOT the scattering amplitude → that was the ×58 wall.

Concrete pieces, status after the fix:
1. [x] **Operator = `V·Rxy·ψ_total`** (post-form `⟨ψ_in|V·Rxy|ψ_total⟩`). Gives η=0.471 ≈ benchmark.
2. [x] **Object** = Faddeev component `ψ_total=ψ_in+ψ_sc` (NOT the full `(I+Rxy)ψ` — that double-counted).
3. [x] **Bra = regular F (=ψ_in), NOT Hankel ĥ⁺.** Confirmed by COLOSS (`scatt.f` uses `fc_rotated`,
       regular) + thesis Eq.1.63 (regular ĵ_l). The old "use ĥ⁺" note was backwards.
4. [x] **Scalar convention locked by COLOSS** (`/Users/jinlei/Desktop/code/COLOSS`, 2-body CS ref code,
       `scatt.f:41-71`): `f = −(e^{iθ}/E_cm)·Σ V·fc·ψ·sqrt(rw)` (one e^{iθ} per radial integral, regular
       fc bra, prefactor −1/E_cm = −2μ/ħ²k², `S=1+2ik·f`). Physical (non-mass-scaled) Jacobi coords in
       swift: x-kinetic ħ²/m (μ_x=m/2), y-kinetic (3/4)ħ²/m (μ_y=2m/3) → prefactor uses μ_y=2m/3.
5. [x] **Measure**: swift `V = V_x ⊗ Ny` carries BOTH the x-integration (in v12) and the y-overlap
       metric (Ny), so `transpose(ψ_in)·V·g` is the correct bilinear form — no extra B-metric needed.
6. [ ] **C_n^{-1}** removes the deuteron eigenvector gauge phase (|C_n|≈0.96). Phase power (e^{1iθ} 1D
       vs e^{3iθ} 3D) still to be pinned together with item 7.

**═══ CONSOLIDATED STATE 2026-06-16 (idea-pk + B1/B5 diagnostics + multichannel attempt) ═══**
Read the FULL paper PRC 84 Eq.16/17/19 (`~/Downloads/PhysRevC.84.034002.pdf`). Structure 100% confirmed:
Eq.17 operator [V_j+V_k]Ψ̄ (=V₂₃+V₃₁, exclude deuteron pair), bra y-kernel e^{−iqy e^{iθ}}/|y| (=Riccati
ĥ⁻₀ for l=0), full Ψ̄, prefactor −C_n⁻¹(m/ħ²)e^{6iθ}. Benchmark re-confirmed δ=105.50°/η=0.4653. Paper
itself notes "Eq.19 converges slowly in y". θ_max=14.2°@14.1MeV (θ=10° legal). Paper uses Hermite splines
30-40/dir; swift uses Laguerre (different basis → boundary/convergence handling differs).

What's SOLID now:
- Scalar 2-body locked (test_2body_cs_1S0.jl δ=63.512°, = COLOSS scatt.f exactly). Prefactor −(1/E_cm)
  (2-body anchor, NOT paper's m/ħ²) is the right magnitude scale.
- B1 (`test_b1_b5_diag.jl`): Rxy is NOT metric-symmetric, `‖B·Rxy−Rxyᵀ·B‖/‖‖=1.40` (O(1)). So the
  P-algebra step `[V₂₃+V₃₁]Ψ̄=Rxy·V·Ψ̄` (assumes Rxyᵀ=Rxy) BREAKS in the truncated basis → that is the
  674-vs-4.4 ordering discrepancy. Ordering matters but is not the sole δ≈0 cause.
- B5 (`test_b1_b5_diag.jl`): `iel=first(cpl)` keeps ONLY (λ=0,J₃=½, |Mp|=4.36, arg −103°) and DROPS
  (λ=2,J₃=3/2, |Mp|=5.12 — bigger!); two ³D₁ blocks are 0. The projection is incomplete.

What's RULED OUT (cheap fixes that DON'T work):
- Born/scattered split (the old "▶ NEXT"): in 2-body it's IDENTICAL to lumped (test_2body_cs_1S0:82-99,
  δ=63.224 either way) → NOT the fix. ψ_sc-only Green integral also diverges.
- Naive channel sum (`/tmp/phase_scan.jl`): nx16 gives η=0.461≈benchmark but NOT mesh-stable (|f|
  0.84→0.49) and NO global phase reaches δ=105 (η=0.4649 only at δ=±33°/±3°).
- Single-solve nd×nd block matrix + recouple (`test_eq17_recouple.jl`): garbage (η 0.5–5, mesh-unstable);
  off-diagonal M[io,ii] from ONE solve is not a real S-matrix element.
- Rigorous 2×2 per-incident-λ solve + diagonalize (`test_eq17_mc.jl`, idea-pk path a, ATTEMPTED): gives
  **offdiag f12=0.00 exactly (channel coupling not captured) and doublet η>1 (unphysical)** → "mask
  incident λ + project channel block" still does NOT isolate the outgoing-channel amplitude correctly.

ROOT CAUSE (consolidated): δ≈0 / non-convergence is fundamentally **the rigorous extraction of the
multichannel elastic S-matrix**, not a one-line operator/prefactor fix. The cheap levers are exhausted.

▶ NEXT — three real paths (idea-pk panel `idea-pk-output/20260616-145124/decision-panel.md`), all
non-trivial; pick one:
  (a) Real multichannel S-matrix: per-incident-channel solve done CORRECTLY (fix the f12=0 / η>1 bug in
      test_eq17_mc.jl — the outgoing projection must give a proper S-matrix element, likely needs the
      correct bra normalization + the V₂₃+V₃₁ operator with the metric-correct Rxy adjoint from B1, not
      the symmetric shortcut).
  (b) Implement thesis App H partial-wave amplitude formula (M-matrix→S^J) verbatim.
  (c) Spectral COLOSS route: f_sc=Σ dₙ²/(E−Eₙ) from swift's existing eigen(H,B), V-protected dₙ.
  Consider asking Lazauskas directly (his own paper flags the y-convergence as delicate).
Old text below (superseded; kept for the Eq.17 piece-by-piece checklist):
- ▶ (superseded) port COLOSS's Born/scattered split — RULED OUT above. Cross-check scalar vs `test_2body_cs_1S0.jl`
  (δ=63.512°) and run COLOSS itself on MT ¹S₀ if a 2-body sanity is needed.

- Validate: `test_cnorm_extraction.jl` reduced+full mesh → η→0.4649, δ→105.49°; then `benchmark_nd_scatt`
  doublet/quartet @14.1 & 42 MeV, θ-plateau ([4°,12.5°]@14.1) + mesh (ymax→100) convergence.
- Scalar conventions (bilinear, ĥ normalization, e^{iθ}-per-radial Jacobian) already locked by the
  2-body test `test_2body_cs_1S0.jl` (δ=63.512° reproduced) — reuse them.
- References: paper PDF `~/Downloads/lazauskas2011.pdf` (Eq.16 p3, Eq.17 p3, Eq.24 Coulomb p4);
  thesis `~/Desktop/TALENT-Trento-2015/LAZAUSKAS R/Rimas-thesis.pdf` (App H p197-200 amplitude/observables,
  §1.1.4 p15 2-body integral phase shift, §2.3 p71 scattering BC); HDR §5.5.

## Housekeeping
- [ ] Once the extraction is fixed: fold 1/C_n into `compute_scattering_amplitude` and set the correct
      `conj_bra` default; update `benchmark_nd_scatt.jl` to use it.
- [ ] Decide whether to keep `test_cnorm_extraction.jl` / `test_scatt_diag*.jl` / `idea-pk-output/` in
      the repo or gitignore them.
- [ ] Move repo out of the cloud-synced folder OR add the conflicted-copy glob to `.gitignore`.

---

## Done earlier (2026-06-15) — code cleanup + bound-state verification (unchanged)
- **Code cleanup → only the fastest path remains**: V-sector M⁻¹ only (`matrices.jl`),
  V-sector + Arnoldi only (`MalflietTjon.jl`), GMRES-only scattering (`scattering.jl`); deleted
  `compare_solvers.jl`. ħ²/m_N = **41.471** MeV·fm² uniformly.
- **Bound states** (cleanup is split-independent): swift_3H AV18 **E(³H)=−7.191400 MeV**;
  swift_3He −6.909; swift_3H_MN −8.302. **benchmark_rimas full jx=1..6 reproduces rimas AV18 3H/3He
  to ~5 keV** (m_N-convention level), exact channel counts, conv=true. Cross-validation PASSED.
- Commits: `c647f1c` → `71c6bf8` → `7138d37` (+ uncommitted scattering-debug WIP).

---

## Lessons (project memory at `~/.claude/projects/-Users-jinlei-Desktop-code-swift-jl/memory/`)
1. Arnoldi eltype trap — buffers via `eltype(v)`, not the cache type param.
2. swift.jl authorship framing — Jin's independent code following rimas's methodological line.
3. Lazauskas benchmark provenance — rimas personal comm., T=1/2, ħ²/m_N=41.471 (now matched exactly).
4. V-sector M⁻¹ near-singular ON-SHELL → trust the RELATIVE GMRES residual (~2e-5), not the absolute.
5. In CS scattering, η<1 is NOT automatic from a breakup-carrying ψ_sc — it requires the bilinear
   c-product extraction + the deuteron c-norm 1/C_n.
6. **NEW (2026-06-16)**: CS amplitude extraction is sensitive to the eigenvector's arbitrary global
   phase. Bilinear `transpose(φ)…φ ∝ e^{2iγ}` carries it; dividing by `C_n = φ^T B φ` removes it.
   `dot` (Hermitian) hides the phase but also hides the absorption (η→1). |C_n|≈0.962 is mesh-stable;
   only arg(C_n) is the random gauge phase.
7. **NEW (2026-06-16)**: when an extracted observable (here δ≈77°) is INVARIANT under a normalization/
   convention change, the bug is in the OPERATOR being projected, not the normalization. δ stuck 28°
   low under both `dot` and `transpose×1/C_n` ⟹ the V·Rxy projection ≠ Lazauskas Eq.16/17.
8. **★ NEW (2026-06-17), the root cause**: the n-d scattering extraction divergence is NOT a formula,
   coefficient, or rotation bug (all verified correct). It is the **Lagrange-Laguerre non-orthonormal
   overlap** `N[i,j]=δ+(-1)^{i-j}/√(yᵢyⱼ)` (compute_overlap_matrix:49): its long-range 1/√y tail, via
   `V_block=kron(V_x,Ny)`, couples the exponentially-growing CS incoming wave (F~1e5 @y120) from small
   y to large y, so the source/integral decay only algebraically and Eq.17 diverges with the box. The
   bound state is immune (exponentially localized). Lazauskas's 2011 paper used local Hermite splines
   (banded overlap, no tail); he reportedly later uses Lagrange-Laguerre, so a specific trick exists —
   that is the question in the email. Diagnostic of record: `swift/test_3body_kmatrix.jl` per-y norms.
9. **NEW (2026-06-17)**: phase-shift extraction back-end = Glöckle PhysRep 274 (1996) Eq.209-214
   (Blatt-Biedenharn channel-spin S=U†ΛU); it is convention-free on S. Build S coordinate-space
   (S=1+2iq·f, Rimas Eq.6) — do NOT mix momentum-space U→S factors. Coefficient = ℏ²q/2µ_y = 31.10·q.

*End 2026-06-17. ⏸ BLOCKED: waiting for Rimas's reply (email in ~/Downloads). Do not resume the K-matrix
or any volume-integral variant — they all fail on the overlap-tail root cause. Resume at the top
"⏸ BLOCKED 2026-06-17" section once Rimas answers; candidate fixes (a)-(d) listed there.*
