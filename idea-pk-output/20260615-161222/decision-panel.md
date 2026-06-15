# idea-pk decision panel — swift.jl n-d scattering η≈1 bug

Goal: reproduce Lazauskas PRC 84,034002 Tab.III doublet (Re δ=105.49°, η=0.4649) at E_lab=14.1 MeV.
Symptom: η≈1 (no breakup absorption), δ≈79°. Recoupling + linear solve already verified correct.
Budget noted: full run ≈220 s; dense A memory ~(nch·nx·ny)²; stay in R-space Faddeev+CS.

Dual-solver verdict: the bug is ONE of two families, and there is a single cheap test (B5) that
decides which before you spend another day. (Groupthink was flagged on round 1 — both solvers
anchored on "extraction"; a respawn opened the orthogonal "generation" branch A5/A6/A7.)

## ⭐ DO FIRST — the router (diagonal + cheapest disambiguator): B5
**B5 — measure whether ψ_sc actually contains breakup-continuum content.** Postprocess the EXISTING
solution (no new solve): project the deuteron bound state φ_d(x) out of each channel block of ψ_sc and
report ‖ψ_sc − P_d ψ_sc‖/‖ψ_sc‖ per channel; repeat on Rxy_31·ψ_total and V·(Rxy_31·ψ_total).
- continuum fraction SIZABLE but η≈1  → bug is in EXTRACTION (H1) → go to Branch 1.
- continuum fraction TINY already in ψ_sc → CS source/operator never generated breakup (H2) → Branch 2.
- cost: half day, no new solve. risk: projection metric must use the CS x-overlap convention; qualitative.

## Branch 1 — EXTRACTION / normalization (H1)  [convergent: both solvers proposed]
The wiki has the benchmark paper (Lazauskas-Carbonell 2011) with the EXACT amplitude formula; swift's
`f=-4μ/(ħ²k²)·⟨φ|V·Rxy|ψ⟩` may be missing the deuteron-norm coefficient C_n⁻¹ and the CS Jacobians
e^{3iθ}/e^{6iθ}, and may use the wrong pair potential ([V_j+V_k] vs V_i).

- **B1 — 2-body isolation (cleanest).** Build a 1-coordinate complex-scaled MT ¹S₀ two-body amplitude,
  θ=10°, r_max=100, and check δ=63.512° (Lazauskas Table I, exact to 5 digits). Removes deuteron,
  recoupling, breakup entirely. If it misses 63.512°, H1 is proven in the simplest setting. ~1 day.
- **B2 — scalar factor sweep (fastest, on frozen ψ).** On one saved reduced solution, multiply the raw
  matrix element by the candidate missing scalars {C_n⁻¹, e^{3iθ}, e^{6iθ}, −(m/ħ²)C_n⁻¹e^{6iθ}} and see
  if the doublet η moves from ≈1 toward 0.4649 WITHOUT changing ψ_sc. 2–4 h. [info-gain corrected: a
  scalar that fixes η is suggestive, NOT proof of the full Eq.17 operator — the [V_j+V_k] projection is
  still untested.]
- **B3 — Eq.16 projection (independent extraction).** Implement Lazauskas Eq.16 y-asymptotic projection
  (C_n⁻¹·y·e^{-iqy e^{iθ}}·∫φ_d*(xe^{-iθ})ψ_sc e^{3iθ}dx, plateau over last 10–15 y nodes), bypassing the
  current V·Rxy integral. Stable plateau η≈0.46 → current integral is the wrong projection; plateau η≈1
  → H2; no plateau → grid/window can't support Eq.16 (use Green-theorem Eq.17). ~half day, no new solve.

## Branch 2 — GENERATION: does the CS solution carry breakup flux? (H2)  [the diagonal branch]
- **A6 — incoming source wave rotation (sharp, specific).** `compute_initial_state_vector`
  (matrices_optimized.jl:1244-1259) evaluates the COULCC F_λ at the ROTATED argument k·y·e^{iθ}, so ψ_in
  itself decays exp(−ky sinθ). In Lazauskas' inhomogeneous form Ψ=Ψ^in+Ψ^sc, only the SCATTERED part
  should decay; if the incoming source is wrongly rotated it becomes a square-integrable driver the
  elastic channel re-absorbs trivially → η→1. Test: a flag to feed F_λ(k·y) un-rotated into the source b
  while keeping operator A scaled; 2 reduced runs + a |ψ_in(y)| plot. ~1 day. RISK: the rotated Ψ^in may
  be CORRECT (so V·Ψ^in is the genuine localized source) — treat an η-drop as a CLUE to justify against
  Lazauskas' source equation, not a fix to keep blindly.
- **A5 — θ-plateau diagnostic (extraction-free).** Scan θ∈{4,6,8,10,12}° at fixed reduced mesh; record
  δ,η AND the extraction-free interior breakup weight W_brk=Σ_{non-deuteron ch}⟨ψ_sc|B|ψ_sc⟩. A correct CS
  calc plateaus in θ with η<1 and nonzero W_brk; flat η≈1 with W_brk≈0 for ALL θ proves the continuum is
  never generated (exonerates extraction). ~1 day, 5 fast runs. [step refined: report a plateau over a
  defined θ-window + one production-mesh θ=10° W_brk point, not a single θ.]
- **A7 — operator-side CS consistency.** Rxy is built θ-independently while V and T are rotated. Check
  whether the deuteron↔breakup coupling block of V·(I+Rxy) actually carries the e^{iθ} rotation (compare
  block norm at θ=0 vs 10°); a θ-frozen coupling block means the rotated continuum's FEED is unrotated →
  flux can't transfer → η≈1. No solve, 2 matrix builds. RISK: Rxy genuinely IS geometric/θ-free — reason
  at the V·Rxy product level, not Rxy alone.

## Secondary (rule-out)
- **B4 — energy-convention triplet.** Split E into E_operator/E_source/E_amp and test the 3 legal
  combinations. [refined: first do a logging AUDIT of which E each of the 3 call sites currently uses,
  before any scan.] Rule-out only — the η gap (1 vs 0.46) is too large for a √E prefactor slip alone.

## Author notes (incompatibilities / sequencing)
- B5 → then exactly ONE branch. The cross-check flagged (B5,B3) and (B3,A6) as mutually constraining:
  if B5 finds tiny continuum content, B3 cannot also find an η≈0.46 plateau on the same ψ_sc (if it does,
  a projection metric is inconsistent). So B5 first; do not run Branch 1 and Branch 2 in parallel.
- B1 is the only path that needs NO existing 3-body solution and isolates the convention with a 5-digit
  reference — strongest standalone confidence-builder, independent of B5's routing.
- Dropped by cross-check: A1,A2 (redundant→B3), A4 (redundant→B1), A3 (leaves the CS framework, infeasible).
