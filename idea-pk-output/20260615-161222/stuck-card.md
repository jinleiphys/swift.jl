# Goal

Make swift.jl (Julia configuration-space Faddeev code, complex-scaling) reproduce the
Lazauskas-Carbonell PRC 84, 034002 (2011) Table III benchmark for n-d ELASTIC scattering
with the MT I-III potential, doublet channel (S=1/2, total J=1/2), L=0:
  target  Re(δ) = 105.49°,  η (inelasticity = |S|) = 0.4649   at E_lab = 14.1 MeV (θ_CS = 10°).
The deuteron bound state and the AV18 3H/3He bound-state benchmark already match Lazauskas
to ~5 keV. The BOUND-state machinery is trusted. The SCATTERING phase-shift extraction is the problem.

# Current state

Method: inhomogeneous Faddeev CC equation [E_total·B − T − V·(I+Rxy)] ψ_sc = 2·V·Rxy_31·φ_in,
solved by preconditioned GMRES (V-sector block-diagonal M⁻¹). Complex scaling r→r·e^{iθ},
θ=10°. Potential MT I-III (central, S-wave only → deuteron is PURE ³S₁, no ³D₁).
n-d reduced mass μ=2m/3, ħ²/m_N=41.471. E_cm = (2/3)·E_lab. Deuteron E_d=-2.2301 MeV (matches MT ref).

Pipeline: bound2b → deuteron φ_d (CS); compute_initial_state_vector builds ψ_in = φ_d(x)·F_λ(q·y)
with q from E_cm (COULCC regular function); solve_scattering_equation builds A and solves GMRES;
compute_scattering_amplitude gives partial-wave f-matrix f = prefactor·⟨φ_out|V·Rxy_31|ψ_total⟩,
prefactor = -4μ/(ħ²k²); collision matrix U = 2ik·f + I; recouple_to_channel_spin transforms U from
the (λ,J3) deuteron-channel basis to the channel-spin (λ,𝕊) basis via a 6j recoupling; elastic
doublet observable = the (λ=0,𝕊=1/2) diagonal element S, with δ=½·arg(S), η=|S|.

CURRENT RESULT (after all fixes below): at full mesh (nx=30,ny=70,xmax=30,ymax=60):
  Re(δ) ≈ 78.9°  (target 105.49°),  η ≈ 1.16  (target 0.4649).
At reduced mesh (nx=12,ny=30,xmax=24,ymax=50): Re(δ)=74.2°, η=1.013.

THE CORE SYMPTOM: η ≈ 1 (elastic S-matrix essentially UNITARY, no flux loss), but the benchmark
η = 0.4649 means ~54% of the flux is absorbed into 3-body BREAKUP. So breakup absorption is NOT
being captured — the complex-scaled elastic channel stays ~unitary. δ is in the right ballpark
(~75-79° vs 105°) but off by ~27°.

# Tried (with results)

## raw
We debugged the scattering phase-shift extraction step by step (cheap decisive tests first).
(1) Confirmed via Julia that the recoupling 6j was called with DOUBLED angular-momentum args but
WignerSymbols.wigner6j wants PHYSICAL args → recoupling coeff² was 0.444 instead of 1.0. FIXED
(physical args + corrected phase (-1)^(λ+s₃+J₁₂+J)); identity test T·I·T† now = I for a single
deuteron component. (2) Found ³S₁ and ³D₁ deuteron components were treated as INDEPENDENT scattering
channels → recoupling double-counted (identity test gave diag 2.0 for the 4-channel case, 1.0 for
³S₁-only). FIXED by keeping only deuteron components the bound state actually occupies (MT: pure ³S₁).
(3) Operator energy: changed from E_cm to total 3-body energy E_total=E_cm+E_d (deuteron binding must
enter the energy balance). (4) Amplitude was using UNSCALED V while the solve used scaled V → made
amplitude use the complex-scaled V. (5) Tried the CS c-product (no conjugation, transpose) in the
amplitude inner product → η got worse (1.04→1.37); REVERTED to conjugated dot (matches bound-state
convention ψ'·B·ψ which uses conjugate). (6) Established the matrix element ⟨φ|V·Rxy|ψ⟩ should NOT
carry the B metric (V is the operator matrix; bound state computes ψ'·V_UIX·ψ without B), only state
OVERLAPS carry B — so current bare transpose/dot product is structurally consistent with bound state.
(7) Channel-convergence scan at reduced mesh: (lmax,λmax,j2b)=(2,2,1)→10 ch, (3,4,2)→18 ch,
(4,6,2)→18 ch ALL gave IDENTICAL δ=74.199°, η=1.0132 → extra channels do not couple to / change the
elastic doublet observable. (8) Checked solve convergence: relative residual ||Aψ-b||/||b|| = 2.1e-5
(the absolute 2e-2 was just because ||b||~1000) → the linear solve IS converged, not the problem.

## normalized
- T1. recoupling 6j passed doubled-j to WignerSymbols (wants physical j) → non-unitary. [status: succeeded-but-insufficient — fixed, verified unitary, but δ/η still off]
- T2. ³S₁/³D₁ deuteron components double-counted as independent channels. [status: succeeded-but-insufficient — fixed, identity test now I]
- T3. operator energy changed E_cm → E_total = E_cm + E_d. [status: partial — δ moved -33°→-51°→ (with later fixes) ~79°; effect entangled]
- T4. amplitude switched to complex-scaled V (was unscaled). [status: partial]
- T5. CS c-product (no-conjugate) inner product in amplitude. [status: failed — η worsened 1.04→1.37, reverted to conjugate dot]
- T6. confirmed amplitude matrix element needs no B metric (consistent w/ bound-state ψ'·V·ψ). [status: succeeded-but-insufficient — convention confirmed, didn't fix η]
- T7. channel-convergence scan 10→18 channels at reduced mesh. [status: failed — δ,η IDENTICAL (74.199, 1.0132); extra channels decouple from elastic observable]
- T8. checked linear-solve convergence. [status: succeeded-but-insufficient — rel_res=2.1e-5, solve is converged; rules out solver as the cause]

# Budget / constraints

- Each full-mesh scattering run ≈ 220 s; reduced-mesh ≈ tens of seconds. Iteration is the bottleneck.
- The scattering operator A is a DENSE matrix of dimension (nch·nx·ny); memory scales as (nch·nx·ny)².
  Full mesh nch=10,nx=30,ny=70 → dim 21000 → A ~7 GB, ~95 GB allocations. Cannot crank channels at
  full mesh (memory). Heavy compute is local only here; cluster available but not yet used for this.
- Must stay within the existing R-space Faddeev + complex-scaling framework (this IS the method under
  test; not switching to CDCC/AGS). Bound-state code must not be broken (it is benchmark-validated).
- Reference values are firm (Lazauskas PRC 84 Table III: doublet 105.49°/0.4649, quartet 68.95°/0.9782
  at E_lab=14.1; also 42 MeV values). Lazauskas himself is the external cross-validation contact.
- Lazauskas warned: CS at low energy (1 MeV) needs excessively large/dense grids because the CS
  wavefunction decays as exp(-q·sinθ·y) with tiny q; advised benchmarking at higher E (14.1, 42 MeV).
  CS angle hard limit (his Eq.18): θ_max=14.2° at 14.1 MeV.
