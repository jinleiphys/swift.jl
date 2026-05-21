# Swift.jl: Three-Body Nuclear Physics Framework
## Faddeev Method for Bound States and Scattering

---

# Slide 1: Project Overview

## Swift.jl - Julia-based Nuclear Three-Body Framework

### Core Capabilities
- **Bound State Calculations**: Solve three-nucleon bound states (³H, ³He)
- **Scattering Calculations**: Nucleon-deuteron elastic scattering
- **Realistic Potentials**: AV18, AV14, Nijmegen + Urbana IX (3-body force)

### Theoretical Foundation
```
┌─────────────────────────────────────────────────────────────┐
│           Faddeev Equations for Three-Body System           │
│                                                             │
│                 Ψ = ψ₁ + ψ₂ + ψ₃                           │
│                                                             │
│     (E - H₀ - V₁)ψ₁ = V₁(ψ₂ + ψ₃)   [cyclic permutations] │
└─────────────────────────────────────────────────────────────┘
```

### Key Features
| Feature | Description |
|---------|-------------|
| Basis | Laguerre polynomials in hyperspherical coordinates (x, y) |
| Coupling | Multi-channel angular momentum coupling |
| Solvers | Direct eigenvalue + Malfiet-Tjon iteration |
| Performance | Fortran potentials + Julia parallelization |

---

# Slide 2: Bound State Calculations

## Hamiltonian Construction
```
     H = T + V + V·Rxy    (Two-body forces)
     H = T + V + V·Rxy + X₁₂   (+ Urbana IX three-body force)
```

## Two Solution Methods

### Method 1: Direct Diagonalization
```julia
# Generalized eigenvalue problem
eigenvalues, eigenvectors = eigen(H, B)
# Finds ALL bound states below threshold
```

### Method 2: Malfiet-Tjon Iteration
```julia
# Reformulation: λ(E)[c] = [E·B - T - V]⁻¹ · V·Rxy · [c]
# Secant method: iterate until λ(E) → 1
```

## Results: ³H (Tritium) Ground State

| Method | Binding Energy | Status |
|--------|----------------|--------|
| Direct | -7.624 MeV | ✓ |
| Malfiet-Tjon | -7.624 MeV | ✓ |
| Experiment | -8.482 MeV | -- |

### Channel Probability Distribution
- ³S₁: ~91% (S-wave dominant)
- ³D₁: ~8% (D-wave admixture)
- Higher partial waves: ~1%

---

# Slide 3: Scattering Calculations

## Inhomogeneous Faddeev Equation
```
┌─────────────────────────────────────────────────────────────┐
│       [E·B - T - V·(I + Rxy)] · c = 2·V·Rxy₃₁·φ            │
│                                                             │
│   A · c = b                                                 │
│   ↑       ↑                                                 │
│   LHS     Source term (deuteron + nucleon initial state)   │
└─────────────────────────────────────────────────────────────┘
```

## Computational Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Initial State│     │ Solve Linear │     │   Extract    │
│  φ = φ_d·F_λ │ ──→ │   System     │ ──→ │  Scattering  │
│              │     │   A·c = b    │     │  Amplitude   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  Deuteron ³S₁+³D₁    LU or GMRES         f_{α₀,α₀'}(k)
  + Coulomb F_λ(ky)   with M⁻¹ precond.   Phase shifts δ
```

## Phase Shift Analysis (Blatt-Biedenharn)

### Collision Matrix
```
U^{α₀,α₀'}(k) = 2ik·f^{α₀,α₀'}(k) + δ_{α₀,α₀'}
```

### Eigenphase Shifts
```
λₖ = exp(2iδₖ)  →  δₖ = ½ arg(λₖ)
```

### Mixing Parameters (for coupled channels)
| J^π | Parameters | Physical Meaning |
|-----|------------|------------------|
| 1/2⁺ | η | Orbital-spin mixing |
| 1/2⁻ | ε | Spin mixing |
| 3/2⁺ | ε, ζ, η | 3×3 coupled channels |

## Key Physics Outputs
- **Elastic cross sections**: σ(θ) from f-matrix
- **Polarization observables**: Analyzing powers, spin correlations
- **Resonance structures**: Complex scaling for continuum states

---

# Summary

## Project Architecture

```
swift.jl/
├── NNpot/          # Fortran NN potentials (AV18, AV14, Nijmegen)
├── 3Npot/          # Three-body forces (Urbana IX)
├── general_modules/
│   ├── channels.jl # Angular momentum coupling
│   └── mesh.jl     # Laguerre basis & quadrature
└── swift/
    ├── threebodybound.jl  # Bound state solver
    ├── MalflietTjon.jl    # Iterative solver
    ├── scattering.jl      # Scattering equations
    └── matrices*.jl       # Matrix elements (T, V, Rxy)
```

## Unified Framework

| Calculation | Equation | Method |
|-------------|----------|--------|
| Bound States | H·ψ = E·B·ψ | Direct eigen / Malfiet-Tjon |
| Scattering | A·c = b | LU / Preconditioned GMRES |

## Future Directions
- Four-body extension (⁴He)
- Breakup channels (N + N + N)
- Electroweak reactions
