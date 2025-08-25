# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Julia-based nuclear physics framework implementing the Faddeev method for three-body quantum mechanical bound state calculations. The codebase specializes in solving nuclear three-body problems (like 3H tritium) using sophisticated numerical techniques including Laguerre basis functions, multi-channel coupling, and nuclear potential models.

## Development Commands

### Julia Environment Setup
Run these commands to install required Julia packages:
```bash
cd swift
julia setup.jl
```

### Building Fortran Libraries
To build the nuclear potential libraries:
```bash
cd NNpot
make clean && make
```

### Running Calculations
- **Interactive development**: Use Jupyter notebooks in any subdirectory (*.ipynb files)
- **Script execution**: Run Julia files directly with `julia filename.jl`
- **Testing modules**: Run test files like `julia test.jl` in respective directories

## Core Architecture

### Module Structure
The codebase is organized into three main module directories:

1. **NNpot/**: Nuclear potential interface
   - Fortran libraries (AV18, AV14, Nijmegen) with Julia wrappers
   - `nuclear_potentials.jl`: Interface to compiled Fortran potentials
   - Dynamic library compilation via makefiles

2. **general_modules/**: Foundation components
   - `channels.jl`: Three-body channel coupling calculations with angular momentum algebra
   - `mesh.jl`: Laguerre-based numerical mesh generation for hyperspherical coordinates

3. **swift/**: Core Faddeev implementation
   - `matrices.jl`: Matrix elements for kinetic energy (T), potential (V), and coordinate transformations (Rxy)
   - `threebodybound.jl`: Eigenvalue solver for bound state energies
   - `twobody.jl`: Two-body reference calculations (deuteron)
   - `laguerre.jl`: Basis function implementations
   - `Gcoefficient.jl`: Angular momentum coupling coefficients

### Key Physics Concepts
- **Faddeev equations**: Three-body quantum mechanics using coordinate transformations
- **Hyperspherical coordinates**: (x,y) representing relative distances in three-body system
- **Channel coupling**: Multi-channel approach with different angular momentum states
- **Nuclear potentials**: Realistic NN interactions (AV18, AV14, Nijmegen, Malfliet-Tjon)

### Computational Workflow
1. **Channel generation**: `α3b()` creates allowed quantum states based on conservation laws
2. **Mesh initialization**: `initialmesh()` sets up hyperspherical coordinate grids
3. **Matrix construction**: Build Hamiltonian H = T + V + V*Rxy using Faddeev rearrangement
4. **Eigenvalue solution**: Find bound states with energies below two-body threshold

### Data Flow
- Input: Nuclear system parameters (J, T, parity, particle spins/isospins)
- Processing: Channel coupling → Matrix elements → Eigenvalue problem
- Output: Binding energies and three-body wave functions

## Important Implementation Details

### Fortran Integration
- Nuclear potentials are implemented in Fortran for performance
- Julia calls Fortran libraries via `ccall` interface
- Platform-specific dynamic library handling (`.dylib` on macOS, `.so` on Linux)

### Numerical Methods
- Laguerre polynomials for radial basis functions with regularization
- Gauss-Legendre quadrature for angular integrations
- Complex eigenvalue decomposition for bound state analysis

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

### Modifying Calculations
- Channel configurations: Edit parameters in notebook initialization cells
- Mesh parameters: Adjust `nx`, `ny`, `xmax`, `ymax`, `alpha` for convergence
- Potential models: Change `potname` variable to switch between models

### Debugging
- Use `debug_library_symbols()` for Fortran library symbol issues
- Check channel generation output for quantum number validation
- Monitor eigenvalue convergence and matrix conditioning