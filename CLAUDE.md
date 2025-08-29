# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Julia-based nuclear physics framework implementing the Faddeev method for three-body quantum mechanical bound state calculations. The codebase specializes in solving nuclear three-body problems (like 3H tritium) using sophisticated numerical techniques including Laguerre basis functions, multi-channel coupling, and nuclear potential models.

## Development Commands

### Julia Environment Setup
Install required Julia packages before any development:
```bash
cd swift
julia setup.jl
```

### Building Fortran Libraries
The nuclear potential libraries must be compiled before use. The makefile automatically detects platform and uses appropriate flags:
```bash
cd NNpot
make clean && make
```
This creates `libpotentials.dylib` (macOS), `libpotentials.so` (Linux), or `libpotentials.dll` (Windows).

### Build System Details
- **Fortran compiler**: `gfortran` with `-O2 -fPIC -Wall -Wextra` optimization
- **Platform detection**: Automatic selection of shared library format
- **F77/F90 compatibility**: Separate compilation flags for legacy and modern Fortran code

### Running Calculations
- **Interactive development**: Use Jupyter notebooks in any subdirectory (*.ipynb files)
- **Memory-optimized runs**: Use `swift_3H_optimized.ipynb` for reduced memory calculations (~1-2 GB instead of 27 GB)
- **Script execution**: Run Julia files directly with `julia filename.jl`
- **Testing modules**: Run test files like `julia test.jl` in respective directories

### Testing
- **Quick module test**: `julia NNpot/test.jl` - basic nuclear potential interface validation
- **Comprehensive test**: `julia NNpot/test_comprehensive.jl` - full system validation with multiple potentials
- **Physics validation**: `julia NNpot/test_channel_physics.jl` - channel coupling and quantum number consistency
- **Specific debugging**: Various `debug_*.jl` and `simple_*test*.jl` files for targeted testing

### Development Workflow
1. **Library setup**: Build Fortran libraries first (`cd NNpot && make`)
2. **Package installation**: Install Julia dependencies (`cd swift && julia setup.jl`)
3. **Development**: Use Jupyter notebooks for interactive exploration and debugging
4. **Testing**: Run specific tests to validate changes before committing

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
   - `threebodybound.jl`: Direct eigenvalue solver for bound state energies
   - `MalflietTjon.jl`: Iterative Malfiet-Tjon eigenvalue solver with secant method convergence
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
4. **Eigenvalue solution**: Two approaches available:
   - **Direct method**: `ThreeBody_Bound()` solves generalized eigenvalue problem `eigen(H, B)`
   - **Iterative method**: `malfiet_tjon_solve()` uses secant iteration to find λ(E) = 1

### Data Flow
- Input: Nuclear system parameters (J, T, parity, particle spins/isospins)
- Processing: Channel coupling → Matrix elements → Eigenvalue problem
- Output: Binding energies and three-body wave functions

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
- **Convergence**: Secant method iteration until `|λ(E) - 1| < tolerance`

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

### Required Julia Packages
The project uses specific Julia packages that must be installed:
- **SphericalHarmonics**: For spherical harmonic calculations
- **WignerSymbols**: For angular momentum coupling coefficients
- **JSON**: For data serialization in notebooks
- **FastGaussQuadrature**: For numerical integration
- **Kronecker**: For tensor product operations
- **Revise**: For development workflow (hot reloading)

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