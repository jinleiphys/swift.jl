"""
    coulcc.jl - Julia wrapper for COULCC Fortran library

This module provides Julia interface to the COULCC (Complex Coulomb wavefunctions)
Fortran library by I.J. Thompson and A.R. Barnett (CPC 36, 1985, 363-372).

COULCC computes complex Coulomb wavefunctions F, G, F', G' and phase shifts
using Steed's method, for complex energy solutions to:
- Coulomb Schrodinger equation
- Klein-Gordon equation
- Spherical and cylindrical Bessel equations

Author: Auto-generated Julia wrapper
"""

module CoulCC

using Libdl

# Library handle (will be initialized on first use)
const libcoulcc = Ref{Ptr{Nothing}}(C_NULL)

"""
    load_library(lib_path::String="")

Load the COULCC Fortran shared library.
If lib_path is empty, searches in the current directory.
"""
function load_library(lib_path::String="")
    if libcoulcc[] != C_NULL
        return libcoulcc[]
    end

    # Determine library extension based on platform
    if Sys.isapple()
        lib_ext = "dylib"
    elseif Sys.islinux()
        lib_ext = "so"
    elseif Sys.iswindows()
        lib_ext = "dll"
    else
        error("Unsupported platform: $(Sys.KERNEL)")
    end

    # Construct library path
    if isempty(lib_path)
        lib_path = joinpath(@__DIR__, "libcoulcc.$lib_ext")
    end

    # Load library
    if !isfile(lib_path)
        error("COULCC library not found at: $lib_path\nPlease compile with: make -f Makefile_coulcc")
    end

    try
        libcoulcc[] = dlopen(lib_path)
        println("Successfully loaded COULCC library from: $lib_path")
    catch e
        error("Failed to load COULCC library: $e")
    end

    return libcoulcc[]
end

"""
    find_symbol(lib::Ptr, name::String)

Find a symbol in the loaded library, trying various name mangling patterns.
"""
function find_symbol(lib::Ptr, name::String)
    # Try different name mangling patterns
    patterns = [
        lowercase(name) * "_",  # gfortran standard
        lowercase(name),         # no underscore
        uppercase(name) * "_",  # uppercase with underscore
        uppercase(name),         # uppercase no underscore
        name                     # original name
    ]

    for pattern in patterns
        try
            sym = dlsym(lib, pattern)
            return sym
        catch
            continue
        end
    end

    error("Symbol '$name' not found in library with any mangling pattern")
end

"""
    coulcc(xx::ComplexF64, eta::ComplexF64, lmin::Number;
           lmax::Union{Number,Nothing}=nothing, nl::Union{Int,Nothing}=nothing,
           mode::Int=1, kfn::Int=0, ifail::Int=0)

Compute complex Coulomb wavefunctions using the COULCC Fortran library.

# Arguments
- `xx::ComplexF64`: Complex argument (energy-dependent)
- `eta::ComplexF64`: Coulomb parameter
- `lmin::Number`: Minimum lambda (angular momentum) value λ_min

# Keyword Arguments (specify either `lmax` or `nl`, not both)
- `lmax::Number`: Maximum lambda value λ_max (computes λ from lmin to lmax inclusive)
- `nl::Int`: Number of lambda values to compute (computes λ = lmin, lmin+1, ..., lmin+nl-1)
  - If both `lmax` and `nl` are provided, `lmax` takes precedence
  - If neither is provided, defaults to nl=1 (compute only at lmin)

# Keyword Arguments
- `mode::Int=1`: Calculation mode
  - `abs(mode) = 1`: Get F, G, F', G'
  - `abs(mode) = 2`: Get F, G only
  - `abs(mode) = 3`: Get F, F' only
  - `abs(mode) = 4`: Get F only
  - `abs(mode) = 11`: Get F, H+, F', H+' (where H+ = G + iF)
  - `abs(mode) = 12`: Get F, H+
  - `abs(mode) = 21`: Get F, H-, F', H-' (where H- = G - iF)
  - `abs(mode) = 22`: Get F, H-
  - If `mode < 0`: Results are scaled by exp(-|scale|) for large |xx|
- `kfn::Int=0`: Function type
  - `0`: Complex Coulomb functions F & G
  - `-1`: Complex Coulomb functions without phase shifts
  - `1`: Spherical Bessel functions j & y
  - `2`: Cylindrical Bessel functions J & Y
  - `3`: Modified cylindrical Bessel functions I & K
- `ifail::Int=0`: Error printing flag (0 = no printing, ≠0 = print errors)

# Returns
- `fc::Vector{ComplexF64}`: Regular solution F (length nl)
- `gc::Vector{ComplexF64}`: Irregular solution G (length nl)
- `fcp::Vector{ComplexF64}`: Derivative F' (length nl)
- `gcp::Vector{ComplexF64}`: Derivative G' (length nl)
- `sig::Vector{ComplexF64}`: Coulomb phase shifts (length nl, only for kfn=0)
- `ifail::Int`: Error flag
  - `= -2`: Argument out of range
  - `= -1`: Continued fraction failed or arithmetic check failed
  - `= 0`: All calculations satisfactory
  - `≥ 0`: Results available for orders up to nl-ifail
  - `= -3`: Values at zlmin not found (overflow/underflow)
  - `= -4`: Roundoff errors make results meaningless

# Examples
```julia
using CoulCC

# Load library
CoulCC.load_library()

# Example 1: Using lmax (recommended - most intuitive)
xx = 10.0 + 0.0im
eta = 1.0 + 0.0im
fc, gc, fcp, gcp, sig, ifail = coulcc(xx, eta, 0, lmax=4)  # λ from 0 to 4

# Example 2: Using nl (number of values)
fc, gc, fcp, gcp, sig, ifail = coulcc(xx, eta, 0, nl=5)  # λ = 0,1,2,3,4 (same as lmax=4)

# Example 3: Single value (default)
fc, gc, fcp, gcp, sig, ifail = coulcc(xx, eta, 0)  # Only λ=0

if ifail == 0
    println("Calculation successful!")
    println("F(λ=0) = ", fc[1])
    println("G(λ=0) = ", gc[1])
else
    println("Calculation failed with ifail = ", ifail)
end
```

# References
- I.J. Thompson and A.R. Barnett, CPC 36 (1985) 363-372
- Original COULFG algorithm: CPC 27 (1982) 147-166
"""
function coulcc(xx::ComplexF64, eta::ComplexF64, lmin::Number;
                lmax::Union{Number,Nothing}=nothing, nl::Union{Int,Nothing}=nothing,
                mode::Int=1, kfn::Int=0, ifail::Int=0)

    # Convert lmin to ComplexF64
    zlmin = ComplexF64(lmin)

    # Determine nl from either lmax or nl
    if lmax !== nothing
        # User specified lmax: compute nl = lmax - lmin + 1
        nl_computed = Int(round(real(lmax - lmin))) + 1
        if nl_computed < 1
            error("lmax must be >= lmin (got lmin=$lmin, lmax=$lmax)")
        end
    elseif nl !== nothing
        # User specified nl directly
        nl_computed = nl
        if nl_computed < 1
            error("nl must be >= 1 (got nl=$nl)")
        end
    else
        # Default: compute only single value at lmin
        nl_computed = 1
    end

    # Ensure library is loaded
    if libcoulcc[] == C_NULL
        load_library()
    end

    # Allocate output arrays
    fc = zeros(ComplexF64, nl_computed)
    gc = zeros(ComplexF64, nl_computed)
    fcp = zeros(ComplexF64, nl_computed)
    gcp = zeros(ComplexF64, nl_computed)
    sig = zeros(ComplexF64, nl_computed)

    # Create mutable references for input/output parameters
    xx_ref = Ref(xx)
    eta_ref = Ref(eta)
    zlmin_ref = Ref(zlmin)
    nl_ref = Ref(Int32(nl_computed))
    mode_ref = Ref(Int32(mode))
    kfn_ref = Ref(Int32(kfn))
    ifail_ref = Ref(Int32(ifail))

    # Find the coulcc subroutine symbol
    coulcc_sym = find_symbol(libcoulcc[], "coulcc")

    # Call Fortran subroutine
    # SUBROUTINE COULCC(XX,ETA1,ZLMIN,NL, FC,GC,FCP,GCP, SIG, MODE1,KFN,IFAIL)
    ccall(coulcc_sym, Cvoid,
          (Ref{ComplexF64}, Ref{ComplexF64}, Ref{ComplexF64}, Ref{Int32},
           Ref{ComplexF64}, Ref{ComplexF64}, Ref{ComplexF64}, Ref{ComplexF64}, Ref{ComplexF64},
           Ref{Int32}, Ref{Int32}, Ref{Int32}),
          xx_ref, eta_ref, zlmin_ref, nl_ref,
          fc, gc, fcp, gcp, sig,
          mode_ref, kfn_ref, ifail_ref)

    return fc, gc, fcp, gcp, sig, ifail_ref[]
end

"""
    coulcc_real(x::Float64, eta::Float64, lmin::Number; kwargs...)

Convenience wrapper for real arguments (converts to complex internally).

# Example
```julia
# Using lmax
fc, gc, fcp, gcp, sig, ifail = coulcc_real(10.0, 1.0, 0, lmax=2)

# Using nl
fc, gc, fcp, gcp, sig, ifail = coulcc_real(10.0, 1.0, 0, nl=3)
```
"""
function coulcc_real(x::Float64, eta::Float64, lmin::Number; kwargs...)
    return coulcc(ComplexF64(x), ComplexF64(eta), lmin; kwargs...)
end

"""
    unload_library()

Unload the COULCC library and free resources.
"""
function unload_library()
    if libcoulcc[] != C_NULL
        dlclose(libcoulcc[])
        libcoulcc[] = C_NULL
        println("COULCC library unloaded")
    end
end

# Export public interface
export coulcc, coulcc_real, load_library, unload_library

end # module CoulCC
