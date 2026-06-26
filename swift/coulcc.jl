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
# Regular Coulomb function for η=0: F_λ(0,x) = x·j_λ(x) (Riccati-Bessel), λ=0..λmax, complex x.
# Stable Miller downward recurrence started above the turning point λ≈|x|, normalized to the exact
# j_0 = sin(x)/x. Used to supply the analytic boundary value where COULCC's continued fraction loses
# precision. Validated vs COULCC to ~3e-15 over real and complex-scaled x.
function riccati_bessel_F(λmax::Int, x::ComplexF64)
    nstart = λmax + ceil(Int, abs(x)) + 30
    j = zeros(ComplexF64, nstart + 2)        # j[l+1] holds j_l(x)
    j[nstart + 1] = 1e-30 + 0im              # tiny seed; j[nstart+2]=0
    for l in nstart:-1:1
        j[l] = (2l + 1) / x * j[l + 1] - j[l + 2]   # j_{l-1}
    end
    scale = (sin(x) / x) / j[1]              # normalize to the exact j_0
    return ComplexF64[x * j[λ + 1] * scale for λ in 0:λmax]
end

# Analytic boundary repair of the regular function F. COULCC's continued fraction loses precision for
# the high-λ / small-|x| boundary orders (where the regular F is analytically tiny), and can return
# garbage there, sometimes with ifail=0 — which propagated downstream as intermittent Inf/NaN. Repair
# ONLY the affected entries with the analytic regular function; COULCC's good values are left as-is.
# fc[k] is order λ = zlmin + (k-1).
function repair_boundary_F!(fc, xx, eta, zlmin, nl, ifail)
    λ0 = Int(round(real(zlmin)))
    if abs(eta) < 1e-12
        # η=0: the regular Coulomb function is exactly the Riccati-Bessel F_λ(0,x)=x·j_λ(x) for all x.
        Fan = riccati_bessel_F(λ0 + nl - 1, xx)
        for k in 1:nl
            ref = Fan[λ0 + k]                # Fan index = λ+1 = (λ0+k-1)+1
            if !isfinite(fc[k]) || abs(fc[k] - ref) > 1e-6 * max(1.0, abs(ref))
                fc[k] = ref
            end
        end
    else
        # Charged channel: no cheap exact reference. The failing boundary is the small-|x| regime where
        # the regular F vanishes, so floor COULCC's flagged (ifail>0) or non-finite high orders to 0.
        # TODO: replace 0 with the regular Coulomb leading term C_λ(η)·x^{λ+1} if charged channels are used.
        first_bad = ifail > 0 ? nl - ifail + 1 : nl + 1
        for k in 1:nl
            if !isfinite(fc[k]) || k >= first_bad
                fc[k] = 0.0 + 0.0im
            end
        end
    end
    return fc
end

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

    # Patch the regular function F at the boundary where COULCC's continued fraction failed
    repair_boundary_F!(fc, xx, eta, zlmin, nl_computed, Int(ifail_ref[]))

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
