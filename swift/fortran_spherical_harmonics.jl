module FortranSphericalHarmonics
# Translation of Fortran spherical harmonics code to Julia

export computeYlm_fortran

"""
    generate_log_factorials(n)

Generate factorial table for log(i!).
Returns array where dlfac[i+1] = log(i!)
"""
function generate_log_factorials(n::Int)
    dlfac = zeros(Float64, n+1)
    dlfac[1] = 0.0  # log(0!) = 0
    
    # Calculate log factorials: ln(i!)
    for i in 1:n
        dlfac[i+1] = dlfac[i] + log(Float64(i))
    end
    
    return dlfac
end

"""
    flog(i, dlfac)

Return log factorial of (i-1) using precomputed table.
"""
function flog(i::Int, dlfac::Vector{Float64})
    if i == 0
        return 0.0
    else
        return dlfac[i]  # log((i-1)!)
    end
end

"""
    ylmc(l, m, dlfac)

Calculate spherical harmonic normalization coefficient.
Translated from Fortran function YLMC.
"""
function ylmc(l::Int, m::Int, dlfac::Vector{Float64})
    phase(i) = (-1)^i
    π = pi
    
    ma = abs(m)
    
    # Correct factorial calculation: log((l-|m|)!) - log((l+|m|)!)
    r = flog(l - ma + 1, dlfac) - flog(l + ma + 1, dlfac)
    r = sqrt((2*l + 1)/(4*π) * exp(r)) * phase(m)
    
    if m < 0
        r = r * phase(ma)
    end
    
    return r
end

"""
    plm(x, n, m, na)

Calculate associated Legendre polynomials P_l^m(x).
Translated from Fortran subroutine PLM.
"""
function plm(x::Float64, n::Int, m::Int, na::Int)
    # Clamp x to valid range
    x = clamp(x, -1.0, 1.0)
    
    if n == 0
        return ones(Float64, na, m+1)
    end
    
    n1 = n + 1
    m1 = m + 1
    
    # Initialize array
    pl = zeros(Float64, na, m1)
    
    # Base cases
    pl[1, 1] = 1.0
    pl[2, 1] = x
    
    sx = sqrt(1.0 - x*x)
    if m1 >= 2 && n1 >= 2
        pl[2, 2] = sx
    end
    
    fact = 1.0
    pmm = 1.0
    
    # Calculate diagonal elements P_m^m
    for j in 2:min(m1, n1)
        mm = j - 1
        pmm = pmm * fact * sx
        fact = fact + 2.0
        if j <= n1 && j <= m1
            pl[j, j] = pmm
        end
        
        if j + 1 <= n1 && j <= m1
            pl[j+1, j] = x * (2*mm + 1.0) * pl[j, j]
        end
    end
    
    # Calculate off-diagonal elements using recurrence relation
    for j in 1:m1
        mm = j - 1
        for i in (j+2):n1
            ll = i - 1
            if i <= n1 && j <= m1 && i-1 <= n1 && i-2 <= n1
                pl[i, j] = ((2.0*ll - 1.0)*x*pl[i-1, j] - (ll + mm - 1.0)*pl[i-2, j]) / (ll - mm)
            end
        end
    end
    
    return pl
end



"""
    computeYlm_fortran(θ, φ, lmax)

Calculate spherical harmonics array matching computeYlm interface.
Returns array of Y_l^m values with same indexing as SphericalHarmonics.jl
but using Fortran-translated implementation with Condon-Shortley phase.
"""
function computeYlm_fortran(θ::Real, φ::Real, lmax::Int)
    # Generate factorial table (call factorialgen inside function)
    dlfac = generate_log_factorials(2*lmax)
    
    # Initialize result array - size is (lmax+1)²
    array_size = (lmax + 1)^2
    ylm_array = zeros(ComplexF64, array_size)
    
    cth = cos(θ)
    
    # Compute associated Legendre polynomials once for all l,m up to lmax
    pl = plm(cth, lmax, lmax, lmax+1)
    
    # Calculate spherical harmonics for all l,m up to lmax
    index = 1
    for l in 0:lmax
        for m in -l:l
            # Calculate the spherical harmonic using our Fortran translation
            ylmc_coeff = ylmc(l, m, dlfac)
            real_part = ylmc_coeff * pl[l+1, abs(m)+1]
            
            # Add complex phase factor: e^(i*m*φ)
            phase_factor = exp(im * m * φ)
            
            # Combine real spherical harmonic with phase
            ylm_array[index] = real_part * phase_factor
            
            index += 1
        end
    end
    
    return ylm_array
end

end # module FortranSphericalHarmonics