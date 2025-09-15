module Gcoefficient
using WignerSymbols
include("fortran_spherical_harmonics.jl")
using .FortranSphericalHarmonics
export computeGcoefficient


# Yλout = zeros(Float64,nθ,λmax^2+2*λmax+1,2)        # last dimension for the permutation operator 1 for P+; 2 for P-
# Ylin = zeros(Float64,nθ,ny,nx,lmax^2+2*lmax+1,2)   # last dimension for the permutation operator 1 for P+; 2 for P-
# Yλin = zeros(Float64,nθ,ny,nx,λmax^2+2*λmax+1,2)   # last dimension for the permutation operator 1 for P+; 2 for P-




 function computeGcoefficient(α, grid)

    λmax = maximum(α.λ)
    lmax = maximum(α.l)
    nθ = grid.nθ
    nx = grid.nx
    ny = grid.ny
    cosθi = grid.cosθi
    xi = grid.xi
    yi = grid.yi

    Yλout, Ylin, Yλin = initialY(λmax, lmax, nθ, nx, ny, cosθi, xi, yi)
    Gresult = Gαα(nθ, ny, nx, α, Yλout, Ylin, Yλin, grid)
    return Gresult
 end 

 function Gαα(nθ, ny, nx, α, Yλout, Ylin, Yλin, grid)
    s1 = α.s1
    s2 = α.s2
    s3 = α.s3
    t1 = α.t1
    t2 = α.t2
    t3 = α.t3
    Gαoutαin = zeros(Float64, nθ, ny, nx, α.nchmax, α.nchmax, 2)
    
    # Pre-compute invariant quantities outside loops
    hat_T12 = [hat(α.T12[i]) for i in 1:α.nchmax]
    hat_J12 = [hat(α.J12[i]) for i in 1:α.nchmax] 
    hat_J3 = [hat(α.J3[i]) for i in 1:α.nchmax]
    hat_s12 = [hat(α.s12[i]) for i in 1:α.nchmax]
    
    # Cache for expensive calculations - using Dict for flexibility with composite keys
    u9_cache = Dict{Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}, Float64}()
    wigner6j_cache = Dict{Tuple{Float64,Float64,Float64,Float64,Float64,Float64}, Float64}()
    Y4_cache = Dict{Tuple{Int,Int,Int}, Array{Float64,4}}()
    
    # Helper function for cached u9 calculation
    function cached_u9(args...)
        key = args
        if !haskey(u9_cache, key)
            u9_cache[key] = u9(args...)
        end
        return u9_cache[key]
    end
    
    # Helper function for cached wigner6j calculation  
    function cached_wigner6j(args...)
        key = args
        if !haskey(wigner6j_cache, key)
            wigner6j_cache[key] = wigner6j(args...)
        end
        return wigner6j_cache[key]
    end
    
    # Helper function for cached Y4 calculation
    function cached_Y4(LL, αout, αin)
        key = (LL, αout, αin)
        if !haskey(Y4_cache, key)
            Y4_cache[key] = YYcoupling(α, nθ, ny, nx, Ylin, Yλin, Yλout, LL, αout, αin)
        end
        return Y4_cache[key]
    end
    
    # Pre-compute valid channel combinations to avoid redundant checks
    valid_combinations = Vector{Tuple{Int,Int,Int,Int,Int,Float64,Float64,Float64,Int,Int}}()
    
    for perm_index in 1:2
        for αout in 1:α.nchmax
            for αin in 1:α.nchmax
                # Check if channels can couple - they must have the same total isospin T
                if α.T[αin] != α.T[αout]
                    continue
                end
                
                # Pre-compute permutation-specific values
                if perm_index == 1
                    phase = (-1)^round(Int, α.s12[αin] + 2*s1 + s2 + s3) * (-1)^round(Int, α.T12[αin] + 2*t1 + t2 + t3)
                    Cisospin = hat_T12[αin] * hat_T12[αout] * cached_wigner6j(t1,t2,α.T12[αout],t3,α.T[αout],α.T12[αin])
                    s_ref = s1  # Reference spin for coupling
                    nSmin = max(round(Int, 2*abs(α.s12[αin]-s1)), round(Int, 2*abs(α.s12[αout]-s3)))
                    nSmax = min(round(Int, 2*(α.s12[αin]+s1)), round(Int, 2*(α.s12[αout]+s3)))
                else
                    phase = (-1)^round(Int, α.s12[αin] + 2*s2 + s1 + s3) * (-1)^round(Int, α.T12[αin] + 2*t2 + t1 + t3)
                    Cisospin =  hat_T12[αin] * hat_T12[αout] * cached_wigner6j(t2,t1,α.T12[αout],t3,α.T[αout],α.T12[αin])
                    s_ref = s2  # Reference spin for coupling
                    nSmin = max(round(Int, 2*abs(α.s12[αin]-s2)), round(Int, 2*abs(α.s12[αout]-s3)))
                    nSmax = min(round(Int, 2*(α.s12[αin]+s2)), round(Int, 2*(α.s12[αout]+s3)))
                end
                
                # Skip if isospin coupling is zero
                if abs(Cisospin) < 1e-14
                    continue
                end
                
                # Pre-compute spin coupling coefficient
                Cspin = hat_J12[αin] * hat_J12[αout] * hat_J3[αin] * hat_J3[αout] * hat_s12[αin] * hat_s12[αout]
                
                LLmin = max(abs(α.l[αin]-α.λ[αin]), abs(α.l[αout]-α.λ[αout]))
                LLmax = min(α.l[αin]+α.λ[αin], α.l[αout]+α.λ[αout])
                
                # Store valid combination for later processing
                for LL in LLmin:LLmax
                    push!(valid_combinations, (perm_index, αout, αin, LL, round(Int,s_ref*2), phase, Cisospin, Cspin, nSmin, nSmax))
                end
            end
        end
    end
    
    # Optimized main computation loop - process by LL first for better Y4 cache usage
    for (perm_index, αout, αin, LL, s_ref_int, phase, Cisospin, Cspin, nSmin, nSmax) in valid_combinations
        s_ref = s_ref_int / 2.0
        
        # Get Y4 coupling (cached)
        Y4 = cached_Y4(LL, αout, αin)
        
        for nSS in nSmin:nSmax
            SS = nSS / 2.0
            
            # Triangle inequality check for |LL - SS| ≤ J ≤ LL + SS
            if (round(Int, 2*abs(LL-SS)) > round(Int, 2*α.J) || round(Int, 2*(LL+SS)) < round(Int, 2*α.J))
                continue
            end
            
            # Pre-compute common u9 factor (cached)
            u9_out = cached_u9(float(α.l[αout]), α.s12[αout], α.J12[αout], 
                              float(α.λ[αout]), α.s3, α.J3[αout], 
                              float(LL), SS, α.J)
            
            u9_in = cached_u9(float(α.l[αin]), α.s12[αin], α.J12[αin], 
                             float(α.λ[αin]), s_ref, α.J3[αin], 
                             float(LL), SS, α.J)
            
            # Skip if either u9 coefficient is zero
            if abs(u9_out) < 1e-14 || abs(u9_in) < 1e-14
                continue
            end
            
            f0 = (2*SS + 1.0) * u9_out * u9_in
            
            # Compute spin recoupling coefficient (cached)
            if perm_index == 1
                f1 = cached_wigner6j(s1, s2, α.s12[αout], s3, SS, α.s12[αin])
            else
                f1 = cached_wigner6j(s2, s1, α.s12[αout], s3, SS, α.s12[αin])
            end
            
            # Skip if wigner6j coefficient is zero  
            if abs(f1) < 1e-14
                continue
            end
            
            # Final coefficient
            coeff = phase * Cisospin * Cspin * f0 * f1
            
            # Optimized array accumulation - use views to avoid temporary arrays
            @views Gαoutαin[:,:,:,αout, αin, perm_index] .+= Y4[:, :, :, perm_index] .* coeff
        end
    end

    return Gαoutαin
 end


 function YYcoupling(α, nθ, ny, nx, Ylin, Yλin, Yλout, LL, αout, αin)

    
    # Initialize output array for specific channel pair
    Y4 = zeros(Float64, nθ, ny, nx, 2)
    
    # Calculate Ylαout coefficient for Y_l^0 (m=0 component)
    Ylαout = sqrt( (2.0 * α.l[αout] + 1.0) / (4.0 * π) )
    
    # The ML loop range is constrained by both LL and λ[αout] due to the Clebsch-Gordan coefficient
    minl = min(LL, α.λ[αout])
    
    # Pre-compute all valid quantum number combinations and their coupling factors
    # This moves invariant calculations outside the spatial loops
    valid_combinations = Vector{Tuple{Int, Int, Int, Float64, Vector{Float64}}}()
    
    for ML in -minl:minl
        # Calculate nchλout index with bounds check
        nchλout = round(Int, α.λ[αout]^2 + α.λ[αout] + ML + 1)
        
        # Calculate Clebsch-Gordan coefficient for output
        # Note: For z-axis (θ=0), ml=0 for the l component
        CGout = clebschgordan(α.l[αout], 0, α.λ[αout], ML, LL, ML)
        
        # Skip if CGout is zero to avoid unnecessary computation
        if abs(CGout) < 1e-14
            continue
        end
        
        # Pre-calculate Yout values for this ML - move outside spatial loops
        Yout = Ylαout .* Yλout[:, nchλout]
        
        # Loop over ml and mλ values
        for ml in -α.l[αin]:α.l[αin]
            for mλ in -α.λ[αin]:α.λ[αin]
                # Skip if quantum numbers don't satisfy coupling condition
                if mλ + ml != ML
                    continue
                end
                
                # Calculate Clebsch-Gordan coefficient for input
                CGin = clebschgordan(α.l[αin], ml, α.λ[αin], mλ, LL, ML)
                
                # Skip if CGin is zero to avoid unnecessary computation
                if abs(CGin) < 1e-14
                    continue
                end
                
                # Calculate indices for input channel components with bounds check
                nchlin = round(Int, α.l[αin]^2 + α.l[αin] + ml + 1)
                nchλin = round(Int, α.λ[αin]^2 + α.λ[αin] + mλ + 1)
                
                # Calculate coupling factor once - move outside spatial loops
                coupling_factor = CGin * CGout
                
                # Store valid combination with pre-computed values
                push!(valid_combinations, (nchlin, nchλin, nchλout, coupling_factor, Yout))
            end
        end
    end
    
    # Optimized loop structure: iterate over spatial points first, then quantum numbers
    # This improves cache locality for the Y arrays and reduces redundant calculations
    for (nchlin, nchλin, nchλout, coupling_factor, Yout) in valid_combinations
        # Restructured loops for better cache performance: 
        # Access patterns are more sequential in memory
        for perm_idx in 1:2
            for ix in 1:nx
                for iy in 1:ny
                    # Pre-load input Y slices for this (ix, iy) - improves cache usage
                    @inbounds for iθ in 1:nθ
                        Yin = Ylin[iθ, iy, ix, nchlin, perm_idx] * Yλin[iθ, iy, ix, nchλin, perm_idx]
                        Y4_contribution = coupling_factor * Yout[iθ] * Yin
                        Y4[iθ, iy, ix, perm_idx] += Y4_contribution
                    end
                end
            end
        end
    end

    Y4 = Y4 .* 8.0 .* π^2
    
    return Y4
end

function initialY(λmax, lmax, nθ, nx, ny, cosθi, xi, yi)
    
    Yλout = zeros(Float64, nθ, λmax^2 + 2*λmax + 1)
    Ylin = zeros(Float64, nθ, ny, nx, lmax^2 + 2*lmax + 1, 2)    
    Yλin = zeros(Float64, nθ, ny, nx, λmax^2 + 2*λmax + 1, 2)

    # Set x_3 as z-direction
    for i in 1:nθ
        Yλ = computeYlm_fortran(acos(cosθi[i]), 0.0, λmax)
        for λ in 0:λmax
           for m in -λ:λ
              nch = λ^2 + λ + m + 1  # Standard indexing for Y_λ^m
              nch_conj = λ^2 + λ + (-m) + 1  # Index for Y_λ^{-m}
              Yλout[i, nch] = real(Yλ[nch_conj]) * (-1)^m  # Y_λ^m* = (-1)^m* Y_λ^{-m}
            end
        end
    end

  for perm_index in 1:2 
    # Now compute the spherical harmonics for incoming channels
       if perm_index == 1  # compute Gα3α1: x1 = -1/2*x3 + 1*y3, y1 = -3/4*x3 - 1/2*y3
           a = -0.5
           b = 1.0
           c = -0.75
           d = -0.5
        elseif perm_index == 2 # compute Gα3α2: x2 = -1/2*x3 - 1*y3, y2 = 3/4*x3 - 1/2*y3
           a = -0.5
           b = -1.0
           c = 0.75
           d = -0.5
        end
    
    # Set the ϕ angle for the spherical harmonics
    ϕx = b < 0 ? π : 0.0
    ϕy = d < 0 ? π : 0.0

    for ix in 1:nx
        for iy in 1:ny
            for iθ in 1:nθ
                # Compute transformed coordinates with safety checks
                xin_squared = a^2*xi[ix]^2 + b^2*yi[iy]^2 + 2*a*b*xi[ix]*yi[iy]*cosθi[iθ]
                yin_squared = c^2*xi[ix]^2 + d^2*yi[iy]^2 + 2*c*d*xi[ix]*yi[iy]*cosθi[iθ]
                
                
                xin = sqrt(max(xin_squared, 1e-15))  # Avoid zero
                yin = sqrt(max(yin_squared, 1e-15))  # Avoid zero
                
                xzin = a*xi[ix] + b*yi[iy]*cosθi[iθ]
                yzin = c*xi[ix] + d*yi[iy]*cosθi[iθ]
                
                # Ensure argument to acos is in valid range [-1, 1] and avoid division by zero
                θx = acos(clamp(xzin/xin, -1.0, 1.0))
                θy = acos(clamp(yzin/yin, -1.0, 1.0))

                
                Yl = computeYlm_fortran(θx, ϕx, lmax)
                Yλ = computeYlm_fortran(θy, ϕy, λmax)
                # Compute spherical harmonics for each (l,m) combination
                for l in 0:lmax
                    for m in -l:l
                        nch = l^2 + l + m + 1
                        Ylin[iθ, iy, ix, nch, perm_index] = real(Yl[nch])  # Extract real part
                    end
                end
                
                # Compute spherical harmonics for each (λ,m) combination
                for λ in 0:λmax
                    for m in -λ:λ
                        nch = λ^2 + λ + m + 1
                        Yλin[iθ, iy, ix, nch, perm_index] = real(Yλ[nch])  # Extract real part
                    end
                end
            end
        end
    end
  end 
    
    return Yλout, Ylin, Yλin
end  # function initialY




 function hat(x)
    y = sqrt(2.0*x+1.0)
    return y
 end 



"""
Nine-j symbol calculation. Definition as in Brink and Satchler.
| a b c |
| d e f |
| g h i |

Uses the racahW function from WignerSymbols.jl for efficient computation.
"""
function u9(ra::Float64, rb::Float64, rc::Float64,
            rd::Float64, re::Float64, rf::Float64,
            rg::Float64, rh::Float64, ri::Float64)::Float64

    # 0) Basic validity: non-negative and half-integer/integer (within tol)
    J = (ra,rb,rc,rd,re,rf,rg,rh,ri)
    if any(x < 0 for x in J); return 0.0; end
    tol = 1e-9
    if any(abs(x*2 - round(x*2)) > 1e-8 for x in J); return 0.0; end

    # 1) Quick triangle screens (not strictly required but cheap)
    triangles = [
        (ra, rb, rc), (rd, re, rf), (rg, rh, ri),  # rows
        (ra, rd, rg), (rb, re, rh), (rc, rf, ri)   # columns
    ]
    for (j1,j2,j3) in triangles
        if (abs(j1-j2) > j3 + tol) || (j1 + j2 < j3 - tol)
            return 0.0
        end
    end

    # 2) Build integer bounds with proper floating-point precision
    # Use much smaller guard to avoid incorrect bounds while handling floating-point errors
    eps_guard = 1e-12
    
    # Calculate bounds for the summation variable 2r1 (stored as integers)
    kmin1 = Int(round(2*abs(ra - ri) + eps_guard))
    kmin2 = Int(round(2*abs(rb - rf) + eps_guard))  
    kmin3 = Int(round(2*abs(rd - rh) + eps_guard))
    minrda_candidates = [kmin1, kmin2, kmin3]
    
    kmax1 = Int(round(2*(ra + ri) + eps_guard))
    kmax2 = Int(round(2*(rb + rf) + eps_guard))
    kmax3 = Int(round(2*(rd + rh) + eps_guard))
    maxrda_candidates = [kmax1, kmax2, kmax3]
    
    # 3) Improved parity logic: find common parity and valid range
    # Check if there exists a valid summation range with consistent parity
    parities = [k & 1 for k in minrda_candidates]
    
    # All minimum bounds should have the same parity for a valid Nine-j symbol
    if !all(p == parities[1] for p in parities)
        return 0.0
    end
    
    common_parity = parities[1]
    minrda = max(minrda_candidates...)
    maxrda = min(maxrda_candidates...)
    
    # Ensure minrda has the correct parity
    if (minrda & 1) != common_parity
        minrda += 1
    end
    
    # Check if valid range exists
    if minrda > maxrda
        return 0.0
    end
    
    # Additional stability check: ensure the range makes physical sense
    if minrda < 0 || maxrda < 0
        return 0.0
    end

    # 4) Sum with numerical stability checks
    result = 0.0
    
    for n1 in minrda:2:maxrda
        r1 = n1 / 2.0
        
        # Calculate the three Racah W coefficients
        # Ensure your racahW is truly Racah W(a,b,c,d;e,f)
        w1 = racahW(Float64, ra, ri, rd, rh, r1, rg)
        w2 = racahW(Float64, rb, rf, rh, rd, r1, re)  
        w3 = racahW(Float64, ra, ri, rb, rf, r1, rc)
        
        # Numerical stability: check for NaN or Inf in Racah W coefficients
        if !isfinite(w1) || !isfinite(w2) || !isfinite(w3)
            continue  # Skip this term if any Racah W is invalid
        end
        
        # Calculate the contribution to the sum
        weight = 2*r1 + 1.0  # This equals n1 + 1
        contribution = weight * w1 * w2 * w3
        
        # Additional numerical stability: skip extremely small contributions
        # that might be due to numerical errors
        if abs(contribution) < 1e-16
            continue
        end
        
        result += contribution
        
        # Safety check to prevent runaway calculations
        if !isfinite(result)
            return 0.0
        end
    end
    
    # Final numerical stability check
    return isfinite(result) ? result : 0.0
end



end # end module 