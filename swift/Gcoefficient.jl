module Gcoefficient
using WignerSymbols
using SphericalHarmonics
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
    Y4 = YYcoupling(α, nθ, ny, nx, Ylin, Yλin, Yλout)
    Gresult = Gαα(nθ, ny, nx, α, Y4)
    return Gresult
 end 

 function Gαα(nθ, ny, nx, α,Y4)
    s1 = α.s1
    s2 = α.s2
    s3 = α.s3
    t1 = α.t1
    t2 = α.t2
    t3 = α.t3
    Gαoutαin = zeros(ComplexF64, nθ, ny, nx, α.nchmax, α.nchmax, 2)
    for perm_index in 1:2 
        for αout in 1:α.nchmax
            for αin in 1:α.nchmax
                if perm_index == 1  # compute Gα3α1: phase = (-1)^(s23 + 2s1 + s2 + s3) * (-1)^(T23 + 2t1 + t2 + t3)
                    # Note: s23 corresponds to α.s12[αin] in the input channel, T23 corresponds to α.T12[αin]
                    phase = (-1)^(α.s12[αin] + 2*s1 + s2 + s3) * (-1)^(α.T12[αin] + 2*t1 + t2 + t3)
                    Cisospin=hat(α.T12[αin])*hat(α.T12[αout])*wigner6j(t1,t2,α.T12[αout],t3,α.T,α.T12[αin])
                    Cspin=hat(α.J12[αin])*hat(α.J12[αout])*hat(α.J3[αin])*hat(α.J3[αout])*hat(α.s12[αin])*hat(α.s12[αout]) 
                    nSmin= max(Int(2*abs(α.s12[αin]-s1)), Int(2*abs(α.s12[αout]-s3)))
                    nSmax= min(Int(2*(α.s12[αin]+s1)), Int(2*(α.s12[αout]+s3)))
                elseif perm_index == 2 # compute Gα3α2: phase = (-1)^(s12 + 2s3 + s1 + s2) * (-1)^(T12 + 2t3 + t1 + t2)
                    # Note: s12 corresponds to α.s12[αout] in the output channel, T12 corresponds to α.T12[αout]
                    phase = (-1)^(α.s12[αout] + 2*s3 + s1 + s2) * (-1)^(α.T12[αout] + 2*t3 + t1 + t2)
                    Cisospin=hat(α.T12[αout])*hat(α.T12[αin])*wigner6j(t3,t1,α.T12[αin],t2,α.T,α.T12[αout])
                    Cspin=hat(α.J12[αin])*hat(α.J12[αout])*hat(α.J3[αin])*hat(α.J3[αout])*hat(α.s12[αin])*hat(α.s12[αout])
                    nSmin= max(Int(2*abs(α.s12[αin]-s2)), Int(2*abs(α.s12[αout]-s3)))
                    nSmax= min(Int(2*(α.s12[αin]+s2)), Int(2*(α.s12[αout]+s3)))
                else
                    error("Parameter P must be '+' or '-'")
                end
                LLmin =max(abs(α.l[αin]-α.λ[αin]),abs(α.l[αout]-α.λ[αout]))
                LLmax =min(α.l[αin]+α.λ[αin],α.l[αout]+α.λ[αout])
                
                for LL in LLmin:LLmax 
                    for nSS in nSmin:nSmax
                        SS = nSS/2.
                        if perm_index == 1
                            f0 = (2*SS+1.0) * u9(float(α.l[αout]),α.s12[αout],α.J12[αout], 
                                                 float(α.λ[αout]),α.s3, α.J3[αout], 
                                                 float(LL), SS, α.J) * 
                                              u9(float(α.l[αin]), α.s12[αin], α.J12[αin], 
                                                 float(α.λ[αin]), α.s1, α.J3[αin], 
                                                 float(LL), SS, α.J)
                            f1=wigner6j(s1, s2, α.s12[αout], s3, SS, α.s12[αin])
                        elseif perm_index == 2
                            f0 = (2*SS+1.0) * u9(float(α.l[αout]),α.s12[αout],α.J12[αout], 
                                                 float(α.λ[αout]),α.s3, α.J3[αout], 
                                                 float(LL), SS, α.J) * 
                                              u9(float(α.l[αin]), α.s12[αin], α.J12[αin], 
                                                 float(α.λ[αin]), α.s2, α.J3[αin], 
                                                 float(LL), SS, α.J)
                            f1=wigner6j(s3, s1, α.s12[αin], s2, SS, α.s12[αout])
                        end
                            
                        Gαoutαin[:,:,:,αin, αout, perm_index] += Y4[:, :, :, αin, αout, perm_index]  * phase * Cisospin * Cspin * f0 * f1

                    end 
                end 
            end 
        end 
    end 



    return Gαoutαin


 end


 function YYcoupling(α, nθ, ny, nx, Ylin, Yλin, Yλout)
    # Add parameter Yλout which was missing
    
    # Input validation
    if size(Ylin, 1) != nθ || size(Ylin, 2) != ny || size(Ylin, 3) != nx
        error("Dimensions of Ylin do not match nθ, ny, nx")
    end
    if size(Yλin, 1) != nθ || size(Yλin, 2) != ny || size(Yλin, 3) != nx
        error("Dimensions of Yλin do not match nθ, ny, nx")
    end
    if size(Yλout, 1) != nθ
        error("First dimension of Yλout does not match nθ")
    end
    
    # Initialize output array
    Y4 = zeros(ComplexF64, nθ, ny, nx, α.nchmax, α.nchmax, 2)
    
    for αout in 1:α.nchmax
        # Calculate Ylαout coefficient
        Ylαout = sqrt((2 * α.l[αout] + 1) / (4 * π))
        
        for αin in 1:α.nchmax
            # Calculate angular momentum bounds
            LLmin = max(abs(α.l[αin] - α.λ[αin]), abs(α.l[αout] - α.λ[αout]))
            LLmax = min(α.l[αin] + α.λ[αin], α.l[αout] + α.λ[αout])
            
            for LL in LLmin:LLmax
                minl = min(LL, α.λ[αout])
                
                for ML in -minl:minl
                    # Calculate nchλout index with bounds check
                    nchλout = Int(α.λ[αout]^2 + α.λ[αout] + ML + 1)
                    if nchλout <= 0 || nchλout > size(Yλout, 2)
                        @warn "nchλout index $nchλout out of bounds, skipping"
                        continue
                    end
                    
                    # Calculate Clebsch-Gordan coefficient for output
                    # Note: For z-axis (θ=0), ml=0 for the l component
                    CGout = clebschgordan(α.l[αout], 0, α.λ[αout], ML, LL, ML)
                    
                    # Pre-calculate Yout values - use broadcast for efficiency
                    Yout = Ylαout .* Yλout[:, nchλout]
                    
                    # Loop over ml and mλ values
                    for ml in -α.l[αin]:α.l[αin]
                        for mλ in -α.λ[αin]:α.λ[αin]
                            # Skip if quantum numbers don't satisfy coupling condition
                            if mλ + ml != ML
                                continue
                            end
                            
                            # Calculate indices for input channel components with bounds check
                            nchlin = Int(α.l[αin]^2 + α.l[αin] + ml + 1)
                            nchλin = Int(α.λ[αin]^2 + α.λ[αin] + mλ + 1)
                            
                            if nchlin <= 0 || nchlin > size(Ylin, 4) || 
                               nchλin <= 0 || nchλin > size(Yλin, 4)
                                @warn "Channel indices out of bounds: nchlin=$nchlin, nchλin=$nchλin, skipping"
                                continue
                            end
                            
                            # Calculate Clebsch-Gordan coefficient for input
                            CGin = clebschgordan(α.l[αin], ml, α.λ[αin], mλ, LL, ML)
                            
                            # Calculate coupling factor once outside the loops
                            coupling_factor = CGin * CGout
                            
                            # Calculate coupled spherical harmonics using broadcasting
                            for ix in 1:nx
                                for iy in 1:ny
                                    # Vectorize the innermost loop for better performance
                                    for iθ in 1:nθ
                                        Yin1 = Ylin[iθ, iy, ix, nchlin, 1] * Yλin[iθ, iy, ix, nchλin, 1]
                                        Yin2 = Ylin[iθ, iy, ix, nchlin, 2] * Yλin[iθ, iy, ix, nchλin, 2]
                                        
                                        Y4_factor = coupling_factor * Yout[iθ]
                                        Y4[iθ, iy, ix, αin, αout, 1] += Y4_factor * Yin1
                                        Y4[iθ, iy, ix, αin, αout, 2] += Y4_factor * Yin2
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    Y4 = Y4 .* 8.0 .* π^2
    
    return Y4
end

function initialY(λmax, lmax, nθ, nx, ny, cosθi, xi, yi)
    
    Yλout = zeros(ComplexF64, nθ, λmax^2 + 2*λmax + 1)
    Ylin = zeros(ComplexF64, nθ, ny, nx, lmax^2 + 2*lmax + 1, 2)    
    Yλin = zeros(ComplexF64, nθ, ny, nx, λmax^2 + 2*λmax + 1, 2)

    # Set x_3 as z-direction
    for i in 1:nθ
        Yλ = computeYlm(acos(cosθi[i]), 0.0, λmax)
        for λ in 0:λmax
           for m in -λ:λ
                nch = λ^2 + λ + m + 1
                Yλout[i, nch] =  Yλ[nch]
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
        else
           error("Parameter P must be '+' or '-'")
        end
    
    # Set the ϕ angle for the spherical harmonics
    ϕx = b < 0 ? π : 0
    ϕy = d < 0 ? π : 0
    
    for ix in 1:nx
        for iy in 1:ny
            for iθ in 1:nθ
                # Compute transformed coordinates
                xin = sqrt(a^2*xi[ix]^2 + b^2*yi[iy]^2 + 2*a*b*xi[ix]*yi[iy]*cosθi[iθ])
                yin = sqrt(c^2*xi[ix]^2 + d^2*yi[iy]^2 + 2*c*d*xi[ix]*yi[iy]*cosθi[iθ])
                
                xzin = a*xi[ix] + b*yi[iy]*cosθi[iθ]
                # Ensure argument to acos is in valid range [-1, 1]
                θx = acos(clamp(xzin/xin, -1.0, 1.0))
                
                yzin = c*xi[ix] + d*yi[iy]*cosθi[iθ]
                # Ensure argument to acos is in valid range [-1, 1]
                θy = acos(clamp(yzin/yin, -1.0, 1.0))

                
                Yl = computeYlm(θx, ϕx, lmax)
                Yλ = computeYlm(θy, ϕy, λmax)
                # Compute spherical harmonics for each (l,m) combination
                for l in 0:lmax
                    for m in -l:l
                        nch = l^2 + l + m + 1
                        Ylin[iθ, iy, ix, nch, perm_index] = Yl[nch]
                    end
                end
                
                # Compute spherical harmonics for each (λ,m) combination
                for λ in 0:λmax
                    for m in -λ:λ
                        nch = λ^2 + λ + m + 1
                        Yλin[iθ, iy, ix, nch, perm_index] = Yλ[nch]
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
function u9(ra::Float64, rb::Float64, rc::Float64, rd::Float64, re::Float64, 
            rf::Float64, rg::Float64, rh::Float64, ri::Float64)::Float64
    
    # Check if all inputs are valid angular momenta (non-negative)
    if any(x < 0 for x in [ra, rb, rc, rd, re, rf, rg, rh, ri])
        return 0.0
    end
    
    # Check triangle inequalities for all triads
    triangles = [
        (ra, rb, rc), (rd, re, rf), (rg, rh, ri),  # rows
        (ra, rd, rg), (rb, re, rh), (rc, rf, ri)   # columns
    ]
    
    for (j1, j2, j3) in triangles
        if abs(j1 - j2) > j3 || j1 + j2 < j3
            return 0.0
        end
    end
    
    # Implementation using the sum formula for 9-j symbols in terms of 6-j symbols
    result = 0.0
    
    # Determine the range for the summation variable
    minr = max(abs(ra - ri), abs(rb - rf), abs(rd - rh))
    maxr = min(ra + ri, rb + rf, rd + rh)
    
    # Skip calculation if the triangle conditions aren't satisfied
    if minr > maxr
        return 0.0
    end
    
    # Sum over all valid angular momenta
    for n1 in Int(2*minr):2:Int(2*maxr)
        r1 = n1/2.0  # Convert back to angular momentum value
        
        # Calculate the triple product of Racah W coefficients
        # Note: racahW(T, j₁, j₂, J, j₃, J₁₂, J₂₃) computes W(j₁, j₂, J, j₃; J₁₂, J₂₃)
        w1 = racahW(Float64, ra, ri, rd, rh, r1, rg)
        w2 = racahW(Float64, rb, rf, rh, rd, r1, re)
        w3 = racahW(Float64, ra, ri, rb, rf, r1, rc)
        
        # Add contribution to the sum
        # Factor of (2*r1 + 1) corresponds to (2x+1) in the sum formula
        result += (2.0 * r1 + 1.0) * w1 * w2 * w3 * ((-1.0)^(2*r1))
    end
    
    return result
end


end # end module 