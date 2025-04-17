module Gcoefficient
using WignerSymbols
using SphericalHarmonics
export Gαα,  YYcoupling, initialY


# Yλout = zeros(Float64,nθ,λmax^2+2*λmax+1,2)        # last dimension for the permutation operator 1 for P+; 2 for P-
# Ylin = zeros(Float64,nθ,ny,nx,lmax^2+2*lmax+1,2)   # last dimension for the permutation operator 1 for P+; 2 for P-
# Yλin = zeros(Float64,nθ,ny,nx,λmax^2+2*λmax+1,2)   # last dimension for the permutation operator 1 for P+; 2 for P-
# clebschgordan(j1, m1, j2, m2, j, m)


# Computing 6j symbols
# wigner6j(j1, j2, j3, j4, j5, j6)
# Corresponds to the symbol { j1 j2 j3 }
#                           { j4 j5 j6 }

# Computing 9j symbols
# wigner9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
# Corresponds to the symbol { j1 j2 j3 }
#                           { j4 j5 j6 }
#                           { j7 j8 j9 }

# Compute Y_l^m(θ,φ)
# Arguments: l (integer), m (integer), θ (polar angle), φ (azimuthal angle)
# computeYlm(θ, ϕ; lmax)
# The returned array may be indexed using (l,m)

 function Gαα(αout, αin,P::Char)
    if P == '+'  # compute Gα3α1   
        a=-0.5
        b=1.0
        c=-0.75
        d=-0.5

        phase = (-1)^(α.T12[αin] + α.s12[αin] + 2*s1+s2+s3+2*t1+t2+t3)
        Cisospin=hat(α.T12[αin])*hat(α.T12[αout])*wigner6j(t1,t2,α.T12[αout],t3,α.T[αin],α.T12[αin])
        Cspin=hat(α.J12[αin])*hat(α.J12[αout])*hat(α.J3[αin])*hat(α.J3[αout])*hat(α.s12[αin])*hat(α.s12[αout]) 

        # LLmin =max(abs(α.l[αin]-α.λ[αin]),abs(α.l[αout]-α.λ[αout]))
        # LLmax =min(α.l[αin]+α.λ[αin],α.l[αout]+α.λ[αout])

        nSmin= max(Int(2*abs(α.s12[αin]-α.s3[αin])), Int(2*abs(α.s12[αout]-α.s3[αout])))
        nSmax= min(Int(2*(α.s12[αin]+α.s3[αin])), Int(2*(α.s12[αout]+α.s3[αout])))




    elseif P == '-'  # compute Gα3α2
        a=-0.5
        b=-1.0
        c=0.75
        d=-0.5

        phase = (-1)^(α.T12[αout] + α.s12[αout] + 2*s3+s1+s2+2*t3+t1+t2)
        Cisospin=hat(α.T12[αout])*hat(α.T12[αin])*wigner6j(t3,t1,α.T12[αin],t2,α.T[αout],α.T12[αout])
        Cspin=hat(α.J12[αin])*hat(α.J12[αout])*hat(α.J3[αin])*hat(α.J3[αout])*hat(α.s12[αin])*hat(α.s12[αout])

        

        nSmin= max(Int(2*abs(α.s12[αin]-α.s3[αin])), Int(2*abs(α.s12[αout]-α.s3[αout])))
        nSmax= min(Int(2*(α.s12[αin]+α.s3[αin])), Int(2*(α.s12[αout]+α.s3[αout])))
    
    else
        error("Parameter P must be '+' or '-'")
    end
    


 end


 function YYcoupling(α, nθ, ny, nx, Ylin, Yλin)
    # Initialize output arrays
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
                    # Calculate nchλout index
                    nchλout = Int(α.λ[αout]^2 + α.λ[αout] + ML + 1)
                    
                    # Calculate Clebsch-Gordan coefficient for output
                    CGout = clebschgordan(α.l[αout], 0, α.λ[αout], ML, LL, ML)
                    
                    # Pre-calculate Yout values
                    Yout = zeros(Float64, nθ)
                    # Assuming Yλout is defined elsewhere and has shape [nθ, nchλout_max]
                    Yout[:] = Ylαout * Yλout[:, nchλout]
                    
                    # Loop over ml values
                    for ml in -α.l[αin]:α.l[αin]
                        # Loop over mλ values
                        for mλ in -α.λ[αin]:α.λ[αin]
                            # Skip if quantum numbers don't satisfy coupling condition
                            if mλ + ml != ML
                                continue
                            end
                            
                            # Calculate indices for input channel components
                            nchlin = Int(α.l[αin]^2 + α.l[αin] + ml + 1)
                            nchλin = Int(α.λ[αin]^2 + α.λ[αin] + mλ + 1)
                            
                            # Calculate Clebsch-Gordan coefficient for input
                            CGin = clebschgordan(α.l[αin], ml, α.λ[αin], mλ, LL, ML)
                            
                            # Calculate coupled spherical harmonics
                            for ix in 1:nx
                                for iy in 1:ny
                                    for iθ in 1:nθ
                                        # Calculate input harmonic products
                                        Yin1 = Ylin[iθ, iy, ix, nchlin, 1] * Yλin[iθ, iy, ix, nchλin, 1]
                                        Yin2 = Ylin[iθ, iy, ix, nchlin, 2] * Yλin[iθ, iy, ix, nchλin, 2]
                                        
                                        # Update Y4 tensor
                                        Y4[iθ, iy, ix, αin, αout, 1] += CGin * CGout * Yout[iθ] * Yin1
                                        Y4[iθ, iy, ix, αin, αout, 2] += CGin * CGout * Yout[iθ] * Yin2
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return Y4
end

function initialY(λmax, lmax, nθ, nx, ny, θi, xi, yi)
    
    Yλout = zeros(ComplexF64, nθ, λmax^2 + 2*λmax + 1)
    Ylin = zeros(ComplexF64, nθ, ny, nx, lmax^2 + 2*lmax + 1, 2)    
    Yλin = zeros(ComplexF64, nθ, ny, nx, λmax^2 + 2*λmax + 1, 2)

    # Set x_3 as z-direction
    for i in 1:nθ
        Yλ = computeYlm(θi[i], 0.0, λmax)
        for λ in 0:λmax
           for m in -λ:λ
                nch = λ^2 + λ + m + 1
                Yλout[i, nch] =  Yλ[(λ,m)]
            end
        end
    end

  for perm_index in 1:2 
    # Now compute the spherical harmonics for incoming channels
       if perm_index == 1  # compute Gα3α1
           a = -0.5
           b = 1.0
           c = -0.75
           d = -0.5
        elseif perm_index == 2 # compute Gα3α2
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
                xin = sqrt(a^2*xi[ix]^2 + b^2*yi[iy]^2 + 2*a*b*xi[ix]*yi[iy]*cos(θi[iθ]))
                yin = sqrt(c^2*xi[ix]^2 + d^2*yi[iy]^2 + 2*c*d*xi[ix]*yi[iy]*cos(θi[iθ]))
                
                # Handle potential division by zero
                if xin ≈ 0.0
                    θx = 0.0  # Default value when xin is very close to zero
                else
                    xzin = a*xi[ix] + b*yi[iy]*cos(θi[iθ])
                    # Ensure argument to acos is in valid range [-1, 1]
                    θx = acos(clamp(xzin/xin, -1.0, 1.0))
                end
                
                if yin ≈ 0.0
                    θy = 0.0  # Default value when yin is very close to zero
                else
                    yzin = c*xi[ix] + d*yi[iy]*cos(θi[iθ])
                    # Ensure argument to acos is in valid range [-1, 1]
                    θy = acos(clamp(yzin/yin, -1.0, 1.0))
                end
                Yl = computeYlm(θx, ϕx, lmax)
                Yλ = computeYlm(θy, ϕy, λmax)
                # Compute spherical harmonics for each (l,m) combination
                for l in 0:lmax
                    for m in -l:l
                        nch = l^2 + l + m + 1
                        Ylin[iθ, iy, ix, nch, perm_index] = Yl[(l, m)]
                    end
                end
                
                # Compute spherical harmonics for each (λ,m) combination
                for λ in 0:λmax
                    for m in -λ:λ
                        nch = λ^2 + λ + m + 1
                        Yλin[iθ, iy, ix, nch, perm_index] = Yλ[(λ, m)]
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




end # end module 