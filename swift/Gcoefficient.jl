module Gcoefficient
using WignerSymbols
using SphericalHarmonics
include("../general_modules/channels.jl")
using .channels
include("../general_modules/mesh.jl")
using .mesh

Yλαout = zeros(Float64,nθ,λmax^2+2*λmax+1,2)        # last dimension for the permutation operator 1 for P+; 2 for P-
Ylαin = zeros(Float64,nθ,ny,nx,lmax^2+2*lmax+1,2)   # last dimension for the permutation operator 1 for P+; 2 for P-
Yλαout = zeros(Float64,nθ,ny,nx,λmax^2+2*λmax+1,2)  # last dimension for the permutation operator 1 for P+; 2 for P-
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
# l, m = 2, 1
# theta, phi = π/4, π/3
# Y = sphericalharmonic(l, m, theta, phi)

 function Gαα'(αout, αin,P::Char)
    if P == '+'  # compute Gα3α1   
        a=-0.5
        b=1.0
        c=-0.75
        d=-0.5

        phase = (-1)^(α.T12[αin] + α.s12[αin] + 2*s1+s2+s3+2*t1+t2+t3)
        Cisospin=hat(α.T12[αin])*hat(α.T12[αout])*wigner6j(t1,t2,α.T12[αout],t3,α.T[αin],α.T12[αin])
        Cspin=hat(α.J12[αin])*hat(α.J12[αout])*hat(α.J3[αin])*hat(α.J3[αout])*hat(α.s12[αin])*hat(α.s12[αout]) 

        LLmin =max(abs(α.l[αin]-α.λ[αin]),abs(α.l[αout]-α.λ[αout]))
        LLmax =min(α.l[αin]+α.λ[αin],α.l[αout]+α.λ[αout])

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

        LLmin =max(abs(α.l[αin]-α.λ[αin]),abs(α.l[αout]-α.λ[αout]))
        LLmax =min(α.l[αin]+α.λ[αin],α.l[αout]+α.λ[αout])

        nSmin= max(Int(2*abs(α.s12[αin]-α.s3[αin])), Int(2*abs(α.s12[αout]-α.s3[αout])))
        nSmax= min(Int(2*(α.s12[αin]+α.s3[αin])), Int(2*(α.s12[αout]+α.s3[αout])))
    
    else
        error("Parameter P must be '+' or '-'")
    end
    


 end


function YYcoupling(Y4,P::Char)
    Y4 = zeros(Float64, nθ, nx, ny, α.nchmax, α.nchmax)

end 

function initialY 
# set x_3 as z-direction,
    Ylαout = sqrt( (2.*α.l[αout]+1.)/(4.*π) )
    Yλαout . = 0.0

    for λ in 0:λmax
        for m in -λ:λ
            nch=λ^2+λ+m+1
                for i in 1:nθ
                Yλαout[i,nch] =  sphericalharmonic(λ, m, θi[i], 0.0)
            end
        end
    end

# now compute the sperical harmonics for incoming channels 
# |x1|   | -0.5    1    |   |x3|           |0 |
#     =  |              |   |  |       x3= |0 |
# |y1|   | -0.75  -0.5  |   |y3|           |x3|

# |x2|   | -0.5    -1    |   |x3|         |y3√{1-cos(θi)}|
#     =  |               |   |  |      y3=|      0       |
# |y2|   | 0.75  -0.5    |   |y3|         |   y3cos(θi)  |

   if P == '+'  # compute Gα3α1   
       a=-0.5
       b=1.0
       c=-0.75
       d=-0.5
    elseif P == '-'  # compute Gα3α2
       a=-0.5
       b=-1.0
       c=0.75
       d=-0.5
    else
       error("Parameter P must be '+' or '-'") 
    end 
    # set the ϕ angle for the spherical harmonics
    ϕx=0
    ϕy=0 
    if b<0 
        ϕx=π
    end
    if d<0
        ϕy=π
    end

    for ix in 1:nx
        for iy in 1:ny
            for iθ in 1:nθ

                xin = sqrt(a^2*xi[ix]^2 + b^2*yi[iy]^2 + 2*a*b*xi[ix]*yi[iy]*cos(θi[iθ]))
                yin = sqrt(c^2*xi[ix]^2 + d^2*yi[iy]^2 + 2*c*d*xi[ix]*yi[iy]*cos(θi[iθ]))
                xzin = a*xi[ix] + b*yi[iy]*cos(θi[iθ])
                yzin = c*xi[ix] + d*yi[iy]*cos(θi[iθ])
                θx = acos(xzin/xin)
                θy = acos(yzin/yin)

                for l in 0:lmax
                    for m in -l:l
                        
                    end


                end 





            end
        end
    end


end # function YYcoupling




 function hat(x)
    y = sqrt(2.0*x+1.0)
    return y
 end 




end # end module 