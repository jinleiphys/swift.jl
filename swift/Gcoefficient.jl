module Gcoefficient
using Pkg
Pkg.add("SphericalHarmonics")
Pkg.add("WignerSymbols")
using WignerSymbols
using SphericalHarmonics
include("../general_modules/channels.jl")
using .channels
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

 function Gαα'(αout, αin, x, y, θ, P::Char)
    if P == '+'  # compute Gα3α1   
        a=-0.5
        b=1.0
        c=-0.75
        d=-0.5
        phase = (-1)^(α.T12(αin) + α.s12(αin) + 2*s1+s2+s3+2*t1+t2+t3)
        Cisospin=hat(α.T12(αin))*hat(α.T12(αout))*wigner6j(t1,t2,α.T12(αout),t3,α.T(αin),α.T12(αin))
    elseif P == '-'  # compute Gα3α2
        a=-0.5
        b=-1.0
        c=0.75
        d=-0.5
        phase = (-1)^(α.T12(αout) + α.s12(αout) + 2*s3+s1+s2+2*t3+t1+t2)
        Cisospin=hat(α.T12(αout))*hat(α.T12(αin))*wigner6j(t3,t1,α.T12(αin),t2,α.T(αout),α.T12(αout))
    else
        error("Parameter P must be '+' or '-'")
    end
    


 end




 function hat(x)
    y = sqrt(2.0*x+1.0)
    return y
 end 




end # end module 