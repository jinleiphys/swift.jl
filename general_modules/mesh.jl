module mesh 
using FastGaussQuadrature
export initialmesh

# # angular mesh 
# nθ = 24
# θi = Float64[]
# dθi = Float64[]

# # radial mesh
# nx = 50
# ny = 50
# xi = Float64[]
# dxi = Float64[]
# yi = Float64[]
# dyi = Float64[]

# xmax = 10.0
# ymax = 10.0



function initialmesh(nθ::Int,nx::Int,ny::Int, 
    xmax::Float64, ymax::Float64,alpha::Float64)
    θi = Vector{Float64}(undef, nθ)
    dθi = Vector{Float64}(undef, nθ)
    xi = Vector{Float64}(undef, nx)
    dxi = Vector{Float64}(undef, nx)
    yi = Vector{Float64}(undef, ny)
    dyi = Vector{Float64}(undef, ny)

    cosθi, dcosθi = gausslegendre(nθ)    
    xi, dxi = scale_gausslaguerre(nx,xmax,0.0)
    yi, dyi = scale_gausslaguerre(ny,ymax,0.0)


    return cosθi, dcosθi, xi, dxi, yi, dyi
end 

function scale_gausslaguerre(nx, xmax, alpha)
    # Generate Gaussian-Laguerre quadrature points and weights on [0,∞]
    xi, dxi = gausslaguerre(nx, alpha)
    
    # Compute scaling factor
    scaling_factor = xmax / xi[end]  # Assuming xi is sorted in ascending order
    
    # Scale the mesh points
    scaled_xi = xi * scaling_factor
    
    # Scale the weights
    # The division by xi.^alpha and exp.(-xi) removes the Laguerre weight function
    scaled_dxi = dxi * scaling_factor ./ (xi.^alpha .* exp.(-xi))
    
    return scaled_xi, scaled_dxi
end


function scale_gausslegendre(x, w, x1, x2)
    # Scale Gauss-Legendre quadrature points and weights from [-1,1] to [x1,x2]
    # Parameters:
    #   x: Points from gausslegendre in the [-1,1] interval
    #   w: Weights from gausslegendre in the [-1,1] interval
    #   x1: Lower bound of the target interval
    #   x2: Upper bound of the target interval
    
    # Calculate midpoint and half-length
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    
    # Scale points
    scaled_x = xm .+ xl .* x
    
    # Scale weights
    scaled_w = xl .* w
    
    return scaled_x, scaled_w
end
end # end module 