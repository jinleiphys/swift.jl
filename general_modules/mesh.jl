module mesh 
using FastGaussQuadrature
export initialmesh, nθ, nx, ny, xmax, ymax
export xi, dxi, yi, dyi
export θi, dθi
# angular mesh 
nθ = 24
θi = Float64[]
dθi = Float64[]

# radial mesh
nx = 50
ny = 50
xi = Float64[]
dxi = Float64[]
yi = Float64[]
dyi = Float64[]

xmax = 10.0
ymax = 10.0


# Function to update parameters
function update_parameters(params)
    global nθ = params["nθ"]
    global nx = params["nx"]
    global ny = params["ny"]
    global xmax = params["xmax"]
    global ymax = params["ymax"]
end

function initialmesh(alpha)
    global θi = Vector{Float64}(undef, nθ)
    global dθi = Vector{Float64}(undef, nθ)
    global xi = Vector{Float64}(undef, nx)
    global dxi = Vector{Float64}(undef, nx)
    global yi = Vector{Float64}(undef, ny)
    global dyi = Vector{Float64}(undef, ny)

    θi, dθi = gausslegendre(nθ)    
    xi, dxi = scale_gausslaguerre(nx,xmax,0.0)
    yi, dyi = scale_gausslaguerre(ny,ymax,0.0)

    # scaling the angular mesh from [-1,1] to [0,π]
    θi = (π/2) .* θi .+ (π/2)
    dθi = (π/2) .* dθi
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
end # end module 