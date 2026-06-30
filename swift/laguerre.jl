module Laguerre
export lagrange_laguerre_basis,
       lagrange_laguerre_regularized_basis,
       lagrange_laguerre_regularized_basis1,
       lagrange_laguerre_regularized_derivs
# module used to generate the Lagrange-laguerre functions

"""
    lagrange_laguerre_regularized_derivs(r, rr, phi, alpha, hs) -> (f, f1, f2)

Value, 1st and 2nd derivative (w.r.t. the REAL physical coordinate r, in fm) of the
regularized Lagrange-Laguerre basis at real point `r`, for ALL nr basis functions.
Same basis as `lagrange_laguerre_regularized_basis` (θ=0):
    f_i(r) = phi_i (r/x_i)^p exp(-(r-x_i)/(2 hs)) ∏_{j≠i} (r-x_j)/(x_i-x_j),  p = α/2 + 1.
Derivatives via the log-derivative L_i(r) = p/r - 1/(2hs) + Σ_{j≠i} 1/(r-x_j):
    f_i' = f_i L_i,   f_i'' = f_i (L_i^2 + L_i'),  L_i' = -p/r^2 - Σ_{j≠i} 1/(r-x_j)^2.
This is the REAL-r value/∂/∂² the q-operator complex-scaling layer (ecs.jl) needs to put
the x coordinate (and any Lagrange-Laguerre coordinate) on a smooth-ECS contour, exactly as
splines.jl supplies them for the spline y coordinate. r must NOT coincide with a mesh node.
"""
function lagrange_laguerre_regularized_derivs(r::Real, rr::Vector{<:Real}, phi::Vector{<:Real},
                                              alpha::Float64, hs::Float64)
    nr = length(rr)
    p  = alpha / 2.0 + 1.0
    f  = Vector{Float64}(undef, nr)
    f1 = Vector{Float64}(undef, nr)
    f2 = Vector{Float64}(undef, nr)
    for i in 1:nr
        xi = rr[i]
        val = phi[i] * (r / xi)^p * exp(-(r - xi) / (2.0 * hs))
        L  = p / r - 1.0 / (2.0 * hs)
        Lp = -p / r^2
        for j in 1:nr
            j == i && continue
            xj = rr[j]
            val *= (r - xj) / (xi - xj)
            L   += 1.0 / (r - xj)
            Lp  -= 1.0 / (r - xj)^2
        end
        f[i]  = val
        f1[i] = val * L
        f2[i] = val * (L^2 + Lp)
    end
    return f, f1, f2
end


function lagrange_laguerre_basis(x::Number, 
                                laguerre_rr::Vector{<:Number}, 
                                phi::Vector{<:Number}, 
                                alpha::Float64,
                                hs::Float64,
                                theta::Float64=0.0)
     """
    Compute Lagrange-Laguerre regularized basis functions at point x with improved numerical stability.
    
    Parameters:
    - x: Point where basis functions are evaluated (can be real or complex)
    - laguerre_rr: Array of Laguerre mesh points
    - phi: Precomputed normalization factors (equivalent to Lagrange functions at the mesh points)
    - alpha: Parameter in Laguerre weight function
    - hs: Scaling factor for the x-coordinate
    - theta: Complex rotation angle in radians (default = 0.0)
    
    Returns:
    - Vector of basis function values at point x
    """
   # Number of basis functions
    nr = length(laguerre_rr)
    
    # Complex rotation factor
    eitheta = exp(im * theta)
    
    # Rotated coordinate
    r = x / eitheta 
    
    # Initialize vector to store basis function values
    lag_func = zeros(Complex{Float64}, nr)
    
    # Compute each basis function
    for i_basis = 1:nr
        # Laguerre mesh point
        xi = laguerre_rr[i_basis] 
        
        # Initial part of the basis function with precomputed normalization
        lag_func[i_basis] = phi[i_basis] * 
                           (r/xi)^(alpha/2.0) * 
                           exp(-(r-xi)/2.0/hs)
        
        # Compute the product term
        prod = Complex{Float64}(1.0)
        for j = 1:nr
            if j == i_basis
                continue
            end
            xj = laguerre_rr[j]
            prod *= (r-xj)/(xi-xj)
        end
        
        # Multiply by the product term
        lag_func[i_basis] *= prod
    end
    


    return lag_func
end


function lagrange_laguerre_regularized_basis(x::Number, 
                                laguerre_rr::Vector{<:Number}, 
                                phi::Vector{<:Number}, 
                                alpha::Float64,
                                hs::Float64,
                                theta::Float64=0.0)
    """
    Compute Lagrange-Laguerre regularized basis functions at point x.
    
    Parameters:
    - x: Point where basis functions are evaluated (can be real or complex)
    - laguerre_rr: Array of Laguerre mesh points
    - phi: Precomputed normalization factors (equivalent to Lagrange functions at the mesh points)
    - alpha: Parameter in Laguerre weight function
    - hs: Scaling factor for the x-coordinate
    - theta: Complex rotation angle in radians (default = 0.0)
    
    Returns:
    - Vector of basis function values at point x
    """
    
    # Number of basis functions
    nr = length(laguerre_rr)
    
    # Complex rotation factor
    eitheta = exp(im * theta)
    
    # Rotated coordinate
    r = x / eitheta 
    
    # Initialize vector to store basis function values
    lag_func = zeros(Complex{Float64}, nr)
    
    # Compute each basis function
    for i_basis = 1:nr
        # Laguerre mesh point
        xi = laguerre_rr[i_basis] 
        
        # Initial part of the basis function with precomputed normalization
        lag_func[i_basis] = phi[i_basis] * 
                           (r/xi)^(alpha/2.0 + 1.0) * 
                           exp(-(r-xi)/2.0/hs)
        
        # Compute the product term
        prod = Complex{Float64}(1.0)
        for j = 1:nr
            if j == i_basis
                continue
            end
            xj = laguerre_rr[j]
            prod *= (r-xj)/(xi-xj)
        end
        
        # Multiply by the product term
        lag_func[i_basis] *= prod
    end
    

    return lag_func
end




function lagrange_laguerre_regularized_basis1(x::Number, 
                                laguerre_rr::Vector{<:Number}, 
                                phi::Vector{<:Number}, 
                                alpha::Float64,
                                hs::Float64,
                                theta::Float64=0.0)
    """
    Compute Lagrange-Laguerre regularized basis functions at point x with sign corrections.
    
    Parameters:
    - x: Point where basis functions are evaluated (can be real or complex)
    - laguerre_rr: Array of Laguerre mesh points
    - phi: Precomputed normalization factors (equivalent to Lagrange functions at the mesh points)
    - alpha: Parameter in Laguerre weight function
    - hs: Scaling factor for the x-coordinate
    - theta: Complex rotation angle in radians (default = 0.0)
    
    Returns:
    - Vector of basis function values at point x
    """
    
    # Number of basis functions
    nr = length(laguerre_rr)
    
    # Complex rotation factor
    eitheta = exp(im * theta)
    
    # Rotated coordinate
    r = x / eitheta 
    
    # For sign corrections, we need the real part for comparison
    r_real = real(r)  
    
    # Initialize vector to store basis function values
    lag_func = zeros(Complex{Float64}, nr)
    
    # Compute each basis function
    for i_basis = 1:nr
        # Laguerre mesh point
        xi = laguerre_rr[i_basis] 
        
        # Initial part of the basis function with precomputed normalization
        lag_func[i_basis] = abs(phi[i_basis]) * 
                           (r/xi)^(alpha/2.0 + 1.0) * 
                           exp(-(r-xi)/2.0/hs)
        
        # Compute the product term
        prod = Complex{Float64}(1.0)
        for j = 1:nr
            if j == i_basis
                continue
            end
            xj = laguerre_rr[j]
            prod *= abs((r-xj)/(xi-xj))
        end
        
        # Multiply by the product term
        lag_func[i_basis] *= prod

        if abs(lag_func[i_basis]) < 0
            println("Warning: Negative value encountered in lag_func at index $i_basis")
        end
    end
    
    # Binary search to find position of r_real in grid (similar to Fortran)
    ik = 0
    if r_real < laguerre_rr[1]
        ik = 0
    elseif r_real > laguerre_rr[nr]
        ik = nr
    else
        # Binary search
        i0 = 1
        in = nr
        ik = div(nr, 2)
        
        while in > i0 + 1
            if r_real > laguerre_rr[ik]
                i0 = ik
                ik = i0 + div(in - i0, 2)
            elseif r_real < laguerre_rr[ik]
                in = ik
                ik = i0 + div(in - i0, 2)
            else
                # Exact match found
                break
            end
        end
    end
    
    # Apply sign corrections (matching Fortran logic)
    # If ik is odd (mod(ik,2) != 0), negate all values
    if mod(ik, 2) != 0
        lag_func .= -lag_func
    end
    
    # Negate values from ik+1 to end
    if ik < nr
        lag_func[ik+1:nr] .= -lag_func[ik+1:nr]
    end
    
    return lag_func
end


end 