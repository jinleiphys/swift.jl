module laguerre 
export laguerre_laguerre_basis
# module used to generate the Lagrange-laguerre functions 
function lagrange_laguerre_basis(x::Number, 
                                laguerre_rr::Vector{<:Number}, 
                                phi::Vector{<:Number}, 
                                alpha::Float64,
                                theta::Float64=0.0)
    """
    Compute Lagrange-Laguerre basis functions at point x.
    
    Parameters:
    - x: Point where basis functions are evaluated (can be real or complex)
    - laguerre_rr: Array of Laguerre mesh points
    - phi: Precomputed normalization factors (equivalent to Lagrange functions at the mesh points)
    - alpha: Parameter in Laguerre weight function
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
                           exp(-(r-xi)/2.0)
        
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


end 