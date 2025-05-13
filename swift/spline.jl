
module spline

export cubherm_interp, cubherm_interp_point

function cubherm_interp(xold::Vector{Float64}, yold::Vector{Float64}, xnew::Vector{Float64})
    n = length(xold)
    m = length(xnew)
    
    # Check if input arrays have consistent dimensions
    if length(yold) != n
        throw(ArgumentError("xold and yold must have the same length"))
    end
    
    # Check if we have enough points for proper cubic interpolation
    if n < 3
        # Fall back to simpler interpolation for small grid sizes
        if n == 1
            return fill(yold[1], m)
        elseif n == 2
            # Linear interpolation for 2 points
            return [yold[1] + (x - xold[1]) * (yold[2] - yold[1]) / (xold[2] - xold[1]) for x in xnew]
        end
    end
    
    # Ensure xold is sorted (required for interpolation)
    if !issorted(xold)
        throw(ArgumentError("xold must be sorted in ascending order"))
    end
    
    # Initialize output array
    ynew = zeros(Float64, m)
    
    # Process each new point
    for j in 1:m
        x = xnew[j]
        
        # Handle extrapolation beyond grid bounds
        if x <= xold[1]
            # Extrapolate below the grid
            if n >= 2
                ynew[j] = yold[1] + (x - xold[1]) * (yold[2] - yold[1]) / (xold[2] - xold[1])
            else
                ynew[j] = yold[1]
            end
            continue
        elseif x >= xold[n]
            # Extrapolate above the grid
            if n >= 2
                ynew[j] = yold[n-1] + (x - xold[n-1]) * (yold[n] - yold[n-1]) / (xold[n] - xold[n-1])
            else
                ynew[j] = yold[n]
            end
            continue
        end
        
        # Find the right interval using binary search
        idx2 = searchsortedlast(xold, x)
        idx3 = min(idx2 + 1, n)
        
        # For cubic interpolation we need 4 points
        # Use special handling for points near edges
        if idx2 == 1
            # Near the left edge of the grid
            idx1 = 1
            idx2 = 1
            idx3 = 2
            idx4 = 3
        elseif idx2 >= n-1
            # Near the right edge of the grid
            idx1 = n-3
            idx2 = n-2
            idx3 = n-1
            idx4 = n
        else
            # Regular case: we're within the grid
            idx1 = idx2 - 1
            idx4 = idx3 + 1
        end
        
        # Safety bounds check
        idx1 = max(1, min(idx1, n))
        idx2 = max(1, min(idx2, n))
        idx3 = max(1, min(idx3, n))
        idx4 = max(1, min(idx4, n))
        
        # Use linear interpolation near the grid edges
        if idx2 == 1 || idx3 == n
            # Linear interpolation between idx2 and idx3
            t = (x - xold[idx2]) / (xold[idx3] - xold[idx2])
            ynew[j] = (1.0 - t) * yold[idx2] + t * yold[idx3]
        else
            x0 = xold[idx1]
            x1 = xold[idx2]
            x2 = xold[idx3]
            x3 = xold[idx4]
            
            # Check for very close points that could cause numerical issues
            if abs(x1 - x0) < 1e-10 || abs(x2 - x1) < 1e-10 || abs(x3 - x2) < 1e-10
                # Fall back to linear interpolation for numerical stability
                t = (x - x1) / (x2 - x1)
                ynew[j] = (1.0 - t) * yold[idx2] + t * yold[idx3]
                continue
            end
            
            # Factors for the derivatives
            d10 = x1 - x0
            d21 = x2 - x1
            d32 = x3 - x2
            d20 = x2 - x0
            d31 = x3 - x1
            
            # Check for potential division by zero
            if abs(d20) < 1e-10 || abs(d31) < 1e-10 || abs(d21) < 1e-10
                # Fall back to linear interpolation
                t = (x - x1) / (x2 - x1)
                ynew[j] = (1.0 - t) * yold[idx2] + t * yold[idx3]
                continue
            end
            
            dfak13 = (d21/d10 - d10/d21) / d20
            dfak14 = -d32 / (d21 * d31)
            dfak23 = d10 / (d21 * d20)
            dfak24 = (d32/d21 - d21/d32) / d31
            dfak03 = -d21 / (d10 * d20)
            dfak34 = d21 / (d32 * d31)
            
            # The cubic Hermitian splines
            dn1 = x - x1
            d2n = x2 - x
            
            # Avoid division by very small numbers
            if abs(d21) < 1e-10
                # Fall back to linear interpolation
                t = (x - x1) / (x2 - x1)
                ynew[j] = (1.0 - t) * yold[idx2] + t * yold[idx3]
                continue
            end
            
            phidiv = 1.0 / (d21 * d21 * d21)
            phi1 = d2n * d2n * phidiv * (d21 + 2.0 * dn1)
            phi2 = dn1 * dn1 * phidiv * (d21 + 2.0 * d2n)
            phidiv = phidiv * d21 * dn1 * d2n
            phi3 = phidiv * d2n
            phi4 = -phidiv * dn1
            
            # Calculate weights
            w1 = phi3 * dfak03
            w2 = phi1 + phi3 * dfak13 + phi4 * dfak14
            w3 = phi2 + phi3 * dfak23 + phi4 * dfak24
            w4 = phi4 * dfak34
            
            # Interpolation of q*f(q) if needed
            # Commented out as it depends on application specifics
            # Uncomment if this scaling is needed for your application
            # if x != 0.0 # Avoid division by zero
            #     w1 *= x0 / x
            #     w2 *= x1 / x
            #     w3 *= x2 / x
            #     w4 *= x3 / x
            # end
            
            # Compute the interpolated value
            ynew[j] = w1 * yold[idx1] + w2 * yold[idx2] + w3 * yold[idx3] + w4 * yold[idx4]
        end
    end
    
    return ynew
end


function cubherm_interp_point(xold::Vector{Float64}, yold::Vector{Float64}, x::Float64)
    return cubherm_interp(xold, yold, [x])[1]
end


end 