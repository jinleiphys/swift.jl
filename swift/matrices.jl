module matrices 
using Kronecker
include("../NNpot/nuclear_potentials.jl")
using .NuclearPotentials
using WignerSymbols

const amu= 931.49432 # MeV
const m=1.008665 # amu
const ħ=197.3269718 # MeV. fm

# 1.008665 amu for neutron  amu=931.49432 MeV

function Rxy_matrix(α,grid)
# the channel index can be computed by i(iα,ix, iy) = (iα-1) * grid.nx * grid.ny + (ix-1)*grid.ny + iy



 end




 function T_matrix(α,grid) 
"""
α.nchmax is the maximum number of α channel index, α0 is the α parameter in Laguerre function 
"""
 Tαx = zeros(α.nchmax*grid.nx,α.nchmax*grid.nx)  # Initialize Tαx matrix
 Iy = Matrix{Float64}(I, grid.ny, grid.ny)

 for i in 1:α.nchmax

    T = Tx(grid.nx,grid.xx,grid.α,α.l[i])  
    T .= T .* ħ^2 / m / amu / grid.hsx^2
    row_start = (i-1)*grid.nx + 1
    row_end = i*grid.nx
    col_start = (i-1)*grid.nx + 1
    col_end = i*grid.nx
    Tαx[row_start:row_end, col_start:col_end] = T

 end 

 Tx_matrix = Tαx ⊗ Iy
 
 Ty_matrix = zeros(α.nchmax*grid.nx*grid.ny,α.nchmax*grid.nx*grid.ny)  # Initialize Ty_matrix
 i=0
 for iα in 1:α.nchmax
    for ix in 1:grid.nx 
       i += 1
       T= Tx(grid.ny,grid.yy,grid.α,α.λ[iα])
       T .= T .* ħ^2 * 0.75 / m / amu /grid.hsy^2

       row_start = (i-1)*grid.ny + 1
       row_end = i*grid.ny
       col_start = (i-1)*grid.ny + 1
       col_end = i*grid.ny
       Ty_matrix[row_start:row_end, col_start:col_end] = T

    end 
 end 

 return Tx_matrix, Ty_matrix
 end 


 function Tx(nx,xi,α0,l)
    # Compute the T matrix 
    # Parameters:
    # nx: number of points in the radial mesh
    # xi: radial mesh points
    # α0: parameter for the Laguerre function
    # l: angular momentum quantum number
    T = zeros(nx,nx)  # Initialize T matrix

    for i in 1:nx
        for j in 1:nx
            if i == j
                # Diagonal elements 
                T[i,j] = (-1.0 / (12.0 * xi[i]^2)) * (xi[i]^2 - 2.0 * (2.0 * nx + α0 + 1.0) * xi[i] + α0^2 - 4.0) - 
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j])) + 
                         l * (l + 1) / xi[i]^2
            else
                # Off-diagonal elements 
                T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) - 
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j]))
            end
        end
    end


    return T

 end # end function Tx

 function V_matrix(α, grid, potname)
    # Initialize matrix for storing potential energy
    Vαx = zeros(α.nchmax * grid.nx, α.nchmax * grid.nx)
    
    # Get nuclear potential matrix
    v12 = pot_nucl(α, grid, potname)
    
    # Convert to integer values for loop limits
    nt1 = Int(2 * α.t1)
    nt2 = Int(2 * α.t2)
    
    for j in 1:α.nchmax
        for i in 1:α.nchmax
            # Calculate start and end indices for blocks in the matrix
            row_start = (i - 1) * grid.nx + 1
            row_end = i * grid.nx
            col_start = (j - 1) * grid.nx + 1
            col_end = j * grid.nx
            
            for nmt1 in -nt1:nt1
                mt1 = nmt1 / 2.0
                for nmt2 in -nt2:nt2
                    mt2 = nmt2 / 2.0
                    for nmt1p in -nt1:nt1
                        mt1p = nmt1p / 2.0
                        for nmt2p in -nt2:nt2
                            mt2p = nmt2p / 2.0
                            
                            # Conservation of total m_t
                            if nmt1 + nmt2 == nmt1p + nmt2p
                                # Total magnetic quantum number
                                mt_total = mt1 + mt2
                                
                                # Common Clebsch-Gordan coefficients that can be precomputed
                                cg1 = clebschgordan(α.t1, mt1, α.t2, mt2, α.T12[i], mt_total)
                                cg2 = clebschgordan(α.t1, mt1p, α.t2, mt2p, α.T12[j], mt_total)
                                cg3 = clebschgordan(α.T12[i], mt_total, α.t3, α.MT - mt_total, α.T, α.MT)
                                cg4 = clebschgordan(α.T12[j], mt_total, α.t3, α.MT - mt_total, α.T, α.MT)
                                
                                # Combined CG coefficient
                                cg_combined = cg1 * cg2 * cg3 * cg4
                                
                                # Select potential based on isospin projection
                                if mt_total == 0
                                    # np pair (isospin index 1)
                                    Vαx[row_start:row_end, col_start:col_end] += v12[:, :, i, j, 1] * cg_combined
                                else
                                    # pp or nn pair (isospin index 2)
                                    Vαx[row_start:row_end, col_start:col_end] += v12[:, :, i, j, 2] * cg_combined
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return Vαx  # Added return statement
end

 function pot_nucl(α, grid, potname)
    # Compute the nuclear potential matrix
    # Parameters:
    # α: parameters for the Laguerre function
    # grid: grid object containing nx, xi, and other parameters
    # proton m1=+1/2  neutron m2=-1/2
    # for the current function, I only consider the local potential(AV8,NIJM,REID,AV14,AV18), for the non-local potential, one needs to modify this function 
    v12 = zeros(grid.nx, grid.nx, α.nchmax, α.nchmax, 2)  # Initialize potential matrix the last dimension is for the isospin 1 for np pair and 2 for nn(MT<0) or pp pair(MT>0)

    for j in 1:α.nchmax
        for i in 1:α.nchmax
            if checkα(i, j, α)
                # Compute the potential matrix elements
                if Int(α.J12[i]) == 0
                    if α.l[i] != α.l[j]
                        error("error: the channel is not allowed")
                    end 
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir], α.l[i], Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        else
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        end
                    end 
                    
                elseif Int(α.J12[i]) == α.l[i]
                    if α.l[i] != α.l[j]
                        error("error: the channel is not allowed")
                    end
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir], α.l[i], Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        else
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        end
                    end
                else
                    l = [Int(α.J12[i])-1, Int(α.J12[i])+1]
                    for ir in 1:grid.nx  
                        if α.l[i] == Int(α.J12[i]-1) && α.l[j] == Int(α.J12[i]-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            else
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 1]
                            end
                        elseif α.l[i] == Int(α.J12[i]+1) && α.l[j] == Int(α.J12[i]+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 2] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            else
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 2]
                            end
                        elseif α.l[i] == Int(α.J12[i]-1) && α.l[j] == Int(α.J12[i]+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 2] 
                            else
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 2]
                            end
                        elseif α.l[i] == Int(α.J12[i]+1) && α.l[j] == Int(α.J12[i]-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 1]  
                            else
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 1]
                            end
                        end
                    end 
                end 
            end
        end
    end
    
    return v12  
end


 
 function checkα(i,j,α)

    # Check if the channel is allowed
    if (-1)^α.l[i] == (-1)^α.l[j] && Int(α.s12[i]*2) == Int(α.s12[j]*2) && Int(α.J12[i]*2) == Int(α.J12[j]*2)
        return true
    else
        return false
    end
 end 


 function VCOUL_point(R, z12)   # use to compute the Coulomb potential
    # Constants
    e2 = 1.43997  # Coulomb constant in appropriate units
    
    # Calculations
    aux = e2 * z12
    vcoul_point = 0.0
    
    # Early return if z12 is very small
    if (z12 < 1e-4)
        return vcoul_point
    end
    
    # Compute Coulomb potential
    vcoul_point = aux / R
    
    return vcoul_point
end



"""
    cubherm_interp(xold::Vector{Float64}, yold::Vector{Float64}, xnew::Vector{Float64}) -> Vector{Float64}

Perform cubic Hermitian spline interpolation from old grid points to new points.

# Arguments
- `xold`: Array of old grid points
- `yold`: Array of function values at the old grid points
- `xnew`: Array of new points where interpolation is desired

# Returns
- `ynew`: Array of interpolated function values at the new points

The interpolating functions are cubic Hermitian splines. The first derivatives at the grid 
points are calculated from a parabola through the actual grid point and its two neighbors.
For end points, the parabola is taken through the two right or left neighbors.
"""
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
            }
            
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

"""
    cubherm_interp_point(xold::Vector{Float64}, yold::Vector{Float64}, x::Float64) -> Float64

Perform cubic Hermitian spline interpolation for a single point.

# Arguments
- `xold`: Array of old grid points
- `yold`: Array of function values at the old grid points
- `x`: Single point where interpolation is desired

# Returns
- The interpolated function value at point x
"""
function cubherm_interp_point(xold::Vector{Float64}, yold::Vector{Float64}, x::Float64)
    return cubherm_interp(xold, yold, [x])[1]
end



end # end module matrices