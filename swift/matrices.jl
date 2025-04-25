module matrices 
using Kronecker
include("../NNpot/nuclear_potentials.jl")
using .NuclearPotentials

const amu= 931.49432 # MeV
const m=1.008665 # amu
const ħ=197.3269718 # MeV. fm

# 1.008665 amu for neutron  amu=931.49432 MeV


 function T_matrix(α,grid) 
"""
α.nchmax is the maximum number of α channel index, α0 is the α parameter in Laguerre function 
"""
 Tαx = zeros(α.nchmax*grid.nx,α.nchmax*grid.nx)  # Initialize Tαx matrix
 Iy = Matrix{Float64}(I, grid.ny, grid.ny)

 for i in 1:α.nchmax

    T = Tx(grid.nx,grid.xi,grid.α,α.l[i])  
    T .= T .* ħ^2 / m / amu  
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
       T= Tx(grid.ny,yi,α0,α.λ[iα])
       T .= T .* ħ^2 * 0.75 / m / amu

       row_start = (i-1)*grid.ny + 1
       row_end = i*grid.ny
       col_start = (i-1)*grid.ny + 1
       col_end = i*grid.ny
       Ty_matrix[row_start:row_end, col_start:col_end] = T

    end 
 end 


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
                T[i,j] =  (-1.0 / 12. * xi[i]^2 ) * ( xi[i]^2 - 2.0 * (2.0* nx+α+1.0)*xi[i] + α^2 - 4.0 ) - (-1) ^{i-j} / (4 * sqrt(xi[i]*xi[j])) + l*(l+1)/xi[i]^2 
            else
                # Off-diagonal elements
                T[i,j] = (-1.0)^{i-j} * (xi[i]+xi[j]) / (sqrt(xi[i]*xi[j]) * (xi[i]-xi[j])^2 ) - (-1) ^{i-j} / (4 * sqrt(xi[i]*xi[j]))
            end
 
        end
    end


    return T

 end # end function Tx

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
                        v = potential_matrix(potname, grid.xi[ir], α.l[i], α.s12[i], α.J12[i], α.T12[i], 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], α.s12[i], α.J12[i], α.T12[i], 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        else
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], α.s12[i], α.J12[i], α.T12[i], -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        end
                    end 
                    
                elseif Int(α.J12[i]) == α.l[i]
                    if α.l[i] != α.l[j]
                        error("error: the channel is not allowed")
                    end
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir], α.l[i], α.s12[i], α.J12[i], α.T12[i], 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], α.s12[i], α.J12[i], α.T12[i], 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        else
                            v = potential_matrix(potname, grid.xi[ir], α.l[i], α.s12[i], α.J12[i], α.T12[i], -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        end
                    end
                else
                    l = [Int(α.J12[i])-1, Int(α.J12[i])+1]
                    for ir in 1:grid.nx  
                        if α.l[i] == Int(α.J12[i]-1) && α.l[j] == Int(α.J12[i]-1) 
                            v = potential_matrix(potname, grid.xi[ir], l[1], α.s12[i], α.J12[i], α.T12[i], 0)
                            v12[ir, ir, i, j, 1] = v[1, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l[1], α.s12[i], α.J12[i], α.T12[i], 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            else
                                v = potential_matrix(potname, grid.xi[ir], l[1], α.s12[i], α.J12[i], α.T12[i], -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 1]
                            end
                        elseif α.l[i] == Int(α.J12[i]+1) && α.l[j] == Int(α.J12[i]+1) 
                            v = potential_matrix(potname, grid.xi[ir], l[2], α.s12[i], α.J12[i], α.T12[i], 0)
                            v12[ir, ir, i, j, 1] = v[2, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l[2], α.s12[i], α.J12[i], α.T12[i], 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 2] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            else
                                v = potential_matrix(potname, grid.xi[ir], l[2], α.s12[i], α.J12[i], α.T12[i], -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 2]
                            end
                        elseif α.l[i] == Int(α.J12[i]-1) && α.l[j] == Int(α.J12[i]+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, α.s12[i], α.J12[i], α.T12[i], 0)
                            v12[ir, ir, i, j, 1] = v[1, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, α.s12[i], α.J12[i], α.T12[i], 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 2] 
                            else
                                v = potential_matrix(potname, grid.xi[ir], l, α.s12[i], α.J12[i], α.T12[i], -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 2]
                            end
                        elseif α.l[i] == Int(α.J12[i]+1) && α.l[j] == Int(α.J12[i]-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, α.s12[i], α.J12[i], α.T12[i], 0)
                            v12[ir, ir, i, j, 1] = v[2, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, α.s12[i], α.J12[i], α.T12[i], 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 1]  
                            else
                                v = potential_matrix(potname, grid.xi[ir], l, α.s12[i], α.J12[i], α.T12[i], -1) # for nn pair
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


end # end module matrices