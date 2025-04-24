module matrices 
using Kronecker

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

 function pot_nucl()


 
 function checkα(i,j,α)

    # Check if the channel is allowed
    if (-1)^α.l[i] == (-1)^α.l[j] && α.s12[i] == α.s12[j] && α.J12[i] == α.J12[j]
        return true
    else
        return false
    end
 end 



end # end module matrices