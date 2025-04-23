module matrices 
using Kronecker

const amu= 931.49432 # MeV
const m=1.008665 # amu
const ħ=197.3269718 # MeV. fm

# 1.008665 amu for neutron  amu=931.49432 MeV


 function T_matrix(nα,nx,ny,xi,yi,α,α0) 
"""
nα is the maximum number of α channel index, α0 is the α parameter in Laguerre function 
"""
 Tαx = zeros(nα*nx,nα*nx)  # Initialize Tαx matrix
 Iy = Matrix{Float64}(I, ny, ny)

 for i in 1:nα

    T = Tx(nx,xi,α0,α.l[i])  
    T .= T .* ħ^2 / m / amu  
    row_start = (i-1)*nx + 1
    row_end = i*nx
    col_start = (i-1)*nx + 1
    col_end = i*nx
    Tαx[row_start:row_end, col_start:col_end] = T

 end 

 Tx_matrix = Tαx ⊗ Iy
 
 Ty_matrix = zeros(nα*nx*ny,nα*nx*ny)  # Initialize Ty_matrix
 i=0
 for iα in 1:nα
    for ix in 1:nx 
       i += 1
       T= Tx(ny,yi,α0,α.λ[iα])
       T .= T .* ħ^2 * 0.75 / m / amu

       row_start = (i-1)*ny + 1
       row_end = i*ny
       col_start = (i-1)*ny + 1
       col_end = i*ny
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



end # end module matrices