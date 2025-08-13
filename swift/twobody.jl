# this module use to compute the two body bound state for testing the NN force 
module twobodybound 
using LinearAlgebra
include("../NNpot/nuclear_potentials.jl")
using .NuclearPotentials

export bound2b


const amu= 931.49432 # MeV
const mn=1.008665 # amu
const mp=1.007276 # amu 
const μ=mn*mp/(mn+mp) # amu
const ħ=197.3269718 # MeV. fm


mutable struct nch2b # channel index for the three body coupling
    nchmax::Int # maximum number of channels
    s1::Float64
    s2::Float64
    l::Vector{Int}
    s12::Vector{Float64}
    J12 :: Float64
    
    # Constructor with default initialization
    function nch2b()
        new(0, 0.0, 0.0, Int[], Float64[], 0.0)
    end
end


function bound2b(grid, potname)
    """
    α.nchmax is the maximum number of α channel index, α0 is the α parameter in Laguerre function 
    """
    α = channelindex()
    # Initialize matrices
    Tαx = T_matrix(α,grid) 
    V = V_matrix(α,grid,potname) 


    # V=gaussianpot(α,grid)

    # Construct the Hamiltonian matrix
    H = Tαx + V

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigen(H)

    # Extract the bound state energies and wave functions
    bound_energies = []
    bound_wavefunctions = []
    for i in 1:grid.nx*α.nchmax
        if eigenvalues[i] < 0.0
            eigenvec = eigenvectors[:, i]
            norm = sum(abs2.(eigenvec))
            println("Bound state $i: Energy = $(eigenvalues[i]) MeV, Norm = $norm")
            if norm ≠ 1.0
                eigenvec = eigenvec / sqrt(norm)
                println("  Normalized eigenvector to unit norm")
            end
            
            # Compute the wave function with multiple components
            wavefunction = zeros(grid.nx, α.nchmax)
            for ich in 1:α.nchmax
                for j in 1:grid.nx
                    idx = (ich-1)*grid.nx + j
                    wavefunction[j, ich] = grid.ϕx[j] * eigenvec[idx]
                end
            end
            
            push!(bound_energies, eigenvalues[i])
            push!(bound_wavefunctions, wavefunction)
        end
    end
    println("Number of bound states: ", length(bound_energies))
    println("Bound state energies: ", bound_energies)

    return bound_energies, bound_wavefunctions

end 


function V_matrix(α,grid,potname)
    l=[α.l[i] for i in 1:α.nchmax]
    V = zeros(α.nchmax*grid.nx,α.nchmax*grid.nx)  # Initialize V matrix
    
    for ir in 1:grid.nx
        vpot=potential_matrix(potname, grid.xi[ir], l, 1, Int(α.J12), 0, 0) # for np pair
        # println("r=", grid.xi[ir],"vpot=",vpot)
        for i in 1:α.nchmax
            for j in 1:α.nchmax
                V[(i-1)*grid.nx+ir, (j-1)*grid.nx+ir] = vpot[i,j]
            end 
        end 
    end
    return V 
end 


function gaussianpot(α,grid)
    # Define the Gaussian potential parameters
    V0 = -72.15  # Depth of the potential well
    r0 = 0  # Range of the potential
    a = 1.484  # Width of the potential
    V = zeros(α.nchmax*grid.nx,α.nchmax*grid.nx)
    # Define the Gaussian potential function
    for ir in 1:grid.nx
        r = grid.xi[ir]
        V[ir,ir] = V0 * exp(-((r - r0)^2) /  a^2)
    end 


    return V

end 



function T_matrix(α,grid) 
    """
    α.nchmax is the maximum number of α channel index, α0 is the α parameter in Laguerre function 
    """

     Tαx = zeros(α.nchmax*grid.nx,α.nchmax*grid.nx)  # Initialize Tαx matrix
     
     for i in 1:α.nchmax
        T = Tx(grid.nx,grid.xx,grid.α,α.l[i])  
        T .= T .* ħ^2 / (2.0 * μ * amu* grid.hsx^2)  # Scale the T matrix
        row_start = (i-1)*grid.nx + 1
        row_end = i*grid.nx
        col_start = (i-1)*grid.nx + 1
        col_end = i*grid.nx
        Tαx[row_start:row_end, col_start:col_end] = T
     end 
    
     return Tαx
 end 


function channelindex()
    # Initialize the channel index
    α = nch2b()
    α.s1 = 0.5
    α.s2 = 0.5
    α.J12 = 1.0 
    for l in 0:2
        if (-1)^l == -1
            continue
        end
        for ns in Int(2*( α.s1- α.s2)):2:Int(2*( α.s1+ α.s2))
            s12 = ns/2.0
            for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))  # Fixed min/max calculation
                if nJ12 != Int(2*α.J12)
                    continue
                end
                α.nchmax += 1
            end 
        end 
    end 

    println("For J=",α.J12," parity=+", " Number of channels: ", α.nchmax)

    α.l = zeros(Int,  α.nchmax)
    α.s12 = zeros(Float64, α.nchmax)
    nch=0
    for l in 0:2
        if (-1)^l == -1
            continue
        end
        for ns in Int(2*( α.s1- α.s2)):2:Int(2*( α.s1+ α.s2))
            s12 = ns/2.0
            for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))  # Fixed min/max calculation
                if nJ12 != Int(2*α.J12)
                    continue
                end
                nch += 1
                α.l[nch] = l
                α.s12[nch] = s12
                
            end 
        end 
    end
    return α
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

                # T[i,j] = (-1.0 / (12.0 * xi[i]^2)) * (xi[i]^2 - 2.0 * (2.0 * nx + α0 + 1.0) * xi[i] + α0^2 - 4.0) + 
                #          l * (l + 1) / xi[i]^2
            else
                # Off-diagonal elements 
                T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) - 
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j]))


                # T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) 
            end
        end
    end


    return T

 end # end function Tx

end 