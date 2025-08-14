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
    
    println("\n" * "="^60)
    println("           TWO-BODY BOUND STATE ANALYSIS")
    println("="^60)
    
    bound_count = 0
    for i in 1:grid.nx*α.nchmax
        if eigenvalues[i] < 0.0
            bound_count += 1
            eigenvec = eigenvectors[:, i]
            norm = sum(abs2.(eigenvec))
            
            if norm ≠ 1.0
                eigenvec = eigenvec / sqrt(norm)
            end
            
            # Compute the wave function with multiple components
            wavefunction = zeros(grid.nx, α.nchmax)
            for ich in 1:α.nchmax
                for j in 1:grid.nx
                    idx = (ich-1)*grid.nx + j
                    wavefunction[j, ich] = grid.ϕx[j] * eigenvec[idx]
                end
            end
            
            # Calculate component probabilities
            component_norms = zeros(α.nchmax)
            for ich in 1:α.nchmax
                component_norms[ich] = sum(abs2.(wavefunction[:, ich]) .* grid.dxi)
            end
            
            # Normalize component probabilities
            total_norm = sum(component_norms)
            component_probs = component_norms / total_norm * 100
            
            # Print detailed bound state information
            println("\nBound State #$bound_count:")
            println("  Binding Energy: $(round(eigenvalues[i], digits=6)) MeV")
            println("  Total J^π = $(Int(α.J12))⁺")
            println("\n  Channel Composition:")
            
            for ich in 1:α.nchmax
                l_val = α.l[ich]
                s_val = α.s12[ich]
                # Determine spectroscopic notation
                if l_val == 0
                    l_notation = "S"
                elseif l_val == 1
                    l_notation = "P"
                elseif l_val == 2
                    l_notation = "D"
                elseif l_val == 3
                    l_notation = "F"
                else
                    l_notation = "L=$l_val"
                end
                
                # Format as 2S+1L_J notation
                spectro_notation = "$(Int(2*s_val+1))$(l_notation)₁"
                
                println("    Channel $ich: $spectro_notation (l=$l_val, s=$(s_val)) - $(round(component_probs[ich], digits=2))%")
            end
            
            # Calculate D-state probability specifically
            d_state_prob = 0.0
            for ich in 1:α.nchmax
                if α.l[ich] == 2  # D-state has l=2
                    d_state_prob += component_probs[ich]
                end
            end
            
            println("\n  D-state Probability: $(round(d_state_prob, digits=3))%")
            println("  S-state Probability: $(round(100.0 - d_state_prob, digits=3))%")
            
            push!(bound_energies, eigenvalues[i])
            push!(bound_wavefunctions, wavefunction)
        end
    end
    
    println("\n" * "="^60)
    println("SUMMARY: Found $bound_count bound state(s)")
    if bound_count > 0
        println("Binding energies (MeV): ", [round(e, digits=6) for e in bound_energies])
    end
    println("="^60)

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

    println("\nTwo-body channel configuration:")
    println("  Total angular momentum J = ", α.J12)
    println("  Parity = +")
    println("  Number of channels: ", α.nchmax)

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
                
                # Print channel information
                l_notation = l == 0 ? "S" : l == 1 ? "P" : l == 2 ? "D" : "L=$l"
                spectro_notation = "$(Int(2*s12+1))$(l_notation)₁"
                println("    Channel $nch: $spectro_notation (l=$l, s=$(s12))")
                
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