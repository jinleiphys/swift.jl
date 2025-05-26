module threebodybound 
 include("matrices.jl")
 using .matrices
 using LinearAlgebra
 export ThreeBody_Bound


 function ThreeBody_Bound(α, grid)

    Rxy=Rxy_matrix(α, grid)
    T=T_matrix(α,grid) 
    V=V_matrix(α, grid, "MT")
    H=V*Rxy+T+V

    H=V*Rxy

    eigenvalues, eigenvectors = eigen(H)

    println("Eigenvalues: ", eigenvalues)


    

    # Extract the bound state energies and wave functions
    bound_states = []
    for i in 1:grid.nx
        if real(eigenvalues[i]) < 0.0
            push!(bound_states, (eigenvalues[i], eigenvectors[:, i]))
        end
    end
    println("Number of bound states: ", length(bound_states))
    println("Bound state energies: ", [state[1] for state in bound_states])

    return bound_states

 end 

end 