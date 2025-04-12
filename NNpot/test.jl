# Import your module
include("nuclear_potentials.jl")
using .NuclearPotentials

"""
Example usage function
"""
function example()
    # Example: Calculate the deuteron potential (J=1, S=1, T=0) with AV18
    angular_momenta = [0, 2]  # S and D waves
    s = 1
    j = 1
    t = 0
    tz = 0  # np system
    r = 1.0  # fm
    
    potential = potential_matrix(AV18, r, angular_momenta, s, j, t, tz)
    println("Deuteron potential at r = $r fm:")
    println(potential)
end



# Run the test
example()