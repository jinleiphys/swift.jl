module NuclearPotentials

using Libdl
export potential_matrix

# """
# Enum for available potential models
# """
# @enum PotentialType begin
#     AV8 = 1
#     NIJM = 2
#     REID = 3
#     AV14 = 4
#     AV18 = 5
# end

# """
# Enum for nucleon-nucleon channel types
# """
# @enum ChannelType begin
#     NN = 1  # neutron-neutron
#     NP = 2  # neutron-proton
#     PP = 3  # proton-proton
# end

# Load the Fortran library with potentials
const libpot = Libdl.dlopen(joinpath(@__DIR__, "libpotentials.dylib"))

# Function to list all symbols in the library (useful for debugging)
function list_symbols(lib)
    symbols = []
    Libdl.dllist() do handle, path
        if handle == lib
            for sym in Libdl.dlsym_e(handle)
                push!(symbols, sym)
            end
        end
        return false
    end
    return symbols
end

# Try to find symbols with different name mangling patterns
function find_symbol(lib, base_name)
    possible_names = [
        base_name,
        "_$base_name",
        "__$base_name",
        "___$base_name",
        "_$base_name"
    ]
    
    for name in possible_names
        sym = Libdl.dlsym_e(lib, name)
        if sym != C_NULL
            return sym
        end
    end
    
    # If we can't find the symbol with standard patterns, list available symbols
    println("Warning: Could not find symbol for $base_name")
    println("Available symbols:")
    for sym in list_symbols(lib)[1:min(10, length(list_symbols(lib)))]
        println("  $sym")
    end
    
    error("Symbol $base_name not found in library")
end

# Define the function pointers to the Fortran subroutines - with robust symbol finding
const av18pw_ptr = find_symbol(libpot, "av18_MOD_av18pw")
const potn_ptr = find_symbol(libpot, "nijm_reid_potentials_MOD_potn")
const pot_c_ptr = find_symbol(libpot, "nijm_reid_potentials_MOD_pot_c")
const pot_r_ptr = find_symbol(libpot, "nijm_reid_potentials_MOD_pot_r")
const pot_c_r_ptr = find_symbol(libpot, "nijm_reid_potentials_MOD_pot_c_r")
const pot_v14_ptr = find_symbol(libpot, "argonne_v14_potential_MOD_pot_v14")

"""
Convert total isospin projection to channel type string for Nijmegen/Reid potentials
"""
function tz_to_channel_type(tz::Int)::String
    if tz == -1
        return "NN"
    elseif tz == 0
        return "NP"
    elseif tz == 1
        return "PP"
    else
        return "NP"  # Default case
    end
end

"""
Call the AV18 potential with proper parameters
"""
function call_av18(lpot::Int, l::Int, s::Int, j::Int, t::Int, t1z::Int, t2z::Int, r::Float64)
    # Create output array for the potential
    vpw = zeros(Float64, 2, 2)
    
    # Call the Fortran function using ccall
    # The signature matches: av18pw(lpot, l, s, j, t, t1z, t2z, r, vpw)
    ccall(
        av18pw_ptr,
        Cvoid,
        (Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, 
         Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Float64}, Ref{Float64}),
        Int32(lpot), Int32(l), Int32(s), Int32(j), 
        Int32(t), Int32(t1z), Int32(t2z), r, vpw
    )
    
    return vpw
end

"""
Call the Nijmegen potential
"""
function call_nijmegen(l::Int, s::Int, j::Int, type::String, r::Float64)
    # Call the Fortran function using ccall for diagonal elements
    result = ccall(
        potn_ptr,
        Float64,
        (Ref{Int32}, Ref{Int32}, Ref{Int32}, Cstring, Ref{Float64}),
        Int32(l), Int32(s), Int32(j), type, r
    )
    
    return result
end

"""
Call the Nijmegen potential coupling term
"""
function call_nijmegen_coupling(l::Int, s::Int, j::Int, type::String, r::Float64)
    # Call the Fortran function using ccall for off-diagonal elements
    result = ccall(
        pot_c_ptr,
        Float64,
        (Ref{Int32}, Ref{Int32}, Ref{Int32}, Cstring, Ref{Float64}),
        Int32(l), Int32(s), Int32(j), type, r
    )
    
    return result
end

"""
Call the Reid93 potential
"""
function call_reid(l::Int, s::Int, j::Int, type::String, r::Float64)
    # Call the Fortran function using ccall for diagonal elements
    result = ccall(
        pot_r_ptr,
        Float64,
        (Ref{Int32}, Ref{Int32}, Ref{Int32}, Cstring, Ref{Float64}),
        Int32(l), Int32(s), Int32(j), type, r
    )
    
    return result
end

"""
Call the Reid93 potential coupling term
"""
function call_reid_coupling(l::Int, s::Int, j::Int, type::String, r::Float64)
    # Call the Fortran function using ccall for off-diagonal elements
    result = ccall(
        pot_c_r_ptr,
        Float64,
        (Ref{Int32}, Ref{Int32}, Ref{Int32}, Cstring, Ref{Float64}),
        Int32(l), Int32(s), Int32(j), type, r
    )
    
    return result
end

"""
Call the AV14 potential
"""
function call_av14(r::Float64)
    # Create a large enough array to hold the potential matrix
    # The AV14 potential is returned as potential(0:5, 0:5, 0:1, 0:4)
    vvv = zeros(Float64, 6, 6, 2, 5)
    
    # Call the Fortran function
    ccall(
        pot_v14_ptr,
        Cvoid,
        (Ref{Float64}, Ref{Float64}),
        r, vvv
    )
    
    return vvv
end

"""
Calculate the potential matrix for given parameters
"""
function potential_matrix(
    potential_type::String,  # Changed from PotentialType to String
    r::Float64,
    angular_momenta::Vector{Int},  # l values for each channel
    s::Int,                        # total spin
    j::Int,                        # total angular momentum
    t::Int,                        # total isospin
    tz::Int                        # isospin projection
)
    # Number of channels
    n_channels = length(angular_momenta)
    
    # Initialize potential matrix
    potential = zeros(Float64, n_channels, n_channels)
    
    # Return zero if r is zero
    if r == 0.0
        return potential
    end
    
    # Determine channel type based on tz
    channel_type = tz_to_channel_type(tz)
    
    # Calculate potential matrix elements
    for ia in 1:n_channels
        for ib in 1:n_channels
            l_ia = angular_momenta[ia]
            l_ib = angular_momenta[ib]
            
            # Choose the appropriate potential model (using string comparison)
            if potential_type == "AV8"
                if r < 140.0  # Distance cutoff
                    if ia == ib  # Diagonal element
                        # For AV8, lpot=2 selects the v8' version
                        potential[ia, ib] = call_av18(2, min(l_ia, l_ib), s, j, t, -1, 1, r)[1, 1]
                    else  # Off-diagonal element (coupled channels)
                        vv2 = call_av18(2, min(l_ia, l_ib), s, j, t, -1, 1, r)
                        potential[ia, ib] = vv2[1, 2]
                    end
                end
            elseif potential_type == "NIJM"
                if ia == ib  # Diagonal element
                    # Important: Nijmegen takes 2*l, 2*s, 2*j as arguments
                    potential[ia, ib] = call_nijmegen(2*l_ia, 2*s, 2*j, channel_type, r)
                else  # Off-diagonal element (coupled channels)
                    potential[ia, ib] = call_nijmegen_coupling(2*min(l_ia, l_ib), 2*s, 2*j, channel_type, r)
                end
            elseif potential_type == "REID"
                if ia == ib  # Diagonal element
                    # Important: Reid93 also takes 2*l, 2*s, 2*j as arguments
                    potential[ia, ib] = call_reid(2*l_ia, 2*s, 2*j, channel_type, r)
                else  # Off-diagonal element (coupled channels)
                    potential[ia, ib] = call_reid_coupling(2*min(l_ia, l_ib), 2*s, 2*j, channel_type, r)
                end
            elseif potential_type == "AV14"
                vvv = call_av14(r)
                # Julia is 1-indexed, but Fortran starts from 0, so we add 1
                potential[ia, ib] = vvv[l_ia+1, l_ib+1, s+1, j+1]
            elseif potential_type == "AV18"
                if r < 140.0  # Distance cutoff
                    if ia == ib  # Diagonal element
                        # For AV18, lpot=1 selects the full v18 version
                        potential[ia, ib] = call_av18(1, min(l_ia, l_ib), s, j, t, -1, 1, r)[1, 1]
                    else  # Off-diagonal element (coupled channels)
                        vv2 = call_av18(1, min(l_ia, l_ib), s, j, t, -1, 1, r)
                        potential[ia, ib] = vv2[1, 2]
                    end
                end
            end
        end
    end
    
    return potential
end

"""
Utility function to debug library symbol issues
"""
function debug_library_symbols()
    # On Linux/Unix systems, we can use nm command to list symbols
    try
        lib_path = Libdl.dlpath(libpot)
        println("Library path: $lib_path")
        run(`nm $lib_path`)
    catch e
        println("Could not run nm command: $e")
        println("Try running in terminal: nm $(Libdl.dlpath(libpot))")
    end
end

end # module