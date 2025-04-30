module NuclearPotentials

using Libdl
export potential_matrix

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

"""
Call the AV18 potential with proper parameters

Arguments:
- lpot: switch for potential choice
  - Argonne models:
    - 1: av18
    - 2: av8'
    - 3: av6'
    - 4: av4'
    - 5: avx'
    - 6: av2'
    - 7: av1'
    - 8: modified av8'
  - Super-Soft Core models:
    - 101: sscc v14
    - 102: sscc v8'
    - 108: modified sscc v8'
- l: orbital angular momentum of pair (0,1,2,...)
- s: total spin of pair (0 or 1)
- j: total angular momentum of pair (0,1,2,...)
- t: total isospin of pair (0 or 1)
- t1z: isospin of particle 1 (1 for p, -1 for n)
- t2z: isospin of particle 2 (1 for p, -1 for n)
- r: separation in fm
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
Map potential type string to lpot parameter value
"""
function potential_type_to_lpot(potential_type::String)::Int
    potential_map = Dict(
        "AV18" => 1,       # Argonne v18
        "AV8" => 2,        # Argonne v8'
        "AV6" => 3,        # Argonne v6'
        "AV4" => 4,        # Argonne v4'
        "AVX" => 5,        # Argonne vX'
        "AV2" => 6,        # Argonne v2'
        "AV1" => 7,        # Argonne v1'
        "AV8M" => 8,       # Modified Argonne v8'
        "SSCC_V14" => 101, # Super-Soft Core (C) v14
        "SSCC_V8" => 102,  # Super-Soft Core (C) v8'
        "SSCC_V8M" => 108  # Modified Super-Soft Core (C) v8'
    )
    
    return get(potential_map, uppercase(potential_type), 1)  # Default to AV18 if not found
end

"""
Calculate the potential matrix for given parameters
"""
function potential_matrix(
    potential_type::String,
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
    
    # Convert tz to nucleon isospins for Argonne potentials
    # For simplicity, we're using a convention here:
    # t1z = -1 (neutron), t2z = 1 (proton) for NP
    # t1z = -1 (neutron), t2z = -1 (neutron) for NN
    # t1z = 1 (proton), t2z = 1 (proton) for PP
    t1z, t2z = -1, -1  # Default to NN
    if tz == 0
        t1z, t2z = -1, 1  # NP
    elseif tz == 1
        t1z, t2z = 1, 1   # PP
    end
    
    # Calculate potential matrix elements
    for ia in 1:n_channels
        for ib in 1:n_channels
            l_ia = angular_momenta[ia]
            l_ib = angular_momenta[ib]
            
            # For Argonne and SSCC potentials
            if r < 140.0  # Distance cutoff
                lpot = potential_type_to_lpot(potential_type)
                
                if ia == ib  # Diagonal element
                    potential[ia, ib] = call_av18(lpot, l_ia, s, j, t, t1z, t2z, r)[1, 1]
                else  # Off-diagonal element (coupled channels)
                    # Using proper l values for each channel
                    vv2 = call_av18(lpot, min(l_ia, l_ib), s, j, t, t1z, t2z, r)
                    potential[ia, ib] = vv2[1, 2]
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