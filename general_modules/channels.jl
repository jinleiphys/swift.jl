module channels
using Printf
# export lmax, lmin, λmax, λmin, s1, s2, s3, Jmin, Jmax, t1, t2, t3, Tmin, Tmax, MT
export  α3b, α2b

# channel parameters
# lmax = 0 # maximum l
# lmin = 0 # minimum l
# λmax = 0 # maximum λ
# λmin = 0 # minimum λ
# s1 = 0.0 # spin of particle 1
# s2 = 0.0 # spin of particle 2
# s3 = 0.0 # spin of particle 3
# # for bound state J=Jmin=Jmax
# Jmin = 0.0 # minimum J
# Jmax = 0.0 # maximum J
# t1 = 0.0 # isospin of particle 1
# t2 = 0.0 # isospin of particle 2
# t3 = 0.0 # isospin of particle 3
# Tmin = 0.0 # minimum T
# Tmax = 0.0 # maximum T
# MT = 0.0 # third component of T, fix for the given system by charge conservation

mutable struct nch2b # channel index for the two body coupling
    nchmax::Int # maximum number of channels
    l::Vector{Int}
    s12::Vector{Float64}
    J12::Vector{Float64}
    T12::Vector{Float64}
    
    # Constructor with default initialization
    function nch2b()
        new(0, Int[], Float64[], Float64[], Float64[])
    end
end

mutable struct nch3b # channel index for the three body coupling
    nchmax::Int # maximum number of channels
    s1::Float64
    s2::Float64
    s3::Float64
    t1::Float64
    t2::Float64
    t3::Float64
    l::Vector{Int}
    s12::Vector{Float64}
    J12::Vector{Float64}
    λ::Vector{Int}
    J3::Vector{Float64}
    J::Float64
    T12::Vector{Float64}
    T::Vector{Float64}
    MT :: Float64
    α2b::nch2b # two-body channel coupling data
    α2bindex::Vector{Int} # mapping from α3b channel index to α2b channel index
    
    # Constructor with default initialization
    function nch3b()
        new(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Int[], Float64[], Float64[], Int[], Float64[], 0.0, Float64[], Float64[], -0.5, nch2b(), Int[])
    end
end


# Two-body channel coupling |(l_{12} (s_1 s_2) s_{12}) J_{12}; (t_1 t_2) T_{12}\rangle
function α2b(fermion::Bool, 
    lmax::Int, lmin::Int, 
    s1::Float64, s2::Float64, 
    t1::Float64, t2::Float64, 
    j2bmax::Float64, parity_pair::Union{Int,Nothing}=nothing)
    
    α = nch2b()
    
    # First pass to count channels
    nch_count = 0
    
    for l in lmin:lmax
        # Check parity_pair constraint for the two-body subsystem only if defined
        if parity_pair !== nothing && (-1)^l != parity_pair
            continue
        end
        for ns in Int(2*(s1-s2)):2:Int(2*(s1+s2))
            s12 = ns/2.0
            for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))
                J12 = nJ12/2.0
                if J12 > j2bmax
                    continue
                end
                for nT12 in Int(2*abs(t1-t2)):2:Int(2*(t1+t2))
                    T12 = nT12/2.0
                    if (-1)^(l+s12+T12) != -1 && fermion
                        continue
                    end
                    nch_count += 1
                end
            end
        end
    end
    
    # Allocate arrays with the correct size
    α.nchmax = nch_count
    α.l = zeros(Int, nch_count)
    α.s12 = zeros(Float64, nch_count)
    α.J12 = zeros(Float64, nch_count)
    α.T12 = zeros(Float64, nch_count)
    
    # Second pass to fill the channels
    ich = 0
    
    for l in lmin:lmax
        # Check parity_pair constraint for the two-body subsystem only if defined
        if parity_pair !== nothing && (-1)^l != parity_pair
            continue
        end
        for ns in Int(2*(s1-s2)):2:Int(2*(s1+s2))
            s12 = ns/2.0
            for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))
                J12 = nJ12/2.0
                if J12 > j2bmax
                    continue
                end
                for nT12 in Int(2*abs(t1-t2)):2:Int(2*(t1+t2))
                    T12 = nT12/2.0
                    if (-1)^(l+s12+T12) != -1 && fermion
                        continue
                    end
                    ich += 1
                    α.l[ich] = l
                    α.s12[ich] = s12
                    α.J12[ich] = J12
                    α.T12[ich] = T12
                end
            end
        end
    end
    
    return α
end

# channel index of the three body channels
#|(l_{12} (s_1 s_2) s_{12}) J_{12}, (\lambda_3 s_3) J_3, J; (t_1 t_2) T_{12}, t_3, T M_T\rangle.
# Function to count and create three-body channels
function α3b(fermion::Bool,J::Float64, T::Float64, parity::Int, 
    lmax::Int, lmin::Int, 
    λmax::Int, λmin::Int, 
    s1::Float64, s2::Float64, s3::Float64, 
    t1::Float64, t2::Float64, t3::Float64, 
    MT::Float64, j2bmax::Float64, parity_pair::Union{Int,Nothing}=nothing)
    α = nch3b()
    α.s1 = s1
    α.s2 = s2
    α.s3 = s3
    α.t1 = t1
    α.t2 = t2
    α.t3 = t3
    α.J = J
    α.MT = MT

    
    # First pass to count channels
    nch_count = 0
    
    # Get two-body channels using α2b
    α.α2b = α2b(fermion, lmax, lmin, s1, s2, t1, t2, j2bmax, parity_pair)
    α2b_channels = α.α2b
    
    for i2b in 1:α2b_channels.nchmax
        l = α2b_channels.l[i2b]
        s12 = α2b_channels.s12[i2b]
        J12 = α2b_channels.J12[i2b]
        T12 = α2b_channels.T12[i2b]
        
        for λ in λmin:λmax
            if (-1)^(l+λ) != parity
                continue
            end
            for nJ3 in Int(2*abs(λ-s3)):2:Int(2*(λ+s3))  # Fixed min/max calculation
                J3 = nJ3/2.0
                for nJ in Int(2*abs(J12-J3)):2:Int(2*(J12+J3))  # Fixed min/max calculation
                    if nJ != Int(2*J)
                        continue
                    end
                    for nT in Int(2*abs(T12-t3)):2:Int(2*(T12+t3))  # Fixed min/max calculation
                        T_current = nT/2.0
                        if T >= 0
                            # T is defined, use the specified value
                            if nT != Int(2*T)
                                continue
                            end
                        end
                        # check if MT is in the range of -T_current to T_current
                        if abs(MT) > T_current
                            continue
                        end
                        
                        nch_count += 1
                    end
                end
            end
        end
    end
    

    println("For J=",J, " T=",T," parity=",parity, " parity_pair=",parity_pair, " Number of channels: ", nch_count)
    # Now allocate arrays with the correct size
    α.nchmax = nch_count
    α.l = zeros(Int, nch_count)
    α.s12 = zeros(Float64, nch_count)
    α.J12 = zeros(Float64, nch_count)
    α.λ = zeros(Int, nch_count)
    α.J3 = zeros(Float64, nch_count)
    α.T12 = zeros(Float64, nch_count)
    α.T = zeros(Float64, nch_count)
    α.α2bindex = zeros(Int, nch_count)

    
    # Second pass to fill the channels
    ich = 0
    
    if nch_count > 0  # Only do second pass if we have channels
        println("---The coupling coefficients are")
        println(" α3b| α2b |( l ( s1 s2 ) s12 ) J12 ( λ s3 ) J3 ,   J; ( t1 t2 ) T12 , t3 , T >")
        
        for i2b in 1:α2b_channels.nchmax
            l = α2b_channels.l[i2b]
            s12 = α2b_channels.s12[i2b]
            J12 = α2b_channels.J12[i2b]
            T12 = α2b_channels.T12[i2b]
            
            for λ in λmin:λmax
                if (-1)^(l+λ) != parity
                    continue
                end
                for nJ3 in Int(2*abs(λ-s3)):2:Int(2*(λ+s3))
                    J3 = nJ3/2.0
                    for nJ in Int(2*abs(J12-J3)):2:Int(2*(J12+J3))
                        if nJ != Int(2*J)
                            continue
                        end
                        for nT in Int(2*abs(T12-t3)):2:Int(2*(T12+t3))
                            T_current = nT/2.0
                            if T >= 0
                                # T is defined, use the specified value
                                if nT != Int(2*T)
                                    continue
                                end
                            end
                            # check if MT is in the range of -T_current to T_current
                            if abs(MT) > T_current
                                continue
                            end
                            ich += 1
                            α.l[ich] = l
                            α.s12[ich] = s12
                            α.J12[ich] = J12
                            α.λ[ich] = λ
                            α.J3[ich] = J3
                            α.T12[ich] = T12
                            α.T[ich] = T_current
                            α.α2bindex[ich] = i2b
                            print_channel_info(ich, i2b, l, s1, s2, s12, J12, λ, s3, J3, J,t1,t2,T12, t3, T_current)
                        end
                    end
                end
            end
        end
    end # end if 
    

    return α
end

# Function to print channel information
function print_channel_info(ich, i2b, l, s1, s2, s12, J12, λ, s3, J3, J,t1,t2,T12, t3, T)
    @printf("%3d |%4d |(%2d (%2.1f %2.1f) %2.1f) %4.1f (%2d %2.1f) %3.1f, %3.1f; (%2.1f %2.1f) %2.1f, %2.1f, %2.1f > \n",
            ich, i2b, l, s1, s2, s12, J12, λ, s3, J3, J, t1, t2, T12, t3, T)
end

end # module