# Tiger Info Problem
# ρPOMDP and robust ρPOMDP formulations

struct TigerInfoPOMDP <: IPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    inforeward::Vector{Vector{Float64}}
end

struct TigerInfoRPOMDP <: RIPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    usize::Float64
    inforeward::Vector{Vector{Float64}}
end

const discount_tiger_info = 0.75
const states_tiger_info = [:tigerleft, :tigerright]
const actions_tiger_info = [:listenleft, :listencenter, :listenright]
const observations_tiger_info = [:tigerleft, :tigerright]
const inforeward_tiger_info = [[1.0, -1.0], [-1.0, 1.0]]

TigerInfoPOMDP(alphas) = TigerInfoPOMDP(discount_tiger_info, alphas)
TigerInfoPOMDP() = TigerInfoPOMDP(discount_tiger_info, inforeward_tiger_info)

TigerInfoRPOMDP(err::Float64, alphas::Vector{Vector{Float64}}) = TigerInfoRPOMDP(discount_tiger_info, err, alphas)
TigerInfoRPOMDP() = TigerInfoRPOMDP(discount_tiger_info, 0.025, inforeward_tiger_info)

states(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = states_tiger_info
actions(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = actions_tiger_info
observations(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = observations_tiger_info

n_states(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = 2
n_actions(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = 3
n_observations(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = 2

state_index(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}, s::Symbol) = s == :tigerleft ? 1 : 2
action_index(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}, a::Symbol) = a == :listenleft ? 1 : a == :listencenter ? 2 : 3
observation_index(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}, z::Symbol) = z == :tigerleft ? 1 : 2
obs_index(prob::Union{TigerInfoPOMDP, TigerInfoRPOMDP}, z::Symbol) = observation_index(prob, z)

# start knowing baby is not not hungry
initial_state_distribution(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = SparseCat([:tigerleft, :tigerright], [0.5, 0.5])
initial_belief(::Union{TigerInfoPOMDP, TigerInfoRPOMDP}) = [0.5, 0.5]

# Nominal transition function distributions
const pϵ_tiger_info = 1e-6
const tdist_tiger_info = [SparseCat(states_tiger_info, [1.0, 0.0]) SparseCat(states_tiger_info, [1.0, 0.0]) SparseCat(states_tiger_info, [1.0, 0.0]);
                    SparseCat(states_tiger_info, [0.0, 1.0]) SparseCat(states_tiger_info, [0.0, 1.0]) SparseCat(states_tiger_info, [0.0, 1.0])]

# Nominal transition function array tarray_tiger_info[s,a,s'] = Pr(s' | s ,a)
const tarray_tiger_info = cat(3, [1.0 1.0 1.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 1.0 1.0 1.0]) # P(tigerleft | s, a), P(tigerright | s, a)

# Nominal transitions
transition(prob::TigerInfoPOMDP, s::Symbol, a::Symbol) = tdist_tiger_info[state_index(prob, s), action_index(prob, a)]
transition(prob::TigerInfoPOMDP, s::Symbol, a::Symbol, sp::Symbol) = tarray_tiger_info[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::TigerInfoRPOMDP, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(tarray_tiger_info[si,ai,:] - prob.usize, 0.0 + pϵ_tiger_info)
    pupper = min.(tarray_tiger_info[si,ai,:] + prob.usize, 1.0 - pϵ_tiger_info)
    plower, pupper
end
function transition(prob::TigerInfoRPOMDP, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(tarray_tiger_info[si, ai, spi] - prob.usize, 0.0 + pϵ_tiger_info)
    pupper = min.(tarray_tiger_info[si, ai, spi] + prob.usize, 1.0 - pϵ_tiger_info)
    plower, pupper
end

# Nominal observation function distributions
const odist_tiger_info = [SparseCat(observations_tiger_info, [0.85 0.15]) SparseCat(observations_tiger_info, [0.6 0.4]);
                    SparseCat(observations_tiger_info, [0.9 0.1]) SparseCat(observations_tiger_info, [0.1 0.9]);
                    SparseCat(observations_tiger_info, [0.4 0.6]) SparseCat(observations_tiger_info, [0.15 0.85])]

# Nominal observation function array oarray_tiger_info[a,sp,z] = Pr(z|a,sp)
const oarray_tiger_info = cat(3, [0.85 0.6; 0.9 0.1; 0.4 0.15], [0.15 0.4; 0.1 0.9; 0.6 0.85]) # Pr(tigerleft|a,sp), Pr(tigerright|a,sp)

# Nominal observations
observation(prob::TigerInfoPOMDP, a::Symbol, sp::Symbol) = odist_tiger_info[action_index(prob, a), state_index(prob, sp)]
observation(prob::TigerInfoPOMDP, a::Symbol, sp::Symbol, z::Symbol) = oarray_tiger_info[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::TigerInfoRPOMDP, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(oarray_tiger_info[ai,spi,:] - prob.usize, 0.0 + pϵ_tiger_info)
    pupper = min.(oarray_tiger_info[ai,spi,:] + prob.usize, 1.0 - pϵ_tiger_info)
    plower, pupper
end
function observation(prob::TigerInfoRPOMDP, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(oarray_tiger_info[ai, spi, zi] - prob.usize, 0.0 + pϵ_tiger_info)
    pupper = min.(oarray_tiger_info[ai, spi, zi] + prob.usize, 1.0 - pϵ_tiger_info)
    plower, pupper
end

function dynamics(prob::TigerInfoPOMDP)
    ns = n_states(prob)
    nz = n_observations(prob)
    na = n_actions(prob)
    tarr = states(prob)
    sarr = states(prob)
    aarr = actions(prob)
    zarr = observations(prob)
    p = Array{Float64}(ns,nz,ns,na)
    plower = Array{Float64}(ns,nz,ns,na)
    pupper = Array{Float64}(ns,nz,ns,na)
    for tind in 1:ns, zind in 1:nz, sind in 1:ns, aind in 1:na
        pt = transition(prob, sarr[sind], aarr[aind], tarr[tind])
        pz = observation(prob, aarr[aind], tarr[tind], zarr[zind])
        p[tind, zind, sind, aind] = pz * pt
    end
    p
end

function dynamics(prob::TigerInfoRPOMDP)
    ns = n_states(prob)
    nz = n_observations(prob)
    na = n_actions(prob)
    tarr = states(prob)
    sarr = states(prob)
    aarr = actions(prob)
    zarr = observations(prob)
    p = Array{Float64}(ns,nz,ns,na)
    plower = Array{Float64}(ns,nz,ns,na)
    pupper = Array{Float64}(ns,nz,ns,na)
    for tind in 1:ns, zind in 1:nz, sind in 1:ns, aind in 1:na
        ptl, ptu = transition(prob, sarr[sind], aarr[aind], tarr[tind])
        pzl, pzu = observation(prob, aarr[aind], tarr[tind], zarr[zind])
        plower[tind, zind, sind, aind] = pzl * ptl
        pupper[tind, zind, sind, aind] = pzu * ptu
    end
    plower, pupper
end
function reward(prob::Union{TigerInfoPOMDP, TigerInfoRPOMDP}, b::Vector{Float64}, a::Symbol)
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::Union{TigerInfoPOMDP,TigerInfoRPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,b,a)

function rewardalpha(prob::Union{TigerInfoPOMDP, TigerInfoRPOMDP}, b::Vector{Float64}, a::Symbol)
    rmax = -Inf
    ralpha = nothing
    for α in prob.inforeward
        rnext = dot(α, b)
        if rnext > rmax
            rmax = rnext
            ralpha = copy(α)
        end
    end
    ralpha
end

discount(p::Union{TigerInfoPOMDP,TigerInfoRPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{TigerInfoPOMDP,TigerInfoRPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::TigerInfoPOMDP, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

function generate_sor(prob::TigerInfoRPOMDP, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
