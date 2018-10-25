# Baby Info Problem
# ρPOMDP and robust ρPOMDP formulations

struct BabyInfoPOMDP <: IPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    inforeward::Vector{Vector{Float64}}
end

struct BabyInfoRPOMDP <: RIPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    usize::Float64
    inforeward::Vector{Vector{Float64}}
end

const discount_baby_info = 0.75
const states_baby_info = [:hungry, :full]
const actions_baby_info = [:check, :monitor, :nothing]
const observations_baby_info = [:quiet, :crying, :yelling]
const inforeward_baby_info = [[1.0, -1.0], [-1.0, 1.0]]

BabyInfoPOMDP(alphas) = BabyInfoPOMDP(discount_baby_info, alphas)
BabyInfoPOMDP() = BabyInfoPOMDP(discount_baby_info, inforeward_baby_info)

BabyInfoRPOMDP(err::Float64, alphas::Vector{Vector{Float64}}) = BabyInfoRPOMDP(discount_baby_info, err, alphas)
BabyInfoRPOMDP() = BabyInfoRPOMDP(discount_baby_info, 0.025, inforeward_baby_info)

states(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = states_baby_info
actions(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = actions_baby_info
observations(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = observations_baby_info

n_states(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = 2
n_actions(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = 3
n_observations(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = 3

state_index(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}, s::Symbol) = s == :hungry ? 1 : 2
action_index(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}, a::Symbol) = a == :check ? 1 : a == :monitor ? 2 : 3
observation_index(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}, z::Symbol) = z == :quiet ? 1 : z == :crying ? 2 : 3
obs_index(prob::Union{BabyInfoPOMDP, BabyInfoRPOMDP}, z::Symbol) = observation_index(prob, z)

initial_state_distribution(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = SparseCat([:hungry, :full], [0.0, 1.0])
initial_belief(::Union{BabyInfoPOMDP, BabyInfoRPOMDP}) = [0.0, 1.0]

# Nominal transition function distributions
const pϵ_baby_info = 1e-6
const tdist_baby_info = [SparseCat(states_baby_info, [0.2, 0.8]) SparseCat(states_baby_info, [0.2, 0.8]) SparseCat(states_baby_info, [0.2, 0.8]);
                    SparseCat(states_baby_info, [0.1, 0.9]) SparseCat(states_baby_info, [0.1, 0.9]) SparseCat(states_baby_info, [0.1, 0.9])]

# Nominal transition function array tarray_baby_info[s,a,s'] = Pr(s' | s ,a)
const tarray_baby_info = cat(3, [0.2 0.2 0.2; 0.1 0.1 0.1], [0.8 0.8 0.8; 0.9 0.9 0.9]) # P(babyleft | s, a), P(babyright | s, a)

# Nominal transitions
transition(prob::BabyInfoPOMDP, s::Symbol, a::Symbol) = tdist_baby_info[state_index(prob, s), action_index(prob, a)]
transition(prob::BabyInfoPOMDP, s::Symbol, a::Symbol, sp::Symbol) = tarray_baby_info[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::BabyInfoRPOMDP, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(tarray_baby_info[si,ai,:] - prob.usize, 0.0 + pϵ_baby_info)
    pupper = min.(tarray_baby_info[si,ai,:] + prob.usize, 1.0 - pϵ_baby_info)
    plower, pupper
end
function transition(prob::BabyInfoRPOMDP, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(tarray_baby_info[si, ai, spi] - prob.usize, 0.0 + pϵ_baby_info)
    pupper = min.(tarray_baby_info[si, ai, spi] + prob.usize, 1.0 - pϵ_baby_info)
    plower, pupper
end

# Nominal observation function distributions
const odist_baby_info = [SparseCat(observations_baby_info, [0.3 0.6 0.1]) SparseCat(observations_baby_info, [0.8 0.1 0.1]);
                    SparseCat(observations_baby_info, [0.3 0.4 0.3]) SparseCat(observations_baby_info, [0.6 0.2 0.2]);
                    SparseCat(observations_baby_info, [0.75 0.3 0.05]) SparseCat(observations_baby_info, [0.9 0.05 0.05])]

# Nominal observation function array oarray_baby_info[a,sp,z] = Pr(z|a,sp)
const oarray_baby_info = cat(3, [0.3 0.8; 0.3 0.6; 0.75 0.9], [0.6 0.1; 0.4 0.2; 0.25 0.05], [0.1 0.1; 0.3 0.2; 0.05 0.05]) # Pr(babyleft|a,sp), Pr(babyright|a,sp)

# Nominal observations
observation(prob::BabyInfoPOMDP, a::Symbol, sp::Symbol) = odist_baby_info[action_index(prob, a), state_index(prob, sp)]
observation(prob::BabyInfoPOMDP, a::Symbol, sp::Symbol, z::Symbol) = oarray_baby_info[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::BabyInfoRPOMDP, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(oarray_baby_info[ai,spi,:] - prob.usize, 0.0 + pϵ_baby_info)
    pupper = min.(oarray_baby_info[ai,spi,:] + prob.usize, 1.0 - pϵ_baby_info)
    plower, pupper
end
function observation(prob::BabyInfoRPOMDP, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(oarray_baby_info[ai, spi, zi] - prob.usize, 0.0 + pϵ_baby_info)
    pupper = min.(oarray_baby_info[ai, spi, zi] + prob.usize, 1.0 - pϵ_baby_info)
    plower, pupper
end

function dynamics(prob::BabyInfoPOMDP)
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

function dynamics(prob::BabyInfoRPOMDP)
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
function reward(prob::Union{BabyInfoPOMDP, BabyInfoRPOMDP}, b::Vector{Float64}, a::Symbol)
    beliefreward = -Inf
    actionreward = a == :checkin ? -0.2 : a == :monitor ? -0.1 : -0.05
    for α in prob.inforeward
        beliefreward = max(beliefreward, dot(α, b))
    end
    beliefreward + actionreward
end
reward(prob::Union{BabyInfoPOMDP,BabyInfoRPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,b,a)

function rewardalpha(prob::Union{BabyInfoPOMDP, BabyInfoRPOMDP}, b::Vector{Float64}, a::Symbol)
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

discount(p::Union{BabyInfoPOMDP,BabyInfoRPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{BabyInfoPOMDP,BabyInfoRPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::BabyInfoPOMDP, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

function generate_sor(prob::BabyInfoRPOMDP, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
