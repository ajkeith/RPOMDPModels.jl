# Crying baby problem (Kochenderfer 2015)
# POMDP, Robust POMDP, and Robust ρPOMDP formulations

struct BabyPOMDP <: POMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
end

struct BabyRPOMDP <: RPOMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
    usize::Float64
end

struct BabyIPOMDP <: IPOMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
    inforeward::Vector{Vector{Float64}}
end

struct BabyRIPOMDP <: RIPOMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
    usize::Float64
    inforeward::Vector{Vector{Float64}}
end

const states_baby = [:hungry, :full]
const actions_baby = [:feed, :nothing]
const observations_baby = [:quiet, :crying, :yelling]
const inforeward_baby = [[1.0, -1.0], [-1.0, 1.0], [0.2, 0.2]]

BabyPOMDP(r_feed, r_hungry) = BabyPOMDP(r_feed, r_hungry, 0.9)
BabyPOMDP() = BabyPOMDP(-5.0, -10.0, 0.9)

BabyRPOMDP(r_feed, r_hungry) = BabyRPOMDP(r_feed, r_hungry, 0.9, 0.025)
BabyRPOMDP(err) = BabyRPOMDP(-5.0, -10.0, 0.9, err)
BabyRPOMDP() = BabyRPOMDP(-5.0, -10.0, 0.9, 0.025)

BabyIPOMDP(r_feed, r_hungry) = BabyIPOMDP(r_feed, r_hungry, 0.9, inforeward_baby)
BabyIPOMDP(alphas) = BabyIPOMDP(-5.0, -10.0, 0.9, alphas)
BabyIPOMDP() = BabyIPOMDP(-5.0, -10.0, 0.9, inforeward_baby)

BabyRIPOMDP(r_feed, r_hungry) = BabyRIPOMDP(r_feed, r_hungry, 0.9, 0.025, inforeward_baby)
BabyRIPOMDP(err::Float64, alphas::Vector{Vector{Float64}}) = BabyRIPOMDP(-5.0, -10.0, 0.9, err, alphas)
BabyRIPOMDP() = BabyRIPOMDP(-5.0, -10.0, 0.9, 0.025, inforeward_baby)

states(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = states_baby
actions(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = actions_baby
observations(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = observations_baby

n_states(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = 2
n_actions(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = 2
n_observations(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = 3

state_index(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, s::Symbol) = s == :hungry ? 1 : 2
action_index(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, a::Symbol) = a == :feed ? 1 : 2
observation_index(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, z::Symbol) = z == :quiet ? 1 : z == :crying ? 2 : 3
obs_index(prob::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, z::Symbol) = observation_index(prob, z)

# start knowing baby is not not hungry
initial_state_distribution(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = SparseCat([:hungry, :full], [0.0, 1.0])
initial_belief(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = [0.0, 1.0]

# Transition functions
# Nominal transition function distributions
const pϵ_baby = 1e-6
const tdist_baby = [SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [1.0, 0.0]);
                    SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [0.1, 0.9])]
# Nominal transition function array tarray_baby[s,a,s'] = Pr(s' | s ,a)
const tarray_baby = cat(3, [0.0 1.0; 0.0 0.1], [1.0 0.0; 1.0 0.9]) # P(hungry | s, a), P(full | s, a)

# Nominal transitions
transition(prob::Union{BabyIPOMDP, BabyPOMDP}, s::Symbol, a::Symbol) = tdist_baby[state_index(prob, s), action_index(prob, a)]
transition(prob::Union{BabyIPOMDP, BabyPOMDP}, s::Symbol, a::Symbol, sp::Symbol) = tarray_baby[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::Union{BabyRPOMDP, BabyRIPOMDP}, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(tarray_baby[si,ai,:] - prob.usize, 0.0 + pϵ_baby)
    pupper = min.(tarray_baby[si,ai,:] + prob.usize, 1.0 - pϵ_baby)
    plower, pupper
end
function transition(prob::Union{BabyRPOMDP, BabyRIPOMDP}, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(tarray_baby[si, ai, spi] - prob.usize, 0.0 + pϵ_baby)
    pupper = min.(tarray_baby[si, ai, spi] + prob.usize, 1.0 - pϵ_baby)
    plower, pupper
end

# Nominal observation function distributions
const odist_baby = [SparseCat([:quiet, :crying, :yelling], [0.3, 0.6, 0.1])
                    SparseCat([:quiet, :crying, :yelling], [0.8, 0.1, 0.1])]

# Nominal observation function array oarray_baby[a,sp,z] = Pr(z|a,sp)
const oarray_baby = cat(3, [0.3 0.8; 0.3 0.8], [0.6 0.1; 0.6 0.1], [0.1 0.1; 0.1 0.1]) # Pr(quiet|a,sp), # Pr(crying|a,sp), # Pr(yelling|a,sp)

# Nominal observations
observation(prob::Union{BabyIPOMDP, BabyPOMDP}, a::Symbol, sp::Symbol) = odist_baby[state_index(prob, sp)]
observation(prob::Union{BabyIPOMDP, BabyPOMDP}, a::Symbol, sp::Symbol, z::Symbol) = oarray_baby[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::Union{BabyRPOMDP, BabyRIPOMDP}, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(oarray_baby[ai,spi,:] - prob.usize, 0.0 + pϵ_baby)
    pupper = min.(oarray_baby[ai,spi,:] + prob.usize, 1.0 - pϵ_baby)
    plower, pupper
end
function observation(prob::Union{BabyRPOMDP, BabyRIPOMDP}, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(oarray_baby[ai, spi, zi] - prob.usize, 0.0 + pϵ_baby)
    pupper = min.(oarray_baby[ai, spi, zi] + prob.usize, 1.0 - pϵ_baby)
    plower, pupper
end

function dynamics(prob::Union{BabyPOMDP,BabyIPOMDP})
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

function dynamics(prob::Union{BabyRPOMDP,BabyRIPOMDP})
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

function reward(prob::Union{BabyPOMDP, BabyRPOMDP}, s::Symbol, a::Symbol)
    r = 0.0
    if s == :hungry
        r += prob.r_hungry
    end
    if a == :feed
        r += prob.r_feed
    end
    return r
end
reward(prob::Union{BabyPOMDP,BabyRPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,s,a)

function reward(prob::Union{BabyIPOMDP, BabyRIPOMDP}, b::Vector{Float64}, a::Symbol)
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::Union{BabyIPOMDP,BabyRIPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,b,a)

function rewardalpha(prob::Union{BabyIPOMDP, BabyRIPOMDP}, b::Vector{Float64}, a::Symbol)
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

discount(p::Union{BabyPOMDP,BabyRPOMDP,BabyIPOMDP,BabyRIPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{BabyPOMDP,BabyIPOMDP,BabyRPOMDP,BabyRIPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::Union{BabyPOMDP,BabyIPOMDP}, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

function generate_sor(prob::Union{BabyRPOMDP,BabyRIPOMDP}, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
