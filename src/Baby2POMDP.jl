# Crying baby2 problem (Kochenderfer 2015)
# POMDP, Robust POMDP, and Robust ρPOMDP formulations

struct Baby2POMDP <: POMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
end

struct Baby2RPOMDP <: RPOMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
    usize::Float64
end

struct Baby2IPOMDP <: IPOMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
    inforeward::Vector{Vector{Float64}}
end

struct Baby2RIPOMDP <: RIPOMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
    usize::Float64
    inforeward::Vector{Vector{Float64}}
end

const states_baby2 = [:hungry, :full]
const actions_baby2 = [:feed, :nothing]
const observations_baby2 = [:quiet, :crying]
const inforeward_baby2 = [[1.0, -1.0], [-1.0, 1.0], [0.2, 0.2]]

Baby2POMDP(r_feed, r_hungry) = Baby2POMDP(r_feed, r_hungry, 0.9)
Baby2POMDP() = Baby2POMDP(-5.0, -10.0, 0.9)

Baby2RPOMDP(r_feed, r_hungry) = Baby2RPOMDP(r_feed, r_hungry, 0.9, 0.025)
Baby2RPOMDP(err) = Baby2RPOMDP(-5.0, -10.0, 0.9, err)
Baby2RPOMDP() = Baby2RPOMDP(-5.0, -10.0, 0.9, 0.025)

Baby2IPOMDP(r_feed, r_hungry) = Baby2IPOMDP(r_feed, r_hungry, 0.9, inforeward_baby2)
Baby2IPOMDP(alphas) = Baby2IPOMDP(-5.0, -10.0, 0.9, alphas)
Baby2IPOMDP() = Baby2IPOMDP(-5.0, -10.0, 0.9, inforeward_baby2)

Baby2RIPOMDP(r_feed, r_hungry) = Baby2RIPOMDP(r_feed, r_hungry, 0.9, 0.025, inforeward_baby2)
Baby2RIPOMDP(err::Float64, alphas::Vector{Vector{Float64}}) = Baby2RIPOMDP(-5.0, -10.0, 0.9, err, alphas)
Baby2RIPOMDP() = Baby2RIPOMDP(-5.0, -10.0, 0.9, 0.025, inforeward_baby2)

states(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = states_baby2
actions(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = actions_baby2
observations(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = observations_baby2

n_states(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = 2
n_actions(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = 2
n_observations(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = 2

state_index(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}, s::Symbol) = s == :hungry ? 1 : 2
action_index(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}, a::Symbol) = a == :feed ? 1 : 2
observation_index(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}, z::Symbol) = z == :quiet ? 1 : 2
obs_index(prob::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}, z::Symbol) = observation_index(prob, z)

# start knowing baby2 is not not hungry
initial_state_distribution(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = SparseCat([:hungry, :full], [0.0, 1.0])
initial_belief(::Union{Baby2POMDP, Baby2RPOMDP, Baby2IPOMDP, Baby2RIPOMDP}) = [0.0, 1.0]

# Transition functions
# Nominal transition function distributions
const pϵ_baby2 = 1e-6
const tdist_baby2 = [SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [1.0, 0.0]);
                    SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [0.1, 0.9])]
# Nominal transition function array tarray_baby2[s,a,s'] = Pr(s' | s ,a)
const tarray_baby2 = cat(3, [0.0 1.0; 0.0 0.1], [1.0 0.0; 1.0 0.9]) # P(hungry | s, a), P(full | s, a)

# Nominal transitions
transition(prob::Union{Baby2IPOMDP, Baby2POMDP}, s::Symbol, a::Symbol) = tdist_baby2[state_index(prob, s), action_index(prob, a)]
transition(prob::Union{Baby2IPOMDP, Baby2POMDP}, s::Symbol, a::Symbol, sp::Symbol) = tarray_baby2[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::Union{Baby2RPOMDP, Baby2RIPOMDP}, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(tarray_baby2[si,ai,:] - prob.usize, 0.0 + pϵ_baby2)
    pupper = min.(tarray_baby2[si,ai,:] + prob.usize, 1.0 - pϵ_baby2)
    plower, pupper
end
function transition(prob::Union{Baby2RPOMDP, Baby2RIPOMDP}, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(tarray_baby2[si, ai, spi] - prob.usize, 0.0 + pϵ_baby2)
    pupper = min.(tarray_baby2[si, ai, spi] + prob.usize, 1.0 - pϵ_baby2)
    plower, pupper
end

# Nominal observation function distributions
const odist_baby2 = [SparseCat([:quiet, :crying], [0.2, 0.8])
                    SparseCat([:quiet, :crying], [0.9, 0.1])]

# Nominal observation function array oarray_baby2[a,sp,z] = Pr(z|a,sp)
const oarray_baby2 = cat(3, [0.8 0.1; 0.8 0.1], [0.2 0.9; 0.2 0.9]) # Pr(quiet|a,sp), # Pr(crying|a,sp), # Pr(yelling|a,sp)

# Nominal observations
observation(prob::Union{Baby2IPOMDP, Baby2POMDP}, a::Symbol, sp::Symbol) = odist_baby2[state_index(prob, sp)]
observation(prob::Union{Baby2IPOMDP, Baby2POMDP}, a::Symbol, sp::Symbol, z::Symbol) = oarray_baby2[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::Union{Baby2RPOMDP, Baby2RIPOMDP}, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(oarray_baby2[ai,spi,:] - prob.usize, 0.0 + pϵ_baby2)
    pupper = min.(oarray_baby2[ai,spi,:] + prob.usize, 1.0 - pϵ_baby2)
    plower, pupper
end
function observation(prob::Union{Baby2RPOMDP, Baby2RIPOMDP}, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(oarray_baby2[ai, spi, zi] - prob.usize, 0.0 + pϵ_baby2)
    pupper = min.(oarray_baby2[ai, spi, zi] + prob.usize, 1.0 - pϵ_baby2)
    plower, pupper
end

function dynamics(prob::Union{Baby2POMDP,Baby2IPOMDP})
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

function dynamics(prob::Union{Baby2RPOMDP,Baby2RIPOMDP})
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

function reward(prob::Union{Baby2POMDP, Baby2RPOMDP}, s::Symbol, a::Symbol)
    r = 0.0
    if s == :hungry
        r += prob.r_hungry
    end
    if a == :feed
        r += prob.r_feed
    end
    return r
end
reward(prob::Union{Baby2POMDP,Baby2RPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,s,a)

function reward(prob::Union{Baby2IPOMDP, Baby2RIPOMDP}, b::Vector{Float64}, a::Symbol)
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::Union{Baby2IPOMDP,Baby2RIPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,b,a)

function rewardalpha(prob::Union{Baby2IPOMDP, Baby2RIPOMDP}, b::Vector{Float64}, a::Symbol)
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

discount(p::Union{Baby2POMDP,Baby2RPOMDP,Baby2IPOMDP,Baby2RIPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{Baby2POMDP,Baby2IPOMDP,Baby2RPOMDP,Baby2RIPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::Union{Baby2POMDP,Baby2IPOMDP}, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

function generate_sor(prob::Union{Baby2RPOMDP,Baby2RIPOMDP}, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
