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

const states_const = [:hungry, :full]
const actions_const = [:feed, :nothing]
const observations_const = [:quiet, :crying, :yelling]
const ra11 = [1.0, -1.0] # belief-reward alpha vector
const ra12 = [-1.0, 1.0] # belief-reward alpha vector
const ra13 = [0.2, 0.2] # belief-reward alpha vector
const ralphas1 = [ra11, ra12, ra13]
const ra21 = [1.0, 0.0]
const ra22 = [0.9, 0.1]
const ra23 = [0.8, 0.2]
const ra24 = [0.7, 0.3]
const ralphas2 = [ra21, ra22, ra23, ra24, 1-ra24, 1-ra23, 1-ra22, 1-ra21]

BabyPOMDP(r_feed, r_hungry) = BabyPOMDP(r_feed, r_hungry, 0.9)
BabyPOMDP() = BabyPOMDP(-5.0, -10.0, 0.9)
BabyRPOMDP(r_feed, r_hungry) = BabyRPOMDP(r_feed, r_hungry, 0.9, 0.025)
BabyRPOMDP() = BabyRPOMDP(-5.0, -10.0, 0.9, 0.025)
BabyIPOMDP(r_feed, r_hungry) = BabyIPOMDP(r_feed, r_hungry, 0.9, ralphas1)
BabyIPOMDP() = BabyIPOMDP(-5.0, -10.0, 0.9, ralphas1)
BabyRIPOMDP(r_feed, r_hungry) = BabyRIPOMDP(r_feed, r_hungry, 0.9, 0.025, ralphas1)
BabyRIPOMDP() = BabyRIPOMDP(-5.0, -10.0, 0.9, 0.025, ralphas1)

states(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = states_const
actions(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = actions_const
observations(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = observations_const

n_states(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = 2
n_actions(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = 2
n_observations(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = 3

state_index(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, s::Symbol) = s == :hungry ? 1 : 2
action_index(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, a::Symbol) = a == :feed ? 1 : 2
observation_index(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, z::Symbol) = z == :quiet ? 1 : z == :crying ? 2 : 3
obs_index(prob::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}, z::Symbol) = observation_index(prob, z)

# start knowing baby is not not hungry
initial_state_distribution(::Union{BabyPOMDP, BabyRPOMDP, BabyIPOMDP, BabyRIPOMDP}) = SparseCat([:hungry, :full], [0.0, 1.0])

# Nominal transition function distributions
const t_dist = [SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [1.0, 0.0]);
                    SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [0.1, 1.0])]

# Nominal transition function array t_array[s,a,s'] = Pr(s' | s ,a)
const t_1 = [0.0 1.0; 0.0 0.1] # P(hungry | s, a)
const t_2 = [1.0 0.0; 1.0 0.9] # P(full | s, a)
t_temp = zeros(2,2,2)
t_temp[:,:,1] = t_1
t_temp[:,:,2] = t_2
const t_array = copy(t_temp)

# Nominal transitions
transition(prob::Union{BabyIPOMDP, BabyPOMDP}, s::Symbol, a::Symbol) = t_dist[state_index(prob, s), action_index(prob, a)]
transition(prob::Union{BabyIPOMDP, BabyPOMDP}, s::Symbol, a::Symbol, sp::Symbol) = t_array[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::Union{BabyRPOMDP, BabyRIPOMDP}, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(t_array[si,ai,:] - prob.usize, 0.0)
    pupper = min.(t_array[si,ai,:] + prob.usize, 1.0)
    plower, pupper
end
function transition(prob::Union{BabyRPOMDP, BabyRIPOMDP}, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(t_array[si, ai, spi] - prob.usize, 0.0)
    pupper = min.(t_array[si, ai, spi] + prob.usize, 1.0)
    plower, pupper
end

# Nominal observation function distributions
const o_dist = [SparseCat([:quiet, :crying, :yelling], [0.3, 0.6, 0.1])
                    SparseCat([:quiet, :crying, :yelling], [0.8, 0.1, 0.1])]

# Nominal observation function array o_array[a,sp,z] = Pr(z|a,sp)
const o_1 = [0.3 0.8; 0.3 0.8] # Pr(quiet|a,sp)
const o_2 = [0.6 0.1; 0.6 0.1] # Pr(crying|a,sp)
const o_3 = [0.1 0.1; 0.1 0.1] # Pr(crying|a,sp)
o_temp = zeros(2,2,3)
o_temp[:,:,1] = o_1
o_temp[:,:,2] = o_2
o_temp[:,:,3] = o_3
const o_array = copy(o_temp)

# Nominal observations
observation(prob::Union{BabyIPOMDP, BabyPOMDP}, a::Symbol, sp::Symbol) = o_dist[state_index(prob, sp)]
observation(prob::Union{BabyIPOMDP, BabyPOMDP}, a::Symbol, sp::Symbol, z::Symbol) = o_array[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::Union{BabyRPOMDP, BabyRIPOMDP}, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(o_array[ai,spi,:] - prob.usize, 0.0)
    pupper = min.(o_array[ai,spi,:] + prob.usize, 1.0)
    plower, pupper
end
function observation(prob::Union{BabyRPOMDP, BabyRIPOMDP}, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(o_array[ai, spi, zi] - prob.usize, 0.0)
    pupper = min.(o_array[ai, spi, zi] + prob.usize, 1.0)
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

function psample(plower::Vector{Float64}, pupper::Vector{Float64})
    n = length(plower)
    A = zeros(Float64, 2n+1, n)
    for i = 1:n
        A[2i - 1,i] = 1.0
        A[2i, i] = -1.0
    end
    A[end,:] = ones(n)
    bc = Array{Float64}(2n+1)
    for i = 1:n
        bc[2i - 1] = pupper[i]
        bc[2i] = plower[i]
    end
    bc[end] = 1.0
    d = vcat(fill("<=", 2n),"=")
    nsample = 1
    rout = R"""
    library(hitandrun)
    constr <- list(constr = $A, rhs = $bc, dir = $d)
    samples <- hitandrun(constr, n.samples = $nsample, thin = ($n) ^ 3)
    """
    p = reshape(rcopy(rout), n)
end
