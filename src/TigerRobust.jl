# Tiger Problem
# POMDP, Robust POMDP, and Robust ρPOMDP formulations

struct TigerPOMDP <: POMDP{Symbol, Symbol, Symbol}
    discount::Float64
end

struct TigerRPOMDP <: RPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    usize::Float64
end

struct TigerIPOMDP <: IPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    inforeward::Vector{Vector{Float64}}
end

struct TigerRIPOMDP <: RIPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    usize::Float64
    inforeward::Vector{Vector{Float64}}
end

const discount_tiger = 0.95
const states_tiger = [:tigerleft, :tigerright]
const actions_tiger = [:listen, :openleft, :openright]
const observations_tiger = [:tigerleft, :tigerright]
const inforeward_tiger = [[1.0, -1.0], [-1.0, 1.0], [0.2, 0.2]]

TigerPOMDP() = TigerPOMDP(discount_tiger)

TigerRPOMDP(err) = TigerRPOMDP(discount_tiger, err)
TigerRPOMDP() = TigerRPOMDP(discount_tiger, 0.025)

TigerIPOMDP(alphas) = TigerIPOMDP(discount_tiger, alphas)
TigerIPOMDP() = TigerIPOMDP(discount_tiger, inforeward_tiger)

TigerRIPOMDP(err::Float64, alphas::Vector{Vector{Float64}}) = TigerRIPOMDP(discount_tiger, err, alphas)
TigerRIPOMDP() = TigerRIPOMDP(discount_tiger, 0.025, inforeward_tiger)

states(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = states_tiger
actions(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = actions_tiger
observations(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = observations_tiger

n_states(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = 2
n_actions(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = 3
n_observations(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = 2

state_index(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}, s::Symbol) = s == :tigerleft ? 1 : 2
action_index(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}, a::Symbol) = a == :listen ? 1 : a == :openleft ? 2 : 3
observation_index(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}, z::Symbol) = z == :tigerleft ? 1 : 2
obs_index(prob::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}, z::Symbol) = observation_index(prob, z)

initial_state_distribution(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = SparseCat([:tigerleft, :tigerright], [0.5, 0.5])
initial_belief(::Union{TigerPOMDP, TigerRPOMDP, TigerIPOMDP, TigerRIPOMDP}) = [0.5, 0.5]

# Nominal transition function distributions
const pϵ_tiger = 1e-6
const tdist_tiger = [SparseCat(states_tiger, [1.0, 0.0]) SparseCat(states_tiger, [0.5, 0.5]) SparseCat(states_tiger, [0.5, 0.5]);
                    SparseCat(states_tiger, [0.0, 1.0]) SparseCat(states_tiger, [0.5, 0.5]) SparseCat(states_tiger, [0.5, 0.5])]

# Nominal transition function array tarray_tiger[s,a,s'] = Pr(s' | s ,a)
const tarray_tiger = cat(3, [1.0 0.5 0.5; 0.0 0.5 0.5], [0.0 0.5 0.5; 1.0 0.5 0.5]) # P(tigerleft | s, a), P(tigerright | s, a)

# Nominal transitions
transition(prob::Union{TigerIPOMDP, TigerPOMDP}, s::Symbol, a::Symbol) = tdist_tiger[state_index(prob, s), action_index(prob, a)]
transition(prob::Union{TigerIPOMDP, TigerPOMDP}, s::Symbol, a::Symbol, sp::Symbol) = tarray_tiger[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::Union{TigerRPOMDP, TigerRIPOMDP}, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(tarray_tiger[si,ai,:] - prob.usize, 0.0 + pϵ_tiger)
    pupper = min.(tarray_tiger[si,ai,:] + prob.usize, 1.0 - pϵ_tiger)
    plower, pupper
end
function transition(prob::Union{TigerRPOMDP, TigerRIPOMDP}, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(tarray_tiger[si, ai, spi] - prob.usize, 0.0 + pϵ_tiger)
    pupper = min.(tarray_tiger[si, ai, spi] + prob.usize, 1.0 - pϵ_tiger)
    plower, pupper
end

# Nominal observation function distributions
const odist_tiger = [SparseCat(observations_tiger, [0.85 0.15]) SparseCat(observations_tiger, [0.15 0.85]);
                    SparseCat(observations_tiger, [0.5 0.5]) SparseCat(observations_tiger, [0.5 0.5]);
                    SparseCat(observations_tiger, [0.5 0.5]) SparseCat(observations_tiger, [0.5 0.5])]

# Nominal observation function array oarray_tiger[a,sp,z] = Pr(z|a,sp)
const oarray_tiger = cat(3, [0.85 0.15; 0.5 0.5; 0.5 0.5], [0.15 0.85; 0.5 0.5; 0.5 0.5]) # Pr(tigerleft|a,sp), Pr(tigerright|a,sp)

# Nominal observations
observation(prob::Union{TigerIPOMDP, TigerPOMDP}, a::Symbol, sp::Symbol) = odist_tiger[action_index(prob, a), state_index(prob, sp)]
observation(prob::Union{TigerIPOMDP, TigerPOMDP}, a::Symbol, sp::Symbol, z::Symbol) = oarray_tiger[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::Union{TigerRPOMDP, TigerRIPOMDP}, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(oarray_tiger[ai,spi,:] - prob.usize, 0.0 + pϵ_tiger)
    pupper = min.(oarray_tiger[ai,spi,:] + prob.usize, 1.0 - pϵ_tiger)
    plower, pupper
end
function observation(prob::Union{TigerRPOMDP, TigerRIPOMDP}, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(oarray_tiger[ai, spi, zi] - prob.usize, 0.0 + pϵ_tiger)
    pupper = min.(oarray_tiger[ai, spi, zi] + prob.usize, 1.0 - pϵ_tiger)
    plower, pupper
end

function dynamics(prob::Union{TigerPOMDP,TigerIPOMDP})
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

function dynamics(prob::Union{TigerRPOMDP,TigerRIPOMDP})
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

function reward(prob::Union{TigerPOMDP, TigerRPOMDP}, s::Symbol, a::Symbol)
    r = 0.0
    if a == :listen
        r = -1.0
    elseif a == :openleft
        r = (s == :tigerleft) ? -100.0 : 10.0
    elseif a == :openright
        r = (s == :tigerright) ? -100.0 : 10.0
    end
    return r
end
reward(prob::Union{TigerPOMDP,TigerRPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,s,a)

function reward(prob::Union{TigerIPOMDP, TigerRIPOMDP}, b::Vector{Float64}, a::Symbol)
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::Union{TigerIPOMDP,TigerRIPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,b,a)

function rewardalpha(prob::Union{TigerIPOMDP, TigerRIPOMDP}, b::Vector{Float64}, a::Symbol)
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

discount(p::Union{TigerPOMDP,TigerRPOMDP,TigerIPOMDP,TigerRIPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{TigerPOMDP,TigerIPOMDP,TigerRPOMDP,TigerRIPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::Union{TigerPOMDP,TigerIPOMDP}, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

function generate_sor(prob::Union{TigerRPOMDP,TigerRIPOMDP}, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
