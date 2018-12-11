# SimpleTiger Problem
# POMDP, Robust POMDP, and Robust ρPOMDP formulations

struct SimpleTigerRPOMDP <: RPOMDP{Symbol, Symbol, Symbol}
    discount::Float64
    usize::Float64
end

const discount_simpletiger = 0.95
const states_simpletiger = [:tigerleft, :tigerright]
const actions_simpletiger = [:listen, :openleft, :openright]
const observations_simpletiger = [:tigerleft, :tigerright]

SimpleTigerRPOMDP(err) = SimpleTigerRPOMDP(discount_simpletiger, err)
SimpleTigerRPOMDP() = SimpleTigerRPOMDP(discount_simpletiger, 0.025)

states(::SimpleTigerRPOMDP) = states_simpletiger
actions(::SimpleTigerRPOMDP) = actions_simpletiger
observations(::SimpleTigerRPOMDP) = observations_simpletiger

n_states(::SimpleTigerRPOMDP) = 2
n_actions(::SimpleTigerRPOMDP) = 3
n_observations(::SimpleTigerRPOMDP) = 2

state_index(::SimpleTigerRPOMDP, s::Symbol) = s == :tigerleft ? 1 : 2
action_index(::SimpleTigerRPOMDP, a::Symbol) = a == :listen ? 1 : a == :openleft ? 2 : 3
observation_index(::SimpleTigerRPOMDP, z::Symbol) = z == :tigerleft ? 1 : 2
obs_index(prob::SimpleTigerRPOMDP, z::Symbol) = observation_index(prob, z)

initial_state_distribution(::SimpleTigerRPOMDP) = SparseCat([:tigerleft, :tigerright], [0.5, 0.5])
initial_belief(::SimpleTigerRPOMDP) = [0.5, 0.5]

# Nominal transition function distributions
const pϵ_simpletiger = 1e-6
const tdist_simpletiger = [SparseCat(states_simpletiger, [1.0, 0.0]) SparseCat(states_simpletiger, [0.5, 0.5]) SparseCat(states_simpletiger, [0.5, 0.5]);
                    SparseCat(states_simpletiger, [0.0, 1.0]) SparseCat(states_simpletiger, [0.5, 0.5]) SparseCat(states_simpletiger, [0.5, 0.5])]

# Nominal transition function array tarray_simpletiger[s,a,s'] = Pr(s' | s ,a)
const tarray_simpletiger = cat(3, [1.0 0.5 0.5; 0.0 0.5 0.5], [0.0 0.5 0.5; 1.0 0.5 0.5]) # P(tigerleft | s, a), P(tigerright | s, a)

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::SimpleTigerRPOMDP, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(tarray_simpletiger[si,ai,:] - pϵ_simpletiger, 0.0 + pϵ_simpletiger)
    pupper = min.(tarray_simpletiger[si,ai,:] + pϵ_simpletiger, 1.0 - pϵ_simpletiger)
    plower, pupper
end
function transition(prob::SimpleTigerRPOMDP, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(tarray_simpletiger[si, ai, spi] - pϵ_simpletiger, 0.0 + pϵ_simpletiger)
    pupper = min.(tarray_simpletiger[si, ai, spi] + pϵ_simpletiger, 1.0 - pϵ_simpletiger)
    plower, pupper
end

# Nominal observation function distributions
const odist_simpletiger = [SparseCat(observations_simpletiger, [0.85 0.15]) SparseCat(observations_simpletiger, [0.15 0.85]);
                    SparseCat(observations_simpletiger, [0.5 0.5]) SparseCat(observations_simpletiger, [0.5 0.5]);
                    SparseCat(observations_simpletiger, [0.5 0.5]) SparseCat(observations_simpletiger, [0.5 0.5])]

# Nominal observation function array oarray_simpletiger[a,sp,z] = Pr(z|a,sp)
const oarray_simpletiger = cat(3, [0.85 0.15; 0.5 0.5; 0.5 0.5], [0.15 0.85; 0.5 0.5; 0.5 0.5]) # Pr(tigerleft|a,sp), Pr(tigerright|a,sp)

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::SimpleTigerRPOMDP, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(oarray_simpletiger[ai,spi,:] - prob.usize, 0.0 + pϵ_simpletiger)
    pupper = min.(oarray_simpletiger[ai,spi,:] + prob.usize, 1.0 - pϵ_simpletiger)
    plower, pupper
end
function observation(prob::SimpleTigerRPOMDP, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(oarray_simpletiger[ai, spi, zi] - prob.usize, 0.0 + pϵ_simpletiger)
    pupper = min.(oarray_simpletiger[ai, spi, zi] + prob.usize, 1.0 - pϵ_simpletiger)
    plower, pupper
end

function dynamics(prob::SimpleTigerRPOMDP)
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

function reward(prob::SimpleTigerRPOMDP, s::Symbol, a::Symbol)
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
reward(prob::SimpleTigerRPOMDP, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,s,a)

discount(p::SimpleTigerRPOMDP) = p.discount

# Simulation Functions
function initial_state(prob::SimpleTigerRPOMDP, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::SimpleTigerRPOMDP, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
