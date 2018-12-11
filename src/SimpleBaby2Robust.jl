struct SimpleBaby2RPOMDP <: RPOMDP{Symbol, Symbol, Symbol}
    r_feed::Float64
    r_hungry::Float64
    discount::Float64
    usize::Float64
end

const states_simplebaby2 = [:hungry, :full]
const actions_simplebaby2 = [:feed, :nothing]
const observations_simplebaby2 = [:quiet, :crying]

SimpleBaby2RPOMDP(r_feed, r_hungry) = SimpleBaby2RPOMDP(r_feed, r_hungry, 0.9, 0.025)
SimpleBaby2RPOMDP(err) = SimpleBaby2RPOMDP(-5.0, -10.0, 0.9, err)
SimpleBaby2RPOMDP() = SimpleBaby2RPOMDP(-5.0, -10.0, 0.9, 0.025)

states(::SimpleBaby2RPOMDP) = states_simplebaby2
actions(::SimpleBaby2RPOMDP) = actions_simplebaby2
observations(::SimpleBaby2RPOMDP) = observations_simplebaby2

n_states(::SimpleBaby2RPOMDP) = 2
n_actions(::SimpleBaby2RPOMDP) = 2
n_observations(::SimpleBaby2RPOMDP) = 2

state_index(::SimpleBaby2RPOMDP, s::Symbol) = s == :hungry ? 1 : 2
action_index(::SimpleBaby2RPOMDP, a::Symbol) = a == :feed ? 1 : 2
observation_index(::SimpleBaby2RPOMDP, z::Symbol) = z == :quiet ? 1 : 2
obs_index(prob::SimpleBaby2RPOMDP, z::Symbol) = observation_index(prob, z)

initial_state_distribution(::SimpleBaby2RPOMDP) = SparseCat([:hungry, :full], [0.0, 1.0])
initial_belief(::SimpleBaby2RPOMDP) = [0.0, 1.0]

# Transition functions
# Nominal transition function distributions
const pϵ_simplebaby2 = 1e-6
const tdist_simplebaby2 = [SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [1.0, 0.0]);
                    SparseCat([:hungry, :full], [0.0, 1.0]) SparseCat([:hungry, :full], [0.1, 0.9])]
# Nominal transition function array tarray_simplebaby2[s,a,s'] = Pr(s' | s ,a)
const tarray_simplebaby2 = cat(3, [0.0 1.0; 0.0 0.1], [1.0 0.0; 1.0 0.9]) # P(hungry | s, a), P(full | s, a)

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::SimpleBaby2RPOMDP, s::Symbol, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    plower = max.(tarray_simplebaby2[si,ai,:] - pϵ_simplebaby2, 0.0 + pϵ_simplebaby2)
    pupper = min.(tarray_simplebaby2[si,ai,:] + pϵ_simplebaby2, 1.0 - pϵ_simplebaby2)
    plower, pupper
end
function transition(prob::SimpleBaby2RPOMDP, s::Symbol, a::Symbol, sp::Symbol)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    plower = max.(tarray_simplebaby2[si, ai, spi] - pϵ_simplebaby2, 0.0 + pϵ_simplebaby2)
    pupper = min.(tarray_simplebaby2[si, ai, spi] + pϵ_simplebaby2, 1.0 - pϵ_simplebaby2)
    plower, pupper
end

# Nominal observation function distributions
const odist_simplebaby2 = [SparseCat([:quiet, :crying], [0.2, 0.8])
                    SparseCat([:quiet, :crying], [0.9, 0.1])]

# Nominal observation function array oarray_simplebaby2[a,sp,z] = Pr(z|a,sp)
const oarray_simplebaby2 = cat(3, [0.2 0.9; 0.2 0.9], [0.8 0.1; 0.8 0.1]) # Pr(quiet|a,sp), # Pr(crying|a,sp), # Pr(yelling|a,sp)

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::SimpleBaby2RPOMDP, a::Symbol, sp::Symbol)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    plower = max.(oarray_simplebaby2[ai,spi,:] - prob.usize, 0.0 + pϵ_simplebaby2)
    pupper = min.(oarray_simplebaby2[ai,spi,:] + prob.usize, 1.0 - pϵ_simplebaby2)
    plower, pupper
end
function observation(prob::SimpleBaby2RPOMDP, a::Symbol, sp::Symbol, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    plower = max.(oarray_simplebaby2[ai, spi, zi] - prob.usize, 0.0 + pϵ_simplebaby2)
    pupper = min.(oarray_simplebaby2[ai, spi, zi] + prob.usize, 1.0 - pϵ_simplebaby2)
    plower, pupper
end

function dynamics(prob::SimpleBaby2RPOMDP)
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

function reward(prob::SimpleBaby2RPOMDP, s::Symbol, a::Symbol)
    r = 0.0
    if s == :hungry
        r += prob.r_hungry
    end
    if a == :feed
        r += prob.r_feed
    end
    return r
end
reward(prob::SimpleBaby2RPOMDP, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,s,a)

discount(p::SimpleBaby2RPOMDP) = p.discount

# Simulation Functions
function initial_state(prob::SimpleBaby2RPOMDP, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::SimpleBaby2RPOMDP, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
