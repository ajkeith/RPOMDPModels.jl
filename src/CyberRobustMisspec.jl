# Cyber Test Problem
# POMDP, Robust POMDP, and Robust ρPOMDP formulations

struct CyberTestIPOMDP <: IPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    inforeward::Vector{Vector{Float64}}
    b0::Vector{Float64}
end

# Discount
const discount_cyber_test = 0.75

# State space
const nmoe_test = 3 # number of measures of effectiveness (MOE)
const nlevels_test = 3 # number of levels at which the MOE is measured
const lmin_test = 1 # min level
const lmax_test = 3 # max level
const nS_test = nlevels_test ^ nmoe_test # number of states
S = Vector{Vector{Int}}(nS_test) # states, as [moe_1_level, ...]
ind = 1
for i = 1:nlevels_test, j = 1:nlevels_test, k = 1:nlevels_test
  S[ind] = [i,j,k]
  ind += 1
end
const states_cyber_test = copy(S)

# Action space
const na_test = 2 # number of assessment assets
const nA_test = 6 # total number of actions
A = Vector{Vector{Int}}() # actions, as [asset_1_assignment, ...]
for i = 1:nmoe_test, j = 1:nmoe_test
  (i != j) && (push!(A, [i,j]))
end
const actions_cyber_test = copy(A)

# Observation space
const nZ_test = nlevels_test ^ na_test
Z = Vector{Vector{Int}}(nZ_test)
ind = 1
for i = 1:nlevels_test, j = 1:nlevels_test
  Z[ind] = [i,j]
  ind += 1
end
const observations_cyber_test = copy(Z)

# Belief reward (PWLC vectors: simple)
br = Vector{Vector{Float64}}(nS_test)
br[1] = vcat(1.0, fill(-1/(nS_test-1), nS_test - 1))
br[nS_test] = vcat(fill(-1/(nS_test-1), nS_test - 1), 1.0)
for i = 2:(nS_test-1)
  br[i] = vcat(fill(-1/(nS_test-1), i - 1), 1.0, fill(-1/(nS_test-1), nS_test - i))
end
const inforeward_cyber_test = copy(br)

# Initial state and belief
e11 = zeros(nS_test); e11[1] = 1.0;
const e1_test = copy(e11)        # intial state
const b0_cyber_test = copy(e11)  # initial belief

CyberTestIPOMDP(alphas) = CyberTestIPOMDP(discount_cyber_test, alphas, b0_cyber_test)
CyberTestIPOMDP() = CyberTestIPOMDP(discount_cyber_test, inforeward_cyber_test, b0_cyber_test)

states(::CyberTestIPOMDP) = states_cyber_test
actions(::CyberTestIPOMDP) = actions_cyber_test
observations(::CyberTestIPOMDP) = observations_cyber_test

n_states(::CyberTestIPOMDP) = nS_test
n_actions(::CyberTestIPOMDP) = nA_test
n_observations(::CyberTestIPOMDP) = nZ_test

state_index(::CyberTestIPOMDP, s::Vector{Int}) = find(x -> x == s, states_cyber_test)[1]
action_index(::CyberTestIPOMDP, a::Vector{Int}) = find(x -> x == a, actions_cyber_test)[1]
observation_index(::CyberTestIPOMDP, z::Vector{Int}) = find(x -> x == z, observations_cyber_test)[1]
obs_index(prob::CyberTestIPOMDP, z::Vector{Int}) = observation_index(prob, z)

initial_state_distribution(::CyberTestIPOMDP) = SparseCat(states_cyber_test, e1_test)
initial_belief(prob::CyberTestIPOMDP) = prob.b0
initial_belief_distribution(prob::CyberTestIPOMDP) = SparseCat(states_cyber_test, initial_belief(prob))

# Transitions
const pϵ_cyber_test = 1e-6 # near-saturation bound
const delt_test = 0.01 # imprecision in transition distribution
const pd_test = 0.15 # prob of decrease
const ps_test = 0.55 # prob of stay
const pi_test = 0.3 # prob of increase
const psd_test = ps_test + pd_test # prob of decline or stay at lower border
const psi_test = ps_test + pi_test # prob of stay or improve at upper border

# Nominal transition function array tarray_cyber[s,a,s'] = Pr(s' | s ,a)
const tarray_cyber_test, tlarray_cyber_test, tuarray_cyber_test = calcTArray(states_cyber_test, actions_cyber_test, (pd_test,ps_test,pi_test), delt_test)

# Nominal transition function distributions
const tdist_cyber_test = calcTDist(nS_test, nA_test, tarray_cyber_test)

# Nominal transitions
transition(prob::CyberTestIPOMDP, s::Vector{Int}, a::Vector{Int}) = tdist_cyber_test[state_index(prob, s), action_index(prob, a)]
transition(prob::CyberTestIPOMDP, s::Vector{Int}, a::Vector{Int}, sp::Vector{Int}) = tarray_cyber_test[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Observation function
const delo1_test = pϵ_cyber_test # sensor 1 immprecision
const delo2_test = 0.4 # sensor 2 imprecision
const sensor_imprecision_test = [delo1_test, delo2_test]
const acc1_test = 0.8 - delo1_test # likelihood of correct observation error for sensor 1 (off-nominal)
const acc2_test = 0.9 - delo2_test # likelihood of correct observation error for sensor 2 (off-nominal)
const sensor_accuracy_test = [acc1_test, acc2_test]

# Nominal transition function distributions
const odist_cyber_test = calcODist(nS_test, nA_test, oarray_cyber_test)

# Nominal transitions
observation(prob::CyberTestIPOMDP, a::Vector{Int}, sp::Vector{Int}) = odist_cyber_test[action_index(prob, a), state_index(prob, sp)]
observation(prob::CyberTestIPOMDP, a::Vector{Int}, sp::Vector{Int}, z::Vector{Int}) = oarray_cyber_test[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

function dynamics(prob::CyberTestIPOMDP)
    ns = n_states(prob)
    nz = n_observations(prob)
    na_test = n_actions(prob)
    tarr = states(prob)
    sarr = states(prob)
    aarr = actions(prob)
    zarr = observations(prob)
    p = Array{Float64}(ns,nz,ns,na_test)
    plower = Array{Float64}(ns,nz,ns,na_test)
    pupper = Array{Float64}(ns,nz,ns,na_test)
    for tind in 1:ns, zind in 1:nz, sind in 1:ns, aind in 1:na_test
        pt = transition(prob, sarr[sind], aarr[aind], tarr[tind])
        pz = observation(prob, aarr[aind], tarr[tind], zarr[zind])
        p[tind, zind, sind, aind] = pz * pt
    end
    p
end

function reward(prob::CyberTestIPOMDP, b::Vector{Float64}, a::Vector{Int})
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::CyberTestIPOMDP, b::Vector{Float64}, s::Vector{Int}, a::Vector{Int}, sp::Vector{Int}) = reward(prob,b,a)

function rewardalpha(prob::CyberTestIPOMDP, b::Vector{Float64}, a::Vector{Int})
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

discount(p::CyberTestIPOMDP) = p.discount

# Simulation Functions
function initial_state(prob::CyberTestIPOMDP, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::CyberTestIPOMDP, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
