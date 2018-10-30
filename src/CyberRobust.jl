# Cyber Problem
# POMDP, Robust POMDP, and Robust ρPOMDP formulations

struct CyberPOMDP <: POMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    b0::Vector{Float64}
end

struct CyberRPOMDP <: RPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    usize::Float64
    b0::Vector{Float64}
end

struct CyberIPOMDP <: IPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    inforeward::Vector{Vector{Float64}}
    b0::Vector{Float64}
end

struct CyberRIPOMDP <: RIPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    usize::Float64
    inforeward::Vector{Vector{Float64}}
    b0::Vector{Float64}
end

# Discount
const discount_cyber = 0.75

# State space
const nmoe = 3 # number of measures of effectiveness (MOE)
const nlevels = 3 # number of levels at which the MOE is measured
const lmin = 1 # min level
const lmax = 3 # max level
const nS = nlevels ^ nmoe # number of states
S = Vector{Vector{Int}}(nS) # states, as [moe_1_level, ...]
ind = 1
for i = 1:nlevels, j = 1:nlevels, k = 1:nlevels
  S[ind] = [i,j,k]
  ind += 1
end
const states_cyber = copy(S)

# Action space
const na = 2 # number of assessment assets
const nA = 6 # total number of actions
A = Vector{Vector{Int}}() # actions, as [asset_1_assignment, ...]
for i = 1:nmoe, j = 1:nmoe
  (i != j) && (push!(A, [i,j]))
end
const actions_cyber = copy(A)

# Observation space
const nZ = nlevels ^ na
Z = Vector{Vector{Int}}(nZ)
ind = 1
for i = 1:nlevels, j = 1:nlevels
  Z[ind] = [i,j]
  ind += 1
end
const observations_cyber = copy(Z)

# Belief reward (PWLC vectors: simple)
br = Vector{Vector{Float64}}(nS)
br[1] = vcat(1.0, fill(-1/(nS-1), nS - 1))
br[nS] = vcat(fill(-1/(nS-1), nS - 1), 1.0)
for i = 2:(nS-1)
  br[i] = vcat(fill(-1/(nS-1), i - 1), 1.0, fill(-1/(nS-1), nS - i))
end
const inforeward_cyber = copy(br)

# Initial state and belief
e11 = zeros(nS); e11[1] = 1.0;
const e1 = copy(e11)        # intial state
const b0_cyber = copy(e11)  # initial belief

CyberPOMDP() = CyberPOMDP(discount_cyber, b0_cyber)

CyberRPOMDP(err) = CyberRPOMDP(discount_cyber, err, b0_cyber)
CyberRPOMDP() = CyberRPOMDP(discount_cyber, 0.4, b0_cyber)

CyberIPOMDP(alphas) = CyberIPOMDP(discount_cyber, alphas, b0_cyber)
CyberIPOMDP() = CyberIPOMDP(discount_cyber, inforeward_cyber, b0_cyber)

CyberRIPOMDP(err::Float64, alphas::Vector{Vector{Float64}}) = CyberRIPOMDP(discount_cyber, err, alphas, b0_cyber)
CyberRIPOMDP(err::Float64) = CyberRIPOMDP(discount_cyber, err, inforeward_cyber, b0_cyber)
CyberRIPOMDP() = CyberRIPOMDP(discount_cyber, 0.4, inforeward_cyber, b0_cyber)

states(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = states_cyber
actions(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = actions_cyber
observations(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = observations_cyber

n_states(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = nS
n_actions(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = nA
n_observations(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = nZ

state_index(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}, s::Vector{Int}) = find(x -> x == s, states_cyber)[1]
action_index(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}, a::Vector{Int}) = find(x -> x == a, actions_cyber)[1]
observation_index(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}, z::Vector{Int}) = find(x -> x == z, observations_cyber)[1]
obs_index(prob::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}, z::Vector{Int}) = observation_index(prob, z)

initial_state_distribution(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = SparseCat(states_cyber, e1)
initial_belief(prob::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = prob.b0
initial_belief_distribution(prob::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = SparseCat(states_cyber, initial_belief(prob))

# Transitions
const pϵ_cyber = 1e-6 # near-saturation bound
const delt = 0.01 # imprecision in transition distribution
const pd = 0.15 # prob of decrease
const ps = 0.55 # prob of stay
const pi = 0.3 # prob of increase
const psd = ps + pd # prob of decline or stay at lower border
const psi = ps + pi # prob of stay or improve at upper border

# transition function: probabilty of going from state s to state j
function psj(s::Vector{Int},j::Vector{Int})
  prob = 1.0
  count = 0
  for i = 1:nmoe
    level_s = s[i]
    level_j = j[i]
    if level_s == lmin # current state is lowest level
      if level_j == level_s
        prob = prob * psd
      elseif level_j == level_s + 1
        prob = prob * pi
      else # zero probability of skipping levels
        return prob = 0.0
      end
    elseif level_s == lmax # current state is highest level
      if level_j == level_s
        prob = prob * psi
      elseif level_j == level_s - 1
        prob = prob * pd
      else # zero probability of skipping levels
        return prob = 0.0
      end
    else # current state is an intermediate level
      if level_j == level_s
        prob = prob * ps
        count += 1
      elseif level_j == level_s + 1
        prob = prob * pi
      elseif level_j == level_s - 1
        prob = prob * pd
      else # zero probability of skipping levels
        return prob = 0.0
      end
    end
  end
  prob
end

# robust transition function: probabilty of going from state s to state j
function psj(s::Array{Int,1}, j::Array{Int,1}, par, delt)
  # add imprecision (one-sided)
  delt > 0 ? (pd,ps,pi) = min.(par .+ delt, 1 - pϵ_cyber) : (pd,ps,pi) = max.(par .+ delt, 0 + pϵ_cyber)
  psd = min(ps + pd, 1 - pϵ_cyber) # prob of decline or stay at lower border
  psi = min(ps + pi, 1 - pϵ_cyber) # prob of stay or improve at upper border
  prob = 1.0
  for i = 1:nmoe
    level_s = s[i]
    level_j = j[i]
    if level_s == lmin # current state is lowest level
      if level_j == level_s
        prob = prob * psd
      elseif level_j == level_s + 1
        prob = prob * pi
      else # zero probability of skipping levels
        return prob = 0.0
      end
    elseif level_s == lmax # current state is highest level
      if level_j == level_s
        prob = prob * psi
      elseif level_j == level_s - 1
        prob = prob * pd
      else # zero probability of skipping levels
        return prob = 0.0
      end
    else # current state is an intermediate level
      if level_j == level_s
        prob = prob * ps
      elseif level_j == level_s + 1
        prob = prob * pi
      elseif level_j == level_s - 1
        prob = prob * pd
      else # zero probability of skipping levels
        return prob = 0.0
      end
    end
  end
  prob
end

# calculate trasnsition matrix
function calcTArray(S::Vector{Vector{Int}}, A::Vector{Vector{Int}}, par, delt)
  ns = length(S)
  na = length(A)
  T = zeros(ns, na, ns)
  Tl = zeros(ns, na, ns)
  Tu = zeros(ns, na, ns)
  for (si,s) in enumerate(S), (ji,j) in enumerate(S)
      T[si,:,ji] = psj(s,j)
      tl = psj(s,j,par,-delt)
      tu = psj(s,j,par,delt)
      # tl = max(tl, 0.0 + pϵ_cyber)
      # tu = min(tu, 1.0 - pϵ_cyber)
      (tu < tl) && (tu = tl + pϵ_cyber / 2)
      Tl[si,:,ji] = tl
      Tu[si,:,ji] = tu
  end
  for (si,s) in enumerate(S), (ai,a) in enumerate(A)
    T[si,ai,:] = T[si,ai,:] ./ sum(T[si,ai,:])
  end
  T, Tl, Tu
end

# Nominal transition function array tarray_cyber[s,a,s'] = Pr(s' | s ,a)
const tarray_cyber, tlarray_cyber, tuarray_cyber = calcTArray(states_cyber, actions_cyber, (pd,ps,pi), delt)

# calculate trasnsition distribution
function calcTDist(ns::Int, na::Int, tarray::Array{Float64,3})
  T = Array{SparseCat}(ns, na)
  for si = 1:ns, ai = 1:na
      T[si,ai] = SparseCat(states_cyber, tarray[si,ai,:])
  end
  T
end

# Nominal transition function distributions
const tdist_cyber = calcTDist(nS, nA, tarray_cyber)

# Nominal transitions
transition(prob::Union{CyberIPOMDP, CyberPOMDP}, s::Vector{Int}, a::Vector{Int}) = tdist_cyber[state_index(prob, s), action_index(prob, a)]
transition(prob::Union{CyberIPOMDP, CyberPOMDP}, s::Vector{Int}, a::Vector{Int}, sp::Vector{Int}) = tarray_cyber[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::Union{CyberRPOMDP, CyberRIPOMDP}, s::Vector{Int}, a::Vector{Int})
    si, ai = state_index(prob, s), action_index(prob, a)
    tlarray_cyber[si,ai,:], tuarray_cyber[si,ai,:]
end
function transition(prob::Union{CyberRPOMDP, CyberRIPOMDP}, s::Vector{Int}, a::Vector{Int}, sp::Vector{Int})
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    tlarray_cyber[si,ai,spi], tuarray_cyber[si,ai,spi]
end

# Observation function
const acc1 = 0.8 # likelihood of correct observation error for sensor 1
const acc2 = 0.9 # likelihood of correct observation error for sensor 2
const sensor_accuracy = [acc1, acc2]
const delo1 = pϵ_cyber # sensor 1 immprecision
const delo2 = 0.4 # sensor 2 imprecision
const sensor_imprecision = [delo1, delo2]

# observation function
function o(a::Vector{Int}, sp::Vector{Int}, z::Vector{Int}, accvec::Vector{Float64})
  pz = 1.0
  nsensor = length(a)
  for i = 1:nsensor
    assingment_sensor = a[i]
    level_sp = sp[assingment_sensor]
    level_z = z[i]
    pacc = accvec[i]
    if level_sp == lmin # current state is lowest level
      if level_z == level_sp
        pz = pz * pacc
      elseif level_z == level_sp + 1
        pz = pz * (1 - pacc)
      else # zero probability of error > 1
        return pz = 0.0
      end
    elseif level_sp == lmax # current state is highest level
      if level_z == level_sp
        pz = pz * pacc
      elseif level_z == level_sp - 1
        pz = pz * (1 - pacc)
      else # zero probability of error > 1
        return pz = 0.0
      end
    else # current state is an intermediate level
      if level_z == level_sp
        pz = pz * pacc
      elseif level_z == level_sp + 1
        pz = pz * (1 - pacc) / 2
      elseif level_z == level_sp - 1
        pz = pz * (1 - pacc) / 2
      else # zero probability of error > 1
        return pz = 0.0
      end
    end
  end
  return pz
end

# calculate trasnsition matrix
function calcOArray(S::Vector{Vector{Int}}, A::Vector{Vector{Int}}, Z::Vector{Vector{Int}})
  O = zeros(nA,nS,nZ)
  Ol = zeros(nA,nS,nZ)
  Ou = zeros(nA,nS,nZ)
  for (spi,sp) in enumerate(S), (zi,z) in enumerate(Z), (ai,a) in enumerate(A)
    O[ai, spi, zi] = o(a, sp, z, sensor_accuracy)
    neg = o(a, sp, z, sensor_accuracy - sensor_imprecision)
    pos = o(a, sp, z, sensor_accuracy + sensor_imprecision)
    ol = min(neg, pos)
    ou = max(neg, pos)
    ((spi == 1) && (zi == 1) && (ai == 1)) && (println(ou))
    (ol < 0.0) && (ol = max(ol, 0.0 + pϵ_cyber))
    (ou > 1.0) && (ou = min(ou, 1.0 + pϵ_cyber))
    (ou < ol) && (ou = ol + pϵ_cyber / 2)
    Ol[ai, spi, zi] = ol
    Ou[ai, spi, zi] = ou
  end
  for (spi,sp) in enumerate(S), (ai,a) in enumerate(A)
    O[ai, spi,:] = O[ai,spi,:] ./ sum(O[ai,spi,:])
  end
  O, Ol, Ou
end

const oarray_cyber, olarray_cyber, ouarray_cyber = calcOArray(states_cyber, actions_cyber, observations_cyber)

# calculate observation distribution
function calcODist(ns::Int, na::Int, oarray::Array{Float64,3})
  O = Array{SparseCat}(na, ns)
  for ai = 1:na, si = 1:ns
      O[ai,si] = SparseCat(observations_cyber, oarray[ai,si,:])
  end
  O
end

# Nominal transition function distributions
const odist_cyber = calcODist(nS, nA, oarray_cyber)

# Nominal transitions
observation(prob::Union{CyberIPOMDP, CyberPOMDP}, a::Vector{Int}, sp::Vector{Int}) = odist_cyber[action_index(prob, a), state_index(prob, sp)]
observation(prob::Union{CyberIPOMDP, CyberPOMDP}, a::Vector{Int}, sp::Vector{Int}, z::Vector{Int}) = oarray_cyber[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::Union{CyberRPOMDP, CyberRIPOMDP}, a::Vector{Int}, sp::Vector{Int})
    ai, spi = action_index(prob, a), state_index(prob, sp)
    olarray_cyber[ai,spi,:], ouarray_cyber[ai,spi,:]
end
function observation(prob::Union{CyberRPOMDP, CyberRIPOMDP}, a::Vector{Int}, sp::Vector{Int}, z::Vector{Int})
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    olarray_cyber[ai,spi,zi], ouarray_cyber[ai,spi,zi]
end

function dynamics(prob::Union{CyberPOMDP,CyberIPOMDP})
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

function dynamics(prob::Union{CyberRPOMDP,CyberRIPOMDP})
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

function reward(prob::Union{CyberPOMDP, CyberRPOMDP}, s::Vector{Int}, a::Vector{Int})
  u = [0.5, 0.4, 0.1] # utility of each MOE
  r = 0.0
  for i = 1:length(a)
    pacc = sensor_accuracy[i]
    r += u[a[i]] * pacc
  end
  r
end
reward(prob::Union{CyberPOMDP,CyberRPOMDP}, b::Vector{Float64}, s::Vector{Int}, a::Vector{Int}, sp::Vector{Int}) = reward(prob,s,a)

function reward(prob::Union{CyberIPOMDP, CyberRIPOMDP}, b::Vector{Float64}, a::Vector{Int})
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::Union{CyberIPOMDP,CyberRIPOMDP}, b::Vector{Float64}, s::Vector{Int}, a::Vector{Int}, sp::Vector{Int}) = reward(prob,b,a)

function rewardalpha(prob::Union{CyberIPOMDP, CyberRIPOMDP}, b::Vector{Float64}, a::Vector{Int})
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

discount(p::Union{CyberPOMDP,CyberRPOMDP,CyberIPOMDP,CyberRIPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{CyberPOMDP,CyberIPOMDP,CyberRPOMDP,CyberRIPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::Union{CyberPOMDP,CyberIPOMDP}, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

function generate_sor(prob::Union{CyberRPOMDP,CyberRIPOMDP}, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
