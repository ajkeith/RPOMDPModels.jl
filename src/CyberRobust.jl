# Cyber Problem
# POMDP, Robust POMDP, and Robust ρPOMDP formulations

struct CyberPOMDP <: POMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
end

struct CyberRPOMDP <: RPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    usize::Float64
end

struct CyberIPOMDP <: IPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    inforeward::Vector{Vector{Float64}}
end

struct CyberRIPOMDP <: RIPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    usize::Float64
    inforeward::Vector{Vector{Float64}}
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
const nA = nmoe ^ na # total number of actions
A = Vector{Vector{Int}}(nA) # actions, as [asset_1_assignment, ...]
ind = 1
for i = 1:nmoe, j = 1:nmoe
  A[ind] = [i,j]
  ind += 1
end
const actions_cyber = copy(A)

# Observation space
const observations_cyber = copy(states_cyber)
const nZ = nS

bs = Vector{Vector{Float64}}(nS)
bs[1] = vcat(1.0, fill(0.0, nS - 1))
bs[nS] = vcat(fill(0.0, nS - 1), 1.0)
for i = 2:(nS-1)
  bs[i] = vcat(fill(0.0, i - 1), 1.0, fill(0.0, nS - i))
end
const inforeward_cyber = copy(bs)

CyberPOMDP() = CyberPOMDP(discount_cyber)
CyberRPOMDP(err) = CyberRPOMDP(discount_cyber, err)
CyberRPOMDP() = CyberRPOMDP(discount_cyber, 0.025)

CyberIPOMDP(alphas) = CyberIPOMDP(discount_cyber, alphas)
CyberIPOMDP() = CyberIPOMDP(discount_cyber, inforeward_cyber)

CyberRIPOMDP(err::Float64, alphas::Vector{Vector{Float64}}) = CyberRIPOMDP(discount_cyber, err, alphas)
CyberRIPOMDP() = CyberRIPOMDP(discount_cyber, 0.025, inforeward_cyber)

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

initial_state_distribution(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = SparseCat(states_cyber, fill(1/nS, nS))
initial_belief(::Union{CyberPOMDP, CyberRPOMDP, CyberIPOMDP, CyberRIPOMDP}) = fill(1/nS, nS)

# Transitions
const pϵ_cyber = 1e-6 # near-saturation bound
const delt = 0.05 # imprecision in p_detect
const pd = 0.1 # prob of decline
const ps = 0.6 # prob of stay
const pi = 0.3 # prob of improve
const psd = ps + pd # prob of decline or stay at lower border
const psi = ps + pi # prob of stay or improve at upper border

# transition function: probabilty of going from state s to state j
function psj(s::Vector{Int},j::Vector{Int})
  prob = 1
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
        prob = 0
      end
    elseif level_s == lmax # current state is highest level
      if level_j == level_s
        prob = prob * psi
      elseif level_j == level_s - 1
        prob = prob * pd
      else # zero probability of skipping levels
        prob = 0
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
        prob = 0
      end
    end
  end
  prob
end

function psj(s::Array{Int,1},j::Array{Int,1},par,delt)
  # add imprecision (one-sided)
  delt > 0 ? (pd,ps,pi) = min.(par .+ delt, 1 - pϵ_cyber) : (pd,ps,pi) = max.(par .+ delt, 0 + pϵ_cyber)
  psd = ps + pd # prob of decline or stay at lower border
  psi = ps + pi # prob of stay or improve at upper border
  prob = 1
  for i = 1:nmoe
    level_s = s[i]
    level_j = j[i]
    if level_s == lmin # current state is lowest level
      if level_j == level_s
        prob = prob * psd
      elseif level_j == level_s + 1
        prob = prob * pi
      else # zero probability of skipping levels
        prob = 0
      end
    elseif level_s == lmax # current state is highest level
      if level_j == level_s
        prob = prob * psi
      elseif level_j == level_s - 1
        prob = prob * pd
      else # zero probability of skipping levels
        prob = 0
      end
    else # current state is an intermediate level
      if level_j == level_s
        prob = prob * ps
      elseif level_j == level_s + 1
        prob = prob * pi
      elseif level_j == level_s - 1
        prob = prob * pd
      else # zero probability of skipping levels
        prob = 0
      end
    end
  end
  prob
end

# calculate trasnsition matrix
function calcTArray(S::Vector{Vector{Int}}, A::Vector{Vector{Int}},par,delt)
  ns = length(S)
  na = length(A)
  T = zeros(ns, na, ns)
  Tl = zeros(ns, na, ns)
  Tu = zeros(ns, na, ns)
  for (si,s) in enumerate(S), (ji,j) in enumerate(S)
      T[si,:,ji] = psj(s,j)
      Tl[si,:,ji] = psj(s,j,par,-delt)
      Tu[si,:,ji] = psj(s,j,par,delt)
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

const delo = 0.05 # imprecision
const erro = 0.1 # likelihood of one-sided, one-level observation error
const el = erro - delo
const eu = erro + delo
const parl = (el, el^2, 1 - eu, 1 - eu^2)
const paru = (eu, eu^2, 1- el, 1 - el^2)
const par = (el,eu)

# observation function
function o(s::Array{Int,1},a::Array{Int,1},j::Array{Int,1}, eps::Float64)
  prob = 1
  for i = 1:nmoe
    na = sum(a .== i) # total number of assets assigned to moe i
    level_s = s[i]
    level_j = j[i]
    if na > 0
      if level_s == lmin # current state is lowest level
        if level_j == level_s
          prob = prob * (1 - (eps ^ na))
        elseif level_j == level_s + 1
          prob = prob * (eps ^ na) # non-linear effect of multiple assets (i.e. if assets disagree, but one is right, the overall observaiton is right)
        else # zero probability of error > 1
          prob = 0
        end
      elseif level_s == lmax # current state is highest level
        if level_j == level_s
          prob = prob * (1 - (eps ^ na))
        elseif level_j == level_s - 1
          prob = prob * (eps ^ na)
        else # zero probability of error > 1
          prob = 0
        end
      else # current state is an intermediate level
        if level_j == level_s
          prob = prob * (1 - 2 * (eps ^ na))
        elseif level_j == level_s + 1
          prob = prob * (eps ^ na)
        elseif level_j == level_s - 1
          prob = prob * (eps ^ na)
        else # zero probability of error > 1
          prob = 0
        end
      end
    else
      level_s == level_j ? true : prob = 0 # unassigned moe can't change observation status
    end
  end
  prob
end

function o(s::Vector{Int},a::Vector{Int},j::Vector{Int},par,del::Float64)
  del > 0 ? (e1, e2) = par : (e2, e1) = par
  prob = 1
  for i = 1:nmoe
    na = sum(a .== i) # total number of assets assigned to moe i
    level_s = s[i]
    level_j = j[i]
    if na > 0
      if level_s == lmin # current state is lowest level
        if level_j == level_s
          prob = prob * (1 - (e1 ^ na))
        elseif level_j == level_s + 1
          prob = prob * (e2 ^ na) # non-linear effect of multiple assets (i.e. if assets disagree, but one is right, the overall observaiton is right)
        else # zero probability of error > 1
          prob = 0
        end
      elseif level_s == lmax # current state is highest level
        if level_j == level_s
          prob = prob * (1 - (e1 ^ na))
        elseif level_j == level_s - 1
          prob = prob * (e2 ^ na)
        else # zero probability of error > 1
          prob = 0
        end
      else # current state is an intermediate level
        if level_j == level_s
          prob = prob * (1 - 2 * (e1 ^ na))
        elseif level_j == level_s + 1
          prob = prob * (e2 ^ na)
        elseif level_j == level_s - 1
          prob = prob * (e2 ^ na)
        else # zero probability of error > 1
          prob = 0
        end
      end
    else
      level_s == level_j ? true : prob = 0 # unassigned moe can't change observation status
    end
  end
  prob
end

# calculate trasnsition matrix
function calcOArray(S::Vector{Vector{Int}}, A::Vector{Vector{Int}}, Z::Vector{Vector{Int}}, par::Tuple{Float64,Float64})
  O = zeros(nA,nS,nZ)
  Ol = zeros(nA,nS,nZ)
  Ou = zeros(nA,nS,nZ)
  for (si,s) in enumerate(S), (zi,z) in enumerate(Z), (ai,a) in enumerate(A)
    O[ai, si, zi] = o(s, a, z, erro)
    neg = o(s, a, z, par, -delo)
    pos = o(s, a, z, par, delo)
    Ol[ai, si, zi] = min(neg, pos)
    Ou[ai, si, zi] = max(neg, pos)
  end
  O, Ol, Ou
end

const oarray_cyber, olarray_cyber, ouarray_cyber = calcOArray(states_cyber, actions_cyber, observations_cyber, par)

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
  r = 0
  for i = 1:nmoe
    nassign = sum(a .== i) # total number of assets assigned to moe i
    if s[i] == lmin
      nassign > 0 ? r += u[i] * (1 - (erro ^ nassign)) : r += u[i] * psd
    elseif s[i] == lmax
      nassign > 0 ? r += u[i] * (1 - (erro ^ nassign)) : r += u[i] * psi
    else
      nassign > 0 ? r += u[i] * (1 - 2*(erro ^ nassign)) : r += u[i] * ps
    end
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
