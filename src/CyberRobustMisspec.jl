# Cyber Off Nominal Test Problem

struct CyberTestIPOMDP <: IPOMDP{Vector{Int}, Vector{Int}, Vector{Int}}
    discount::Float64
    inforeward::Vector{Vector{Float64}}
    b0::Vector{Float64}
end
CyberTestIPOMDP(alphas) = CyberTestIPOMDP(discount_cyber, alphas, b0_cyber)
CyberTestIPOMDP() = CyberTestIPOMDP(discount_cyber, inforeward_cyber, b0_cyber)

states(::CyberTestIPOMDP) = states_cyber
actions(::CyberTestIPOMDP) = actions_cyber
observations(::CyberTestIPOMDP) = observations_cyber

n_states(::CyberTestIPOMDP) = nS
n_actions(::CyberTestIPOMDP) = nA
n_observations(::CyberTestIPOMDP) = nZ

state_index(::CyberTestIPOMDP, s::Vector{Int}) = find(x -> x == s, states_cyber)[1]
action_index(::CyberTestIPOMDP, a::Vector{Int}) = find(x -> x == a, actions_cyber)[1]
observation_index(::CyberTestIPOMDP, z::Vector{Int}) = find(x -> x == z, observations_cyber)[1]
obs_index(prob::CyberTestIPOMDP, z::Vector{Int}) = observation_index(prob, z)

e11 = zeros(nS); e11[1] = 1.0;
const e1_test = copy(e11)
initial_state_distribution(::CyberTestIPOMDP) = SparseCat(states_cyber, e1_test)
initial_belief(prob::CyberTestIPOMDP) = prob.b0

# Transitions
const pd_test = 0.05 # prob of decline
const ps_test = 0.45 # prob of stay
const pi_test = 0.5 # prob of improve
const psd_test = ps_test + pd_test # prob of decline or stay at lower border
const psi_test = ps_test + pi_test # prob of stay or improve at upper border

# transition function: probabilty of going from state s to state j
function psj_test(s::Vector{Int},j::Vector{Int})
  prob = 1
  count = 0
  for i = 1:nmoe
    level_s = s[i]
    level_j = j[i]
    if level_s == lmin # current state is lowest level
      if level_j == level_s
        prob = prob * psd_test
      elseif level_j == level_s + 1
        prob = prob * pi_test
      else # zero probability of skipping levels
        prob = 0
      end
    elseif level_s == lmax # current state is highest level
      if level_j == level_s
        prob = prob * psi_test
      elseif level_j == level_s - 1
        prob = prob * pd_test
      else # zero probability of skipping levels
        prob = 0
      end
    else # current state is an intermediate level
      if level_j == level_s
        prob = prob * ps_test
        count += 1
      elseif level_j == level_s + 1
        prob = prob * pi_test
      elseif level_j == level_s - 1
        prob = prob * pd_test
      else # zero probability of skipping levels
        prob = 0
      end
    end
  end
  prob
end

function psj_test(s::Array{Int,1},j::Array{Int,1},parl_test,delt)
  # add imprecision (one-sided)
  delt > 0 ? (pd_test,ps_test,pi_test) = min.(parl_test .+ delt, 1 - pϵ_cyber) : (pd_test,ps_test,pi_test) = max.(parl_test .+ delt, 0 + pϵ_cyber)
  psd_test = min(ps_test + pd_test, 1 - pϵ_cyber) # prob of decline or stay at lower border
  psi_test = min(ps_test + pi_test, 1 - pϵ_cyber) # prob of stay or improve at upper border
  prob = 1
  for i = 1:nmoe
    level_s = s[i]
    level_j = j[i]
    if level_s == lmin # current state is lowest level
      if level_j == level_s
        prob = prob * psd_test
      elseif level_j == level_s + 1
        prob = prob * pi_test
      else # zero probability of skipping levels
        prob = 0
      end
    elseif level_s == lmax # current state is highest level
      if level_j == level_s
        prob = prob * psi_test
      elseif level_j == level_s - 1
        prob = prob * pd_test
      else # zero probability of skipping levels
        prob = 0
      end
    else # current state is an intermediate level
      if level_j == level_s
        prob = prob * ps_test
      elseif level_j == level_s + 1
        prob = prob * pi_test
      elseif level_j == level_s - 1
        prob = prob * pd_test
      else # zero probability of skipping levels
        prob = 0
      end
    end
  end
  prob
end

# calculate trasnsition matrix
function calcTArray_test(S::Vector{Vector{Int}}, A::Vector{Vector{Int}},parl_test,delt)
  ns = length(S)
  na = length(A)
  T = zeros(ns, na, ns)
  Tl = zeros(ns, na, ns)
  Tu = zeros(ns, na, ns)
  for (si,s) in enumerate(S), (ji,j) in enumerate(S)
      T[si,:,ji] = psj_test(s,j)
      tl = psj_test(s,j,parl_test,-delt)
      tu = psj_test(s,j,parl_test,delt)
      (tl == tu) && (tu = tu + pϵ_cyber)
      Tl[si,:,ji] = tl
      Tu[si,:,ji] = tu
  end
  for (si,s) in enumerate(S), (ai,a) in enumerate(A)
    T[si,ai,:] = T[si,ai,:] ./ sum(T[si,ai,:])
  end
  T, Tl, Tu
end

# Nominal transition function array tarray_cyber_test[s,a,s'] = Pr(s' | s ,a)
const tarray_cyber_test, tlarray_cyber_test, tuarray_cyber_test = calcTArray_test(states_cyber, actions_cyber, (pd_test,ps_test,pi_test), delt)

# calculate trasnsition distribution
function calcTDist_test(ns::Int, na::Int, tarray::Array{Float64,3})
  T = Array{SparseCat}(ns, na)
  for si = 1:ns, ai = 1:na
      T[si,ai] = SparseCat(states_cyber, tarray[si,ai,:])
  end
  T
end

# Nominal transition function distributions
const tdist_cyber_test = calcTDist_test(nS, nA, tarray_cyber_test)

# Nominal transitions
transition(prob::CyberTestIPOMDP, s::Vector{Int}, a::Vector{Int}) = tdist_cyber_test[state_index(prob, s), action_index(prob, a)]
transition(prob::CyberTestIPOMDP, s::Vector{Int}, a::Vector{Int}, sp::Vector{Int}) = tarray_cyber_test[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Observation function
const delo_test = 0.05 # imprecision
const erro_test = 0.35 # likelihood of one-sided, one-level observation erro_testr
const el_test = erro_test - delo_test
const eu_test = erro_test + delo_test
const parl_test = (el_test, el_test^2, 1 - eu_test, 1 - eu_test^2)
const paru_test = (eu_test, eu_test^2, 1- el_test, 1 - el_test^2)
const par_test = (el_test,eu_test)

# observation function
function o_test(a::Vector{Int}, sp::Vector{Int}, z::Vector{Int}, eps::Float64)
  prob = 1.0
  for i = 1:nmoe
    na = sum(a .== i) # total number of assets assigned to moe i
    level_sp = sp[i]
    (na > 0) && (level_z = z[find(a .== i)[1]])
    if na == 0
      nothing
    else
      any(z[a .== i] .!= level_z) && return prob = pϵ_cyber
      if level_sp == lmin # current state is lowest level
        if level_z == level_sp
          prob = prob * (1 - (eps ^ na))
        elseif level_z == level_sp + 1
          prob = prob * (eps ^ na) # non-linear effect of multiple assets (i.e. if assets disagree, but one is right, the overall observaiton is right)
        else # zero probability of error > 1
          return prob = pϵ_cyber
        end
      elseif level_sp == lmax # current state is highest level
        if level_z == level_sp
          prob = prob * (1 - (eps ^ na))
        elseif level_z == level_sp - 1
          prob = prob * (eps ^ na)
        else # zero probability of error > 1
          return prob = pϵ_cyber
        end
      else # current state is an intermediate level
        if level_z == level_sp
          prob = prob * (1 - 2 * (eps ^ na))
        elseif level_z == level_sp + 1
          prob = prob * (eps ^ na)
        elseif level_z == level_sp - 1
          prob = prob * (eps ^ na)
        else # zero probability of error > 1
          return prob = pϵ_cyber
        end
      end
    end
  end
  return prob
end

function o_test(a::Vector{Int}, sp::Vector{Int}, z::Vector{Int}, par_test, del::Float64)
  del > 0 ? (e1, e2) = par_test : (e2, e1) = par_test
  prob = 1.0
  for i = 1:nmoe
    na = sum(a .== i) # total number of assets assigned to moe i
    level_sp = sp[i]
    (na > 0) && (level_z = z[find(a .== i)[1]])
    if na == 0
      nothing
    else
      any(z[a .== i] .!= level_z) && return prob = pϵ_cyber
      if level_sp == lmin # current state is lowest level
        if level_z == level_sp
          prob = prob * (1 - (e1 ^ na))
        elseif level_z == level_sp + 1
          prob = prob * (e2 ^ na) # non-linear effect of multiple assets (i.e. if assets disagree, but one is right, the overall observaiton is right)
        else # zero probability of error > 1
          return prob = pϵ_cyber
        end
      elseif level_sp == lmax # current state is highest level
        if level_z == level_sp
          prob = prob * (1 - (e1 ^ na))
        elseif level_z == level_sp - 1
          prob = prob * (e2 ^ na)
        else # zero probability of error > 1
          return prob = pϵ_cyber
        end
      else # current state is an intermediate level
        if level_z == level_sp
          prob = prob * (1 - 2 * (e1 ^ na))
        elseif level_z == level_sp + 1
          prob = prob * (e2 ^ na)
        elseif level_z == level_sp - 1
          prob = prob * (e2 ^ na)
        else # zero probability of error > 1
          return prob = pϵ_cyber
        end
      end
    end
  end
  return prob
end

# calculate trasnsition matrix
function calcOArray_test(S::Vector{Vector{Int}}, A::Vector{Vector{Int}}, Z::Vector{Vector{Int}}, par_test::Tuple{Float64,Float64})
  O = zeros(nA,nS,nZ)
  Ol = zeros(nA,nS,nZ)
  Ou = zeros(nA,nS,nZ)
  for (spi,sp) in enumerate(S), (zi,z) in enumerate(Z), (ai,a) in enumerate(A)
    O[ai, spi, zi] = o(a, sp, z, erro_test)
    neg = o_test(a, sp, z, par_test, -delo_test)
    pos = o_test(a, sp, z, par_test, delo_test)
    ol = min(neg, pos)
    ou = max(neg, pos)
    (ol == ou) && (ou = ou + pϵ_cyber)
    Ol[ai, spi, zi] = ol
    Ou[ai, spi, zi] = ou
  end
  for (spi,sp) in enumerate(S), (ai,a) in enumerate(A)
    O[ai, spi,:] = O[ai,spi,:] ./ sum(O[ai,spi,:])
  end
  O, Ol, Ou
end

const oarray_cyber_test, olarray_cyber_test, ouarray_cyber_test = calcOArray_test(states_cyber, actions_cyber, observations_cyber, par_test)

# calculate observation distribution
function calcODist_test(ns::Int, na::Int, oarray::Array{Float64,3})
  O = Array{SparseCat}(na, ns)
  for ai = 1:na, si = 1:ns
      O[ai,si] = SparseCat(observations_cyber, oarray[ai,si,:])
  end
  O
end

# Nominal transition function distributions
const odist_cyber_test = calcODist_test(nS, nA, oarray_cyber_test)

# Nominal transitions
observation(prob::CyberTestIPOMDP, a::Vector{Int}, sp::Vector{Int}) = odist_cyber_test[action_index(prob, a), state_index(prob, sp)]
observation(prob::CyberTestIPOMDP, a::Vector{Int}, sp::Vector{Int}, z::Vector{Int}) = oarray_cyber_test[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

function dynamics(prob::CyberTestIPOMDP)
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
