# Rock Diagnosis
# Araya-Lopez 2013 (Active Diagnosis Through Information-Lookahead Planning)

struct RockIPOMDP <: IPOMDP{Int, Symbol, Symbol}
    tdist::Array
    tarray::Array{Float64}
    odist::Array
    oarray::Array{Float64}
    inforeward::Vector{Vector{Float64}}
    discount::Float64
end

struct RockRIPOMDP <: RIPOMDP{Int, Symbol, Symbol}
    tarrayl::Array{Float64}
    tarrayu::Array{Float64}
    oarrayl::Array{Float64}
    oarrayu::Array{Float64}
    inforeward::Vector{Vector{Float64}}
    discount::Float64
    usize::Float64
end

# rock loation, starting location, rock status, and max sensor efficiency
const lrock = 2
const lstart = 1
const srock = :good
const seff = 1.0

const discount_rock = 0.95
const states_rock = [10, 11, 20, 21] # location: 1 or 2, rock status: 0 (bad) or 1 (good)
const actions_rock = [:left, :right, :check]
const observations_rock = [:good, :bad, :none]
const inforeward_rock = [[1.0, -1/3, -1/3, -1/3],
                         [-1/3, 1.0, -1/3, -1/3],
                         [-1/3, -1/3, 1.0, -1/3],
                         [-1/3, -1/3, -1/3, 1.0]]
const initbelief_rock = [0.5, 0.5, 0.0, 0.0]
const initdist_rock = SparseCat(states_rock, [1.0, 0.0, 0.0, 0.0])
const pϵ_rock = 1e-6

# Nominal transition function distributions (deterministic)
const tdist_rock = [hcat(SparseCat(states_rock, [1.0, 0.0, 0.0, 0.0]),  #1B, L to 1B
                    SparseCat(states_rock, [0.0, 0.0, 1.0, 0.0]),  #1B, R to 2B
                    SparseCat(states_rock, [1.0, 0.0, 0.0, 0.0])); #1B, C to 1B
                    hcat(SparseCat(states_rock, [0.0, 1.0, 0.0, 0.0]),  #1G, L to 1G
                    SparseCat(states_rock, [0.0, 0.0, 0.0, 1.0]),  #1G, R to 2G
                    SparseCat(states_rock, [0.0, 1.0, 0.0, 0.0])); #1G, C to 1G
                    hcat(SparseCat(states_rock, [1.0, 0.0, 0.0, 0.0]),  #2B, L to 1B
                    SparseCat(states_rock, [0.0, 0.0, 1.0, 0.0]),  #2B, R to 2B
                    SparseCat(states_rock, [0.0, 0.0, 1.0, 0.0])); #2B, C to 2B
                    hcat(SparseCat(states_rock, [0.0, 1.0, 0.0, 0.0]),  #2G, L to 1G
                    SparseCat(states_rock, [0.0, 0.0, 0.0, 1.0]),  #2G, R to 2G
                    SparseCat(states_rock, [0.0, 0.0, 0.0, 1.0]))] #2G, C to 2G

# Nominal transition function array tarray_rock[s,a,s'] = Pr(s' | s ,a)
const tarray_rock = cat(3, [1.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0], # s' = 1B
                           [0.0 0.0 0.0; 1.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 0.0], # s' = 1G
                           [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 1.0 1.0; 0.0 0.0 0.0], # s' = 2B
                           [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 1.0 1.0]) # s' = 2G

# Nominal observation function distributions
function peff(xs::Int, ys::Int, xr::Int, yr::Int, maxeff::Float64)
   dist = sqrt((yr -ys)^2 + (xr - xs)^2)
   return maxeff * (1 + exp(-dist)) / 2
end
peff(xs::Int, ys::Int, xr::Int, yr::Int) = peff(xs, ys, xr, yr, 1.0)
const ps1 = peff(1, 0, 2, 0, seff)
const ps2 = peff(2, 0, 2, 0, seff)

const odist_rock = [hcat(SparseCat(observations_rock, [0.0, 0.0, 1.0]),  # L, 1B
                   SparseCat(observations_rock, [0.0, 0.0, 1.0]),  # L, 1G
                   SparseCat(observations_rock, [0.0, 0.0, 1.0]),  # L, 2B
                   SparseCat(observations_rock, [0.0, 0.0, 1.0])); # L, 2G
                   hcat(SparseCat(observations_rock, [0.0, 0.0, 1.0]),  # R, 1B
                   SparseCat(observations_rock, [0.0, 0.0, 1.0]),  # R, 1G
                   SparseCat(observations_rock, [0.0, 0.0, 1.0]),  # R, 2B
                   SparseCat(observations_rock, [0.0, 0.0, 1.0])); # R, 2G
                   hcat(SparseCat(observations_rock, [1 - ps1, ps1, 0.0]),  # C, 1B
                   SparseCat(observations_rock, [ps1, 1 - ps1, 0.0]),  # C, 1G
                   SparseCat(observations_rock, [1 - ps2, ps2, 0.0]),  # C, 2B
                   SparseCat(observations_rock, [ps2, 1 - ps2, 0.0]))] # C, 2G

# Nominal observation function array oarray_rock[a,sp,z] = Pr(z|a,sp)
const oarray_rock = cat(3, [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 1-ps1 ps1 1-ps2 ps2], # z = good
                          [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; ps1 1-ps1 ps2 1-ps2], # z = bad
                          [1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0])   # z = nothing

function buildor(array::Array{Float64}, ps1::Float64, ps2::Float64, uncsize::Float64)
  ps1_l = max(ps1 - uncsize, 0.0 + pϵ_rock/2)
  ps1_u = min(ps1 + uncsize, 1.0 - pϵ_rock/2)
  ps2_l = max(ps2 - uncsize, 0.0 + pϵ_rock/2)
  ps2_u = min(ps2 + uncsize, 1.0 - pϵ_rock/2)
  o_l = cat(3, [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 1-ps1_l ps1_l 1-ps2_l ps2_l],
               [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; ps1_l 1-ps1_l ps2_l 1-ps2_l],
               [1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0])
  o_u = cat(3, [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 1-ps1_u ps1_u 1-ps2_u ps2_u],
               [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; ps1_u 1-ps1_u ps2_u 1-ps2_u],
               [1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0])
  min.(o_l, o_u), max.(o_l, o_u)
end

function RockIPOMDP(alphas::Vector{Vector{Float64}})
    tdist = tdist_rock
    tarray = tarray_rock
    odist = odist_rock
    oarray = oarray_rock
    discount = discount_rock
    RockIPOMDP(tdist, tarray, odist, oarray, alphas, discount)
end
RockIPOMDP() = RockIPOMDP(inforeward_rock)

function RockRIPOMDP(alphas::Vector{Vector{Float64}}, usize::Float64)
    # tarrayl = max.(tarray_rock - pϵ_rock, 0.0 + pϵ_rock / 2)
    # tarrayu = min.(tarray_rock + pϵ_rock, 1.0 - pϵ_rock / 2)
    tarrayl = max.(tarray_rock - pϵ_rock, 0.0)
    tarrayu = min.(tarray_rock + pϵ_rock, 1.0)
    oarrayl, oarrayu = buildor(oarray_rock, ps1, ps2, usize)
    # oarrayl = max.(oarrayl - pϵ_rock, 0.0 + pϵ_rock / 2)
    # oarrayu = min.(oarrayu + pϵ_rock, 1.0 - pϵ_rock / 2)
    oarrayl = max.(oarrayl - pϵ_rock, 0.0)
    oarrayu = min.(oarrayu + pϵ_rock, 1.0)
    discount = discount_rock
    RockRIPOMDP(tarrayl, tarrayu, oarrayl, oarrayu, alphas, discount, usize)
end
RockRIPOMDP(alphas::Vector{Vector{Float64}}) = RockRIPOMDP(alphas, 0.025)
RockRIPOMDP(err::Float64) = RockRIPOMDP(inforeward_rock, err)
RockRIPOMDP() = RockRIPOMDP(0.025)

states(::Union{RockIPOMDP, RockRIPOMDP}) = states_rock
actions(::Union{RockIPOMDP, RockRIPOMDP}) = actions_rock
observations(::Union{RockIPOMDP, RockRIPOMDP}) = observations_rock

n_states(::Union{RockIPOMDP, RockRIPOMDP}) = 4
n_actions(::Union{RockIPOMDP, RockRIPOMDP}) = 3
n_observations(::Union{RockIPOMDP, RockRIPOMDP}) = 3

function state_index(::Union{RockIPOMDP, RockRIPOMDP}, s::Int)
    if s == 10
        return 1
    elseif s == 11
        return 2
    elseif s == 20
        return 3
    elseif s== 21
        return 4
    else
        return nothing
    end
end

function action_index(::Union{RockIPOMDP, RockRIPOMDP}, a::Symbol)
    if a == :left
        return 1
    elseif a == :right
        return 2
    elseif a == :check
        return 3
    else
        return nothing
    end
end

function observation_index(::Union{RockIPOMDP, RockRIPOMDP}, z::Symbol)
    if z == :good
        return 1
    elseif z == :bad
        return 2
    elseif z == :none
        return 3
    else
        return nothing
    end
end

obs_index(prob::Union{RockIPOMDP, RockRIPOMDP}, z::Symbol) = observation_index(prob, z)

initial_state_distribution(::Union{RockIPOMDP, RockRIPOMDP}) = initdist_rock
initial_belief(::Union{RockIPOMDP, RockRIPOMDP}) = initbelief_rock

# Nominal transitions
transition(prob::RockIPOMDP, s::Int, a::Symbol) = prob.tdist[state_index(prob, s), action_index(prob, a)]
transition(prob::RockIPOMDP, s::Int, a::Symbol, sp::Int) = prob.tarray[state_index(prob, s), action_index(prob, a), state_index(prob, sp)]

# Robust transitions
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PtLower = Plower(t|s,a) = t_lower[s,a,sp]
function transition(prob::RockRIPOMDP, s::Int, a::Symbol)
    si, ai = state_index(prob, s), action_index(prob, a)
    prob.tarrayl[si, ai, :], prob.tarrayu[si, ai, :]
end
function transition(prob::RockRIPOMDP, s::Int, a::Symbol, sp::Int)
    si, ai, spi = state_index(prob, s), action_index(prob, a), state_index(prob, sp)
    prob.tarrayl[si, ai, spi], prob.tarrayu[si, ai, spi]
end

# Nominal observations
observation(prob::RockIPOMDP, a::Symbol, sp::Int) = prob.odist[action_index(prob, a), state_index(prob, sp)]
observation(prob::RockIPOMDP, a::Symbol, sp::Int, z::Symbol) = prob.oarray[action_index(prob, a), state_index(prob, sp), observation_index(prob, z)]

# Robust observations
# P(z,t|s,a) = P(z|s,a,t)P(t|s,a)
# PztLower = PzLower * PtLower
# PzLower = Plower(z|s,a,t) = o_lower[a,sp,z]
function observation(prob::RockRIPOMDP, a::Symbol, sp::Int)
    ai, spi = action_index(prob, a), state_index(prob, sp)
    prob.oarrayl[ai,spi,:], prob.oarrayu[ai,spi,:]
end
function observation(prob::RockRIPOMDP, a::Symbol, sp::Int, z::Symbol)
    ai, spi, zi = action_index(prob, a), state_index(prob, sp), observation_index(prob, z)
    prob.oarrayl[ai,spi,zi], prob.oarrayu[ai,spi,zi]
end

function dynamics(prob::RockIPOMDP)
    ns = n_states(prob)
    nz = n_observations(prob)
    na = n_actions(prob)
    tarr = states(prob)
    sarr = states(prob)
    aarr = actions(prob)
    zarr = observations(prob)
    p = Array{Float64}(ns,nz,ns,na)
    for tind in 1:ns, zind in 1:nz, sind in 1:ns, aind in 1:na
        pt = transition(prob, sarr[sind], aarr[aind], tarr[tind])
        pz = observation(prob, aarr[aind], tarr[tind], zarr[zind])
        p[tind, zind, sind, aind] = pz * pt
    end
    p
end

function dynamics(prob::RockRIPOMDP)
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
function reward(prob::Union{RockIPOMDP, RockRIPOMDP}, b::Vector{Float64}, a::Symbol)
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::Union{RockIPOMDP,RockRIPOMDP}, b::Vector{Float64}, s::Int, a::Symbol, sp::Int) = reward(prob,b,a)

function rewardalpha(prob::Union{RockIPOMDP, RockRIPOMDP}, b::Vector{Float64}, a::Symbol)
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

discount(p::Union{RockIPOMDP,RockRIPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{RockIPOMDP,RockRIPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::RockIPOMDP, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

function generate_sor(prob::RockRIPOMDP, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
