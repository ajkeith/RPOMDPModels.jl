# Simple Robust IPOMDP and RIPOMDP for testing

struct SimpleIPOMDP <: IPOMDP{Symbol, Symbol, Symbol}
    ps::Float64
    pb::Float64
    pst::Float64
    pbt::Float64
    discount::Float64
    inforeward::Vector{Vector{Float64}}
end

struct SimpleRIPOMDP <: RIPOMDP{Symbol, Symbol, Symbol}
    ps::Float64
    pb::Float64
    pst::Float64
    pbt::Float64
    discount::Float64
    inforeward::Vector{Vector{Float64}}
    usize::Float64
end

const states_simple = [:left, :right]
const actions_simple = [:single, :both]
const observations_simple = [:L,:R,:LL,:LR,:RL,:RR]
const inforeward_simple = [[1.0, -1.0], [-1.0, 1.0]]
const discount_simple = 0.75
const ps_simple = 0.8 # probability of correct detect single
const pb_simple = 0.6 # probability of correct detect both
const err_simple = 0.4 # ambiguity size
const pst_simple = 0.6 # probability of stay after single action
const pbt_simple = 0.9 # probability of stay after both action

SimpleIPOMDP(ps::Float64, pb::Float64, pst::Float64, pbt::Float64) = SimpleIPOMDP(ps, pb, pst, pbt, discount_simple, inforeward_simple)
SimpleIPOMDP(ps::Float64, pb::Float64) = SimpleIPOMDP(ps, pb, pst_simple, pbt_simple, discount_simple, inforeward_simple)
SimpleIPOMDP() = SimpleIPOMDP(ps_simple, pb_simple, pst_simple, pbt_simple, discount_simple, inforeward_simple)

SimpleRIPOMDP(ps::Float64, pb::Float64, pst::Float64, pbt::Float64, err::Float64) = SimpleRIPOMDP(ps, pb, pst, pbt, discount_simple, inforeward_simple, err)
SimpleRIPOMDP(ps::Float64, pb::Float64, err::Float64) = SimpleRIPOMDP(ps, pb, pst_simple, pbt_simple, discount_simple, inforeward_simple, err)
SimpleRIPOMDP(ps::Float64, pb::Float64) = SimpleRIPOMDP(ps, pb, pst_simple, pbt_simple, discount_simple, inforeward_simple, err_simple)
SimpleRIPOMDP(err::Float64) = SimpleRIPOMDP(ps_simple, pb_simple, pst_simple, pbt_simple, discount_simple, inforeward_simple, err)
SimpleRIPOMDP() = SimpleRIPOMDP(ps_simple, pb_simple, pst_simple, pbt_simple, discount_simple, inforeward_simple, err_simple)

states(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = states_simple
actions(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = actions_simple
observations(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = observations_simple

n_states(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = length(states_simple)
n_actions(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = length(actions_simple)
n_observations(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = length(observations_simple)

state_index(::Union{SimpleIPOMDP,SimpleRIPOMDP}, s::Symbol) = find(x -> x == s, states_simple)[1]
action_index(::Union{SimpleIPOMDP,SimpleRIPOMDP}, a::Symbol) = find(x -> x == a, actions_simple)[1]
observation_index(::Union{SimpleIPOMDP,SimpleRIPOMDP}, z::Symbol) = find(x -> x == z, observations_simple)[1]
obs_index(prob::Union{SimpleIPOMDP,SimpleRIPOMDP}, z::Symbol) = observation_index(prob, z)

initial_state_distribution(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = SparseCat([:left, :right], [1.0, 0.0])
initial_belief(::Union{SimpleIPOMDP,SimpleRIPOMDP}) = [0.5, 0.5]

# Transition functions
# Nominal transition function distributions
const pϵ_simple = 1e-6

function transition(prob::SimpleIPOMDP, s::Symbol, a::Symbol)
    pst = prob.pst
    pbt = prob.pbt
    if s == :left
        if a == :single
            return SparseCat([:left, :right], [pst, 1-pst])
        else
            return SparseCat([:left, :right], [pbt, 1-pbt])
        end
    else
        if a == :single
            return SparseCat([:left, :right], [1-pst, pst])
        else
            return SparseCat([:left, :right], [1-pbt, pbt])
        end
    end
    return nothing
end

function transition(prob::SimpleIPOMDP, s::Symbol, a::Symbol, sp::Symbol)
    pst = prob.pst
    pbt = prob.pbt
    if s == :left
        if a == :single
            return sp == :left ? pst : 1-pst
        else
            return sp == :left ? pbt : 1-pbt
        end
    else
        if a == :single
            return sp == :left ? 1-pst : pst
        else
            return sp == :left ? 1-pbt : pbt
        end
    end
end

function transition(prob::SimpleRIPOMDP, s::Symbol, a::Symbol)
    pst = prob.pst
    pbt = prob.pbt
    if s == :left
        if a == :single
            plower = max.([pst - prob.usize, 1-pst-prob.usize/10], 0.0 + pϵ_simple)
            pupper = min.([pst + prob.usize/10, 1-pst+prob.usize], 1.0 - pϵ_simple)
            return plower, pupper
        else
            plower = max.([pbt, 1-pbt] - prob.usize/10, 0.0 + pϵ_simple)
            pupper = min.([pbt, 1-pbt] + prob.usize/10, 1.0 - pϵ_simple)
            return plower, pupper
        end
    else
        if a == :single
            plower = max.([1-pst - prob.usize/10, pst - prob.usize], 0.0 + pϵ_simple)
            pupper = min.([1-pst + prob.usize, pst + prob.usize/10], 1.0 - pϵ_simple)
            return plower, pupper
        else
            plower = max.([1-pbt, pbt] - prob.usize/10, 0.0 + pϵ_simple)
            pupper = min.([1-pbt, pbt] + prob.usize/10, 1.0 - pϵ_simple)
            return plower, pupper
        end
    end
    return nothing
end

function transition(prob::SimpleRIPOMDP, s::Symbol, a::Symbol, sp::Symbol)
    pst = prob.pst
    pbt = prob.pbt
    if s == :left
        if a == :single
            if sp == :left
                plower = max.(pst - prob.usize, 0.0 + pϵ_simple)
                pupper = min.(pst + prob.usize/10, 1.0 - pϵ_simple)
                return plower, pupper
            else
                plower = max.(1-pst - prob.usize/10, 0.0 + pϵ_simple)
                pupper = min.(1-pst + prob.usize, 1.0 - pϵ_simple)
                return plower, pupper
            end
        else
            if sp == :left
                plower = max.(pbt - prob.usize/10, 0.0 + pϵ_simple)
                pupper = min.(pbt + prob.usize, 1.0 - pϵ_simple)
                return plower, pupper
            else
                plower = max.(1-pbt - prob.usize, 0.0 + pϵ_simple)
                pupper = min.(1-pbt + prob.usize/10, 1.0 - pϵ_simple)
                return plower, pupper
            end
        end
    else
        if a == :single
            if sp == :left
                plower = max.(1-pst - prob.usize/10, 0.0 + pϵ_simple)
                pupper = min.(1-pst + prob.usize, 1.0 - pϵ_simple)
                return plower, pupper
            else
                plower = max.(pst - prob.usize, 0.0 + pϵ_simple)
                pupper = min.(pst + prob.usize/10, 1.0 - pϵ_simple)
                return plower, pupper
            end
        else
            if sp == :left
                plower = max.(1-pbt - prob.usize, 0.0 + pϵ_simple)
                pupper = min.(1-pbt + prob.usize/10, 1.0 - pϵ_simple)
                return plower, pupper
            else
                plower = max.(pbt - prob.usize/10, 0.0 + pϵ_simple)
                pupper = min.(pbt + prob.usize, 1.0 - pϵ_simple)
                return plower, pupper
            end
        end
    end
end

function observation(prob::SimpleIPOMDP, a::Symbol, sp::Symbol)
    ps = prob.ps
    pb = prob.pb
    if a == :single
        if sp == :left
            return SparseCat([:L, :R, :LL, :LR, :RL, :RR], [ps-0.001, 1-ps, 0.001/4, 0.001/4, 0.001/4, 0.001/4])
        else
            return SparseCat([:L, :R, :LL, :LR, :RL, :RR], [1-ps, ps-0.001, 0.001/4, 0.001/4, 0.001/4, 0.001/4])
        end
    else
        if sp == :left
            return SparseCat([:L, :R, :LL, :LR, :RL, :RR], [0.001/2, 0.001/2, pb*pb-0.001, (1-pb)*pb, pb*(1-pb), (1-pb)*(1-pb)])
        else
            return SparseCat([:L, :R, :LL, :LR, :RL, :RR], [0.001/2, 0.001/2, (1-pb)*(1-pb), (1-pb)*pb, pb*(1-pb), pb*pb-0.001])
        end
    end
    return nothing
end

function observation(prob::SimpleIPOMDP, a::Symbol, sp::Symbol, z::Symbol)
    pdf(observation(prob, a, sp), z)
end

function observation(prob::SimpleRIPOMDP, a::Symbol, sp::Symbol)
    ps = prob.ps
    pb = prob.pb
    if a == :single
        if sp == :left
            plower = max.([ps-0.001 - prob.usize, 1-ps - prob.usize/10, 0, 0, 0, 0], 0.0 + pϵ_simple)
            pupper = min.([ps-0.001 + prob.usize/10, 1-ps + prob.usize, 0.001/4, 0.001/4, 0.001/4, 0.001/4], 1.0 - pϵ_simple)
            return plower, pupper
        else
            plower = max.([1-ps - prob.usize/10, ps-0.001-prob.usize, 0, 0, 0, 0], 0.0 + pϵ_simple)
            pupper = min.([1-ps + prob.usize, ps-0.001+prob.usize/10, 0.001/4, 0.001/4, 0.001/4, 0.001/4], 1.0 - pϵ_simple)
            return plower, pupper
        end
    else
        if sp == :left
            plower = max.([0, 0, pb*pb-0.001, (1-pb)*pb, pb*(1-pb), (1-pb)*(1-pb)] - prob.usize/10, 0.0 + pϵ_simple)
            pupper = min.([0.001/2, 0.001/2, pb*pb-0.001, (1-pb)*pb, pb*(1-pb), (1-pb)*(1-pb)] + prob.usize/10, 1.0 - pϵ_simple)
            return plower, pupper
        else
            plower = max.([0, 0, (1-pb)*(1-pb), (1-pb)*pb, pb*(1-pb), pb*pb-0.001] - prob.usize/10, 0.0 + pϵ_simple)
            pupper = min.([0.001/2, 0.001/2, (1-pb)*(1-pb), (1-pb)*pb, pb*(1-pb), pb*pb-0.001] + prob.usize/10, 1.0 - pϵ_simple)
            return plower, pupper
        end
    end
    return nothing
end

function observation(prob::SimpleRIPOMDP, a::Symbol, sp::Symbol, z::Symbol)
    zind = observation_index(prob, z)
    plold, puold = observation(prob, a, sp)
    plower = plold[zind]
    pupper = puold[zind]
    plower, pupper
end

function dynamics(prob::SimpleIPOMDP)
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

function dynamics(prob::SimpleRIPOMDP)
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

function reward(prob::Union{SimpleIPOMDP, SimpleRIPOMDP}, b::Vector{Float64}, a::Symbol)
    rmax = -Inf
    for α in prob.inforeward
        rmax = max(rmax, dot(α, b))
    end
    rmax
end
reward(prob::Union{SimpleIPOMDP,SimpleRIPOMDP}, b::Vector{Float64}, s::Symbol, a::Symbol, sp::Symbol) = reward(prob,b,a)

function rewardalpha(prob::Union{SimpleIPOMDP, SimpleRIPOMDP}, b::Vector{Float64}, a::Symbol)
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

discount(p::Union{SimpleIPOMDP, SimpleRIPOMDP}) = p.discount

# Simulation Functions
function initial_state(prob::Union{SimpleIPOMDP, SimpleRIPOMDP}, rng::AbstractRNG)
    d = initial_state_distribution(prob)
    return rand(rng, d)
end

function generate_sor(prob::SimpleIPOMDP, b, s, a, rng::AbstractRNG)
    sp = rand(rng, transition(prob, s, a))
    o = rand(rng, observation(prob, a, sp))
    r = reward(prob, b, s, a, sp)
    sp, o, r
end

# uniform random generate_sor (original)
function generate_sor(prob::SimpleRIPOMDP, b, s, a, rng::AbstractRNG)
    tdist = SparseCat(states(prob), psample(transition(prob, s, a)...))
    @show tdist
    sp = rand(rng, tdist)
    odist = SparseCat(observations(prob), psample(observation(prob, a, sp)...))
    @show odist
    o = rand(rng, odist)
    r = reward(prob, b, s, a, sp)
    sp, o, r
end
