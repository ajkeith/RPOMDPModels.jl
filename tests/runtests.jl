using Base.Test
using RPOMDPs, RPOMDPModels
using RobustValueIteration
using SimpleProbabilitySets, RPOMDPToolbox, Distances
const RPBVI = RobustValueIteration

@testset "Robust and Info-Reward Baby POMDP" begin
    # constructor
    p = BabyPOMDP()
    ip = BabyIPOMDP()
    rp = BabyRPOMDP()
    rip = BabyRIPOMDP()
    @test (rip.r_feed, rip.r_hungry, rip.discount) == (-5., -10., 0.9)

    # s,a,z indexes
    # s = (true (hungry), false (full))
    # a = (true (feed), false (no action))
    # z = (:quiet, :crying, :yelling)
    @test state_index(p, :hungry) == 1
    @test action_index(ip, :nothing) == 2
    @test observation_index(rp, :yelling) == 3

    # transition function
    # P(hungry | full, no action) = 0.1
    @test pdf(transition(p, :full, :nothing), :hungry) == 0.1
    @test pdf(transition(ip, :full, :nothing), :hungry) == 0.1
    @test transition(rp, :full, :nothing, :hungry)[1] ≈ 0.075 atol = 1e-6
    @test transition(rip, :full, :nothing, :hungry)[1] ≈ 0.075 atol = 1e-6

    # observation function
    # P_lowercdf(yelling | hungry, feed) = 0.3
    # P_uppercdf(yelling | hungry, feed) = 0.05
    # note: p-box (not upper and lower probabilities)
    @test pdf(observation(p, :feed, :hungry), :quiet) == 0.3
    @test pdf(observation(ip, :feed, :hungry), :yelling) == 0.1
    @test observation(rp, :feed, :hungry)[1][1] ≈ 0.275 atol = 1e-6
    @test observation(rip, :feed, :hungry)[2][1] ≈ 0.325 atol = 1e-6

    # dynamics
    @test dynamics(p)[1,1,1,1] == 0.0
    @test dynamics(rip)[1][2,2,2,2] ≈ 0.065625 atol = 1e-6

    # reward
    # R(hungry, feed) = -15
    @test reward(p, :hungry, :feed) == -15
    @test reward(rp, :hungry, :feed) == -15
    @test reward(ip, [0.2, 0.8], :feed) ≈ 0.6 atol = 1e-6
    @test reward(rip, [0.2, 0.8], :feed) ≈ 0.6 atol = 1e-6

    # generate samples
    p = BabyPOMDP()
    rp = BabyRPOMDP()
    b = [0.8, 0.2]
    s = :hungry
    a = :nothing
    rng = MersenneTwister(20348)
    @test generate_sor(p, b, s, a, rng)[2] == :crying
    @test generate_sor(rp, b, s, a, rng)[2] == :crying
end

@testset "Cyber Assessment" begin
    p = CyberPOMDP()
    ip = CyberIPOMDP()
    rp = CyberRPOMDP()
    rip = CyberRIPOMDP()

    @test state_index(p, [3,3,3]) == 27
    @test action_index(ip, [2,3]) == 4
    @test observation_index(rp, [1,3]) == 3
    @test transition(ip, [1,1,1], [3,2]).probs |> sum ≈ 1.0
    @test transition(ip, [2,1,3], [3,2]).probs |> sum ≈ 1.0
    @test all(transition(ip, [1,1,2], [2,1]).probs .== transition(ip, [1,1,2], [1,2]).probs)
    @test all(transition(ip, [1,1,2], [2,1]).probs .== transition(ip, [1,1,2], [1,3]).probs)
    @test transition(rp, [2,2,2], [1,3], [3,2,3])[2] ≈ 0.053816
    @test RPOMDPModels.o([1,2], [2,2,2], [1,1], [0.8, 0.9]) ≈ 0.005
    @test observation(ip, [1,2], [1,1,1]).probs |> sum == 1.0
    @test observation(rp, [1,2], [2,1,3])[1] |> sum < 1.0
    @test dynamics(ip)[:,:,20,4] |> sum == 1
    @test all(dynamics(rip)[2] .>= dynamics(rip)[1])
    @test all(dynamics(rip)[2] .>= dynamics(ip))
    @test all(dynamics(rip)[1] .<= dynamics(ip))
    @test reward(rp, [1,3,2], [1,3]) ≈ 0.49
    @test reward(rip, fill(1/27,27), [3,3]) ≈ 0.0 atol = 1e-9
    @test reward(rip, vcat(1.0, zeros(26)), [1,3]) == 1.0

    b = vcat([0.8, 0.2], zeros(25))
    s = [1,2,3]
    a = [2,3]
    rng = MersenneTwister(20348)
    @test generate_sor(p, b, s, a, rng)[2] == [2,3]
    @test generate_sor(rp, b, s, a, rng)[2] == [2,2]

    nb = 5
    srand(47329342)
    bs = [psample(zeros(27), ones(27)) for i = 1:nb]

    maxiter = 5
    solver = RPBVISolver(beliefpoints = bs, max_iterations = maxiter)
    Vold = fill(RPBVI.AlphaVec(zeros(n_states(rip)), ordered_actions(rip)[1]),
        length(solver.beliefpoints))
    Volda = [Vold[i].alpha for i = 1:length(Vold)]
    e11 = zeros(27); e11[1] = 1.0;
    b = e11
    a = [1,2]
    ai = action_index(ip, a)
    umin, pmin = RPBVI.minutil(rip, b, a, Volda)
    @test pmin[:,:,1] |> sum ≈ 1.0 atol = 1e-9
    @test pmin[:,:,2] |> sum ≈ 1.0 atol = 1e-9
    @test pmin[:,:,13] |> sum ≈ 1.0 atol = 1e-9
    @test pmin[:,:,27] |> sum ≈ 1.0 atol = 1e-9
end

@testset "Cyber Assessment (Off Nominal)" begin
    ip = CyberTestIPOMDP()
    @test state_index(ip, [3,3,3]) == 27
    @test action_index(ip, [2,3]) == 4
    @test observation_index(ip, [1,3]) == 3
    @test transition(ip, [1,1,1], [3,2]).probs |> sum ≈ 1.0
    @test transition(ip, [2,1,3], [3,2]).probs |> sum ≈ 1.0
    @test all(transition(ip, [1,1,2], [2,1]).probs .== transition(ip, [1,1,2], [1,2]).probs)
    @test all(transition(ip, [1,1,2], [2,1]).probs .== transition(ip, [1,1,2], [1,3]).probs)
    @test transition(ip, [2,2,2], [1,3], [3,2,3]) ≈ 0.0495
    @test observation(ip, [1,2], [1,1,1]).probs |> sum == 1.0
    @test dynamics(ip)[:,:,20,4] |> sum == 1
    @test reward(ip, fill(1/27,27), [3,3]) ≈ 0.0 atol = 1e-9
    @test reward(ip, vcat(1.0, zeros(26)), [1,3]) == 1.0

    b = vcat([0.8, 0.2], zeros(25))
    s = [1,2,3]
    a = [2,3]
    rng = MersenneTwister(20348)
    @test generate_sor(ip, b, s, a, rng)[2] == [2,3]

    nb = 5
    srand(47329342)
    bs = [psample(zeros(27), ones(27)) for i = 1:nb]
    maxiter = 5
    solver = RPBVISolver(beliefpoints = bs, max_iterations = maxiter)
    polinit = RPBVI.create_policy(solver, ip)
    polip = RPBVI.solve(solver, ip, verbose = false)
    @test value(polip, b) > value(polinit, b)
end

@testset "Rock Diagnosis" begin
    ip = RockIPOMDP()
    rip = RockRIPOMDP()
    @test states(ip) == [10, 11, 20, 21]
    @test actions(ip) == [:left, :right, :check]
    @test observations(ip) == [:good, :bad, :none]
    @test states(rip) == [10, 11, 20, 21]
    @test actions(rip) == [:left, :right, :check]
    @test observations(rip) == [:good, :bad, :none]
    @test state_index.(ip, states(ip)) == [1,2,3,4]
    @test action_index.(ip, actions(ip)) == [1,2,3]
    @test observation_index.(ip, observations(ip)) == [1,2,3]

    s = 11
    a = :right
    z = :none
    sp = 21
    @test pdf(transition(ip, s, a), sp) == 1.0
    @test transition(ip, s, a, sp) == 1.0
    @test pdf(observation(ip, a, sp), z) == 1.0
    @test observation(ip, a, sp, z) == 1.0
    @test transition(rip, s, a)[1][4] == 0.999999
    @test transition(rip, s, a, sp) == (0.999999, 1.0)
    @test all(dynamics(rip)[2] .> dynamics(rip)[1])
    @test all(dynamics(rip)[2] .>= dynamics(ip))
    @test all(dynamics(ip) .>= dynamics(rip)[1])
    @test reward(ip, [0.5, 0.5, 0.0, 0.0], :left) ≈ 0.3333333 atol = 1e-5
    @test reward(rip, [0.5, 0.5, 0.0, 0.0], :left) ≈ 0.3333333 atol = 1e-5
    @test reward(ip, [1.0, 0.0, 0.0, 0.0], a) == 1.0
    @test reward(ip, [0.0, 1.0, 0.0, 0.0], a) == 1.0
    @test reward(ip, [0.0, 0.0, 1.0, 0.0], a) == 1.0
    @test reward(ip, [0.0, 0.0, 0.0, 1.0], a) == 1.0

    b = [0.7, 0.3, 0.0, 0.0]
    s = 11
    a = :check
    rng = MersenneTwister(0)
    @test generate_sor(ip, b, s, a, rng) == (11, :bad, 0.6)
    @test generate_sor(rip, b, s, a, rng) == (11, :good, 0.6)

    nb = 10
    srand(47329342)
    bs1 = [vcat(psample(zeros(2), ones(2)),zeros(2)) for i = 1:nb]
    bs2 = [vcat(zeros(2), psample(zeros(2), ones(2))) for i = 1:nb]
    bs = vcat(bs1, bs2)
    brand = rand() > 0.5 ? vcat(zeros(2), psample(zeros(2), ones(2))) : vcat(psample(zeros(2), ones(2)), zeros(2))
    @test minimum(norm(brand - bs[i]) for i=1:(2nb)) < 0.3

    maxiter = 100
    uncsize = 0.2
    solver = RPBVISolver(beliefpoints = bs, max_iterations = maxiter)
    Vold = fill(RPBVI.AlphaVec(zeros(n_states(rip)), ordered_actions(rip)[1]),
        length(solver.beliefpoints))
    Volda = [Vold[i].alpha for i = 1:length(Vold)]
    b = [0.620804, 0.379196, 0.0, 0.0]
    a = :left
    ai = action_index(ip, a)
    umin, pmin = RPBVI.minutil(rip, b, a, Volda)
    @test pmin[:,:,1] |> sum == 1.0
    @test pmin[:,:,2] |> sum == 1.0
    @test pmin[:,:,3] |> sum == 1.0
    @test pmin[:,:,4] |> sum == 1.0

    polinit = RPBVI.create_policy(solver, rip)
    polip = RPBVI.solve(solver, ip)
    polrip = RPBVI.solve(solver, rip)
    @test value(polip, [0.5, 0.5, 0.0, 0.0]) ≈ 18.5815 atol = 1e-3
    @test value(polip, [1.0, 0.0, 0.0, 0.0]) ≈ 19.8816 atol = 1e-3
    @test value(polinit, [0.5, 0.5, 0.0, 0.0]) ≈ 0.0 atol = 1e-3
    @test value(polinit, [1.0, 0.0, 0.0, 0.0]) ≈ 0.0 atol = 1e-3
    @show value(polrip, [0.5, 0.5, 0.0, 0.0])
    @show value(polrip, [1.0, 0.0, 0.0, 0.0])
    @test value(polrip, [0.5, 0.5, 0.0, 0.0]) ≈ 18.5051 atol = 1e-3
    @test value(polrip, [1.0, 0.0, 0.0, 0.0]) ≈ 19.8657 atol = 1e-3
end





# ######################
# #
# # Developmental Testing
# #
# #####################
#
# using Base.Test
# using RPOMDPs, RPOMDPModels
# using RobustValueIteration
# using SimpleProbabilitySets, RPOMDPToolbox, Distances
# const RPBVI = RobustValueIteration
#
#
# ip = CyberTestIPOMDP()
#
#
# @test state_index(ip, [3,3,3]) == 27
# @test action_index(ip, [2,3]) == 4
# @test observation_index(ip, [1,3]) == 3
# @test transition(ip, [1,1,1], [3,2]).probs |> sum ≈ 1.0
# @test transition(ip, [2,1,3], [3,2]).probs |> sum ≈ 1.0
# @test all(transition(ip, [1,1,2], [2,1]).probs .== transition(ip, [1,1,2], [1,2]).probs)
# @test all(transition(ip, [1,1,2], [2,1]).probs .== transition(ip, [1,1,2], [1,3]).probs)
# @test transition(ip, [2,2,2], [1,3], [3,2,3]) ≈ 0.0495
# @test observation(ip, [1,2], [1,1,1]).probs |> sum == 1.0
# @test dynamics(ip)[:,:,20,4] |> sum == 1
# @test reward(ip, fill(1/27,27), [3,3]) ≈ 0.0 atol = 1e-9
# @test reward(ip, vcat(1.0, zeros(26)), [1,3]) == 1.0
#
# b = vcat([0.8, 0.2], zeros(25))
# s = [1,2,3]
# a = [2,3]
# rng = MersenneTwister(20348)
# @test generate_sor(ip, b, s, a, rng)[2] == [2,3]
#
# nb = 5
# srand(47329342)
# bs = [psample(zeros(27), ones(27)) for i = 1:nb]
# maxiter = 5
# solver = RPBVISolver(beliefpoints = bs, max_iterations = maxiter)
# polinit = RPBVI.create_policy(solver, ip)
# polip = RPBVI.solve(solver, ip, verbose = true)
# @test value(polip, b) > value(polinit, b)
