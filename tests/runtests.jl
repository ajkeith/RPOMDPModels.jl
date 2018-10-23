using Base.Test
using RPOMDPs, RPOMDPModels

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

    @test state_index(p, [2,3,2]) == 17
    @test action_index(ip, [2,3]) == 6
    @test observation_index(rp, [1,3]) == 3
    @test transition(ip, [1,1,1], [3,2]).probs |> sum ≈ 1.0
    @test transition(ip, [2,1,3], [3,2]).probs |> sum ≈ 1.0
    @test all(transition(ip, [1,1,1], [3,2]).probs .== transition(ip, [1,1,1], [1,1]).probs)
    @test transition(rp, [2,2,2], [1,3], [3,2,3])[2] ≈ 0.079625
    @test RPOMDPModels.o([2,3], [1,3,1], [3,1], 0.1) == 0.81
    @test observation(ip, [1,1], [1,1,1]).probs |> sum == 1
    @test observation(rp, [1,1], [2,1,3])[1][1] ≈ 0.0025
    @test dynamics(ip)[:,:,1,1] |> sum == 1
    @test dynamics(ip)[10,9,1,1] ≈ 0.00147
    @test dynamics(rip)[1][1,1,1,1] ≈ 0.21114
    @test reward(rp, [1,3,2], [1,3]) ≈ 0.89
    @test reward(rip, fill(1/27,27), [3,3]) ≈ 0.0370370370370
    @test reward(rip, vcat(1.0, zeros(26)), [1,3]) == 1.0

    b = vcat([0.8, 0.2], zeros(25))
    s = [1,2,3]
    a = [2,3]
    rng = MersenneTwister(20348)
    @test generate_sor(p, b, s, a, rng)[2] == [2,3]
    @test generate_sor(rp, b, s, a, rng)[2] == [2,3]
end
