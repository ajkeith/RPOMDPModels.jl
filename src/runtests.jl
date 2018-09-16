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
end

# @testset "RobustBelief" begin
#     # constructor
#     rp = Baby3RPOMDP()
#     uniform_belief(rp)
#     @test b == vec
#
#     # belief Updater
#     @test update_belief(prob, true, true, b) == new_vec
# end
