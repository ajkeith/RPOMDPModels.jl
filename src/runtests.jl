
using Base.Test
using RPOMDPs, RPOMDPModels, POMDPToolbox

@testset "Robust POMDP" begin
    @testset "Baby3RPOMDP" begin
        # constructor
        prob = Baby3RPOMDP()
        @test (prob.r_feed, prob.r_hungry, prob.discount) == (-5., -10., 0.9)

        # s,a,z indexes
        # s = (true (hungry), false (full))
        # a = (true (feed), false (no action))
        # z = (:quiet, :crying, :yelling)
        @test state_index(prob, true) == 1
        @test action_index(prob, false) == 2
        @test observation_index(prob, :yelling) == 3

        # transition function
        # P(hungry | full, no action) = 0.1
        @test pdf(transition(prob, false, false), true) == 0.1

        # observation function
        # P_lowercdf(yelling | hungry, feed) = 0.3
        # P_uppercdf(yelling | hungry, feed) = 0.05
        # note: p-box (not upper and lower probabilities)
        @test pdf(observation(prob, true, true, "lower"), 3) == 0.3
        @test pdf(observation(prob, true, true, "upper"), 3) == 0.05

        # reward
        # R(hungry, feed) = -15
        @test reward(prob, true, true) == -15

        # generate observation
        @test generate_o(prob, true, MersenneTwister(88), "upper") == :quiet
    end

    @testset "RobustBelief" begin
        # constructor
        b = new_belief(prob, vec)
        @test b == vec

        # belief Updater
        @test update_belief(prob, true, true, b) == new_vec
    end
end
