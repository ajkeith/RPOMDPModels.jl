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


using RPOMDPs, RPOMDPModels
using RobustValueIteration
using RPOMDPToolbox, DataFrames

p = CyberPOMDP()
ip = CyberIPOMDP()
rp = CyberRPOMDP()
rip = CyberRIPOMDP()

state_index(p, [2,3,2]) == 17
action_index(ip, [2,3]) == 6
observation_index(rp, [1,3]) == 3
transition(ip, [1,1,1], [3,2]).probs |> sum
transition(ip, [2,1,3], [3,2]).probs |> sum
all(transition(ip, [1,1,1], [3,2]).probs .== transition(ip, [1,1,1], [1,1]).probs)
transition(rp, [2,2,2], [1,3], [3,2,3])[2] ≈ 0.079625
RPOMDPModels.o([2,3], [1,3,1], [3,1], err) == 0.81
observation(ip, [1,1], [1,1,1]).probs |> sum == 1
observation(rp, [1,1], [2,1,3])[1][1] ≈ 0.0025
dynamics(ip)[:,:,1,1] |> sum == 1
dynamics(ip)[10,9,1,1] ≈ 0.00147
dynamics(rip)[1][1,1,1,1] ≈ 0.21114
reward(rp, [1,3,2], [1,3]) ≈ 0.89
reward(rip, fill(1/27,27), [3,3]) ≈ 0.0370370370370
reward(rip, vcat(1.0, zeros(26)), [1,3]) == 1.0

b = vcat([0.8, 0.2], zeros(25))
s = [1,2,3]
a = [2,3]
rng = MersenneTwister(20348)
generate_sor(p, b, s, a, rng)[2] == [2,3]
generate_sor(rp, b, s, a, rng)[2] == [2,3]

# select belief points
s0 = fill(1/27, 27);
s1 = zeros(27); s1[14] = 1;
s2 = zeros(27); s2[1] = 1;
s32 = zeros(27); s32[1] = 0.5; s32[2] = 0.5;
s42 = zeros(27); s42[1] = 0.5; s42[4] = 0.5;
s52 = zeros(27); s52[1] = 0.5; s52[10] = 0.5;
s33 = zeros(27); s33[1] = 0.5; s33[3] = 0.5;
s43 = zeros(27); s43[1] = 0.5; s43[7] = 0.5;
s53 = zeros(27); s53[1] = 0.5; s53[19] = 0.5;
s6 = zeros(27); s6[1] = 0.5; s6[6] = 0.5;
s7 = zeros(27); s7[1] = 0.5; s7[22] = 0.5;
ss = [s0, s1, s2, s32, s42, s52, s33, s43, s53, s6, s7]
nS = length(ss)
bs = Vector{Vector{Float64}}(nS)
for i = 1:length(ss)
    bs[i] = ss[i]
end
push!(bs ,vcat(fill(0.0, 27 - 1), 1.0))
push!(bs, fill(1/27, 27))

# intialize solver
solver = RPBVISolver(beliefpoints = bs, max_iterations = 15)

# solve
solipt = RobustValueIteration.solve(solver, ip)

# check results
e1 = zeros(27); e1[1] = 1
policyvalue(solipt, e1)

buip = updater(solipt)
dbsip = [DiscreteBelief(ip, states(ip), s) for s in ss]
update(buip, dbsip[1], [3,3], [1,1]).b

# ipomdp and ripomdp actions for some interesting states
actionind = ["unif", "2,2,2", "1,1,1", "1,1,1 - 1,1,2", "1,1,1 - 1,2,1", "1,1,1 - 2,1,1",
        "1,1,1 - 1,1,3" ,"1,1,1 - 1,3,1", "1,1,1 - 3,1,1",
        "1,1,1 - 1,2,3", "1,1,1 - 3,2,1" ]
dbsip = [DiscreteBelief(ip, states(ip), s) for s in ss]
asip = [action(solipt, db) for db in dbsip]
actiondata = DataFrame(Belief = actionind, StdAction = asip)
@show actiondata
