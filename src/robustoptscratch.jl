using RPOMDPs
using RPOMDPModels
# using POMDPToolbox

# ambiguity set examples
using JuMP, Clp
mins = [0.05, 0.6, 0.05]
maxs = [0.3, 0.9, 0.4]
m = Model(solver = ClpSolver())
@variable(m, mins[i] <= p[i = 1:3] <= maxs[i])
@constraint(m, sum(p) == 1)
@objective(m, Min, 5 * p[1] + 10 * p[2] - 3 * p[3])
solve(m)
argmin = getvalue(p)


using JuMP, Clp
using BenchmarkTools

# Lower Prob (5 vertices)
# p[1] = pq, p[2] = pc, p[3] = py
mlp = Model(solver = ClpSolver())
@variable(mlp, 0 <= p[i = 1:3] <= 1)
@objective(mlp, Min, 5 * p[1] + 10 * p[2] - 3 * p[3])
@constraint(mlp, sum(p) == 1)
@constraint(mlp, 0.15 <= p[1] <= 0.35)
@constraint(mlp, 0.55 <= p[2] <= 0.7)
@constraint(mlp, 0.05 <= p[3] <= 0.3)
solve(mlp)
argmin_mlp = getvalue(p)
obj_mlp = getobjectivevalue(mlp)


# P-box (4 vertices)
mpb = Model(solver = ClpSolver())
@variable(mpb, 0 <= p[i = 1:3] <= 1)
@objective(mpb, Min, 5 * p[1] + 10 * p[2] - 3 * p[3])
@constraint(mpb, sum(p) == 1)
@constraint(mpb, 0.15 <= p[1] <= 0.35)
@constraint(mpb, 0.05 <= p[3] <= 0.3)
solve(mpb)
argmin_mpb = getvalue(p)
obj_mpb = getobjectivevalue(mpb)

# Polytopic (3 vertices)
m1 = (0.05 - 0.3) / (0.6 - 0.55)
m2 = (0.1 - 0.3) / (0.7 - 0.55)
m3 = (0.05 - 0.1) / (0.6 - 0.7)
y1(c) = m1 * (c - 0.55) + 0.3
y2(c) = m2 * (c - 0.55) + 0.3
y3(c) = m3 * (c - 0.7) + 0.1
mp = Model(solver = ClpSolver())
@variable(mp, 0 <= p[i = 1:3] <= 1)
@objective(mp, Min, 5 * p[1] + 10 * p[2] - 3 * p[3])
@constraint(mp, sum(p) == 1)
@constraint(mp, p[3] >= y1(p[2]))
@constraint(mp, p[3] <= y2(p[2]))
@constraint(mp, p[3] >= y3(p[2]))
solve(mp)
argmin_mp = getvalue(p)
obj_mp = getobjectivevalue(mp)

(argmin_mlp, argmin_mpb, argmin_mp)
(obj_mlp, obj_mpb, obj_mp)

# benchmarks
@benchmark solve($mlp)
@benchmark solve($mpb)
@benchmark solve($mp)

# scratch

temp_array = Array{Float64}(2, 2)
temp_array[1,1] = 10.0


rpomdp = Baby3RPOMDP()
t_array = Array{Float64}(2, 2)
t_array[1, 1] = BoolDistribution(0.0)
t_array[1, 2] = BoolDistribution(1.0)
t_array[2, 1] = BoolDistribution(0.0)
t_array[2, 2] = BoolDistribution(0.1)
t_array[state_index(rpomdp, s), action_index(rpomdp, a)]
t_array = [BoolDistribution(0.0) BoolDistribution(1.0);
    BoolDistribution(0.0) BoolDistribution(0.1)]


prob = Baby3RPOMDP()
transition(prob, false, false)
observation(prob, false, true, "upper")
RPOMDPModels.observation_index(prob, :crying)


# DRO toy problem
# x ∈ R^3
# E(x) = [1,2,3]
# P(x ∈ [(0,5),(0,5),(0,5)]) ∈ [1,1]
# P(x ∈ [(0,3),(0,3),(0,3)]) ∈ [0.7, 0.9]

# Convert from v-rep to h-rep for JUMP model (polytopic ambiguity set)
using Polyhedra, CDDLib
using JuMP, Clp
polyv = vrep([0.1 0.1 0.8; 0.1 0.3 0.6; 0.3 0.3 0.4])
poly = polyhedron(polyv, CDDLibrary())
polyh = hrep(poly)
polyhv = polyhedron(polyh, CDDLibrary())
polyhvv = vrep(polyhv)
ph = MixedMatHRep(polyh)
pv = MixedMatVRep(polyhvv)

mpoly = Model(solver = ClpSolver())
@variable(mpoly, 0 <= p[i = 1:3] <= 1)
@objective(mpoly, Min, 5 * p[1] + 10 * p[2] - 3 * p[3])
@constraint(mpoly, sum(p) == 1)
@constraint(mpoly, ph.A * p .<= ph.b)
solve(mpoly)
argmin_mpoly = getvalue(p)
obj_mpoly = getobjectivevalue(mpoly)


# compare RPOMDP to POMDP
using POMDPs, POMDPModels, POMDPToolbox
using IncrementalPruning, FIB, QMDP
using Plots; gr()

prob = TigerPOMDP()
s1 = PruneSolver(max_iterations = 20)
s2 = FIBSolver(max_iterations = 10_000)
s3 = QMDPSolver(max_iterations = 10_000)
pol1 = solve(s1, prob)
pol2 = solve(s2, prob)
pol3 = solve(s3, prob)
bu1 = updater(pol1)
bu2 = updater(pol2)
bu3 = updater(pol3)
h1 = simulate(HistoryRecorder(max_steps=100), prob, pol1, bu1)
h2 = simulate(HistoryRecorder(max_steps=100), prob, pol2, bu2)
h3 = simulate(HistoryRecorder(max_steps=100), prob, pol3, bu3)
v1 = discounted_reward(h1)
v2 = discounted_reward(h2)
v3 = discounted_reward(h3)
POMDPToolbox.test_solver(s1, prob) # 17.7
POMDPToolbox.test_solver(s2, prob) # 17.7
POMDPToolbox.test_solver(s3, prob) # 17.7
value(pol1, [0.5,0.5])
value(pol2, [0.5,0.5])
value(pol3, [0.5,0.5])
N = 1000
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1000), prob, pol1, bu1)) for i = 1:N)
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1000), prob, pol2, bu2)) for i = 1:N)
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1000), prob, pol3, bu3)) for i = 1:N)


# Incremental Pruning
# too many to plot
Plots.plot([0,1], pol1.alphas, xticks = 0:0.05:1, lab = pol1.action_map, legend = :bottomright)

# FIB
# action 1 at <0.05, action 2 at >0.95
Plots.plot([0,1], pol2.alphas, xticks = 0:0.05:1, lab = pol2.action_map, legend = :bottomright)

# AEMS
# action 1 at <0.05, action 2 at >0.95
Plots.plot([0,1], pol3.alphas, xticks = 0:0.05:1, lab = pol3.action_map, legend = :bottomright)

# different y-values for alpha-vectors
# sim value agrees for tigerpomdp discount = 0.95 is about 19.5



###############################
#
# Correctness testing
#
# TODO: Add to actual test file (i.e. take constant values from POMDP results and use that to test against in the runtests of RobustValueIteration)
################################
# compare RPOMDP to POMDP
using RPOMDPs, RPOMDPModels, RPOMDPToolbox
using RobustValueIteration
using Plots; gr()

prob = TigerPOMDP(0.95)
prob2 = TigerRPOMDP(0.95, 0.01)
sr = RPBVISolver(max_iterations = 10_000)
sr2 = RPBVISolver(max_iterations = 1000)
polr = RobustValueIteration.solve(sr, prob)
polr2 = RobustValueIteration.solve(sr2, prob2)
bur = updater(polr)
bur2 = updater(polr2)
hr = simulate(HistoryRecorder(max_steps=100), prob, polr, bur)
hr2 = simulate(HistoryRecorder(max_steps=100), prob2, polr2, bur2)
vr = discounted_reward(hr)
vr2 = discounted_reward(hr2)
value(polr, [0.5,0.5])
value(polr2, [0.5,0.5])
N = 1_000
N2 = 1
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1_000),
                                                prob, polr, bur)) for i = 1:N)
mean(discounted_reward(simulate(HistoryRecorder(max_steps=100),
                                                prob2, polr2, bur2)) for i = 1:N2)

# Robust point-based value iteration
# POMDP
Plots.plot([0,1], polr.alphas, xticks = 0:0.05:1,
            lab = polr.action_map, legend = :bottomright)

# Robust point-based value iteration
# RPOMDP
Plots.plot([0,1], polr2.alphas, xticks = 0:0.05:1,
            lab = polr2.action_map, legend = :bottomright)


prob = BabyPOMDP()
sr = RPBVISolver(max_iterations = 10_000)
polr = RobustValueIteration.solve(sr, prob)
bur = updater(polr)
hr = simulate(HistoryRecorder(max_steps=100), prob, polr, bur)
vr = discounted_reward(hr)
value(polr, [0.0, 1.0])
N = 1_000
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1_000),
                                                prob, polr, bur)) for i = 1:N)

prob2 = BabyRPOMDP(0.2)
sr2 = RPBVISolver(max_iterations = 1000)
polr2 = RobustValueIteration.solve(sr2, prob2)
bur2 = updater(polr2)
hr2 = simulate(HistoryRecorder(max_steps=100), prob2, polr2, bur2)
vr2 = discounted_reward(hr2)
value(polr2, [0.0, 1.0])
N2 = 1
# mean(discounted_reward(simulate(HistoryRecorder(max_steps=100),
                                            # prob2, polr2, bur2)) for i = 1:N2)

# Robust point-based value iteration
# POMDP
Plots.plot([0,1], polr.alphas, xticks = 0:0.1:1,
        lab = polr.action_map, legend = :bottomright)

# Robust point-based value iteration
# RPOMDP
Plots.plot([0,1], polr2.alphas, xticks = 0:0.1:1,
        lab = polr2.action_map, legend = :bottomright)
