# # using Revise # in REPL
# Pkg.clone("https://github.com/ajkeith/POMDPs.jl")
# Pkg.clone("https://github.com/ajkeith/POMDPModels.jl")
# Pkg.checkout("POMDPModels","ajk/robust")
push!(LOAD_PATH,"C:\\Users\\op\\Documents\\Julia Projects\\RPOMDPs.jl\\src")
push!(LOAD_PATH,"C:\\Users\\op\\Documents\\Julia Projects\\RPOMDPModels.jl\\src")
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
