using JuMP, Clp

rpomdp = Baby3RPOMDP()
dim = n_observations(rpomdp)
uncset = observation(rpomdp,true,true)
cdfl = cumsum(pdf.(uncset.lower))
cdfu = cumsum(pdf.(uncset.upper))

coeff = [5,2,3]
m = Model(solver = ClpSolver())
@variable(m, 0 <= p[i = 1:dim] <= 1)
@objective(m, Min, dot(coeff,p))
@constraint(m, sum(p) == 1)
@constraint(m, cdfl[1] <= p[1] <= cdfu[1])
for i = 2:dim
    pl = cdfl[i] - cdfu[i-1]
    pu = cdfu[i] - cdfl[i-1]
    @constraint(m, pl <= p[i] <= pu)
end
JuMP.solve(m)
argmin = getvalue(p)
obj = getobjectivevalue(m)


rpomdp = Baby3RPOMDP()
dist_t = transition(rpomdp,true,true)
dist = observation(rpomdp,true,true)
dim = n_observations(rpomdp)
cdfl = cumsum(pdf.(uncset.lower))
cdfu = cumsum(pdf.(uncset.upper))
pt = [0.5,0.6]
α = [1,2]
m = Model(solver = ClpSolver())
@variable(m, 0 <= p[i = 1:dim] <= 1)
@objective(m, Min, dot(α, p .* pt))
@constraint(m, cdfl[zind] <= p <= cdfu[zind])
JuMP.solve(m)
argmin = getvalue(p)
obj = getobjectivevalue(m)


dist = observation(prob,a,sp)
dim = n_observations(rpomdp)
function robustdist(uncset::PBox, dim::Int, α::Vector{Float64})
    cdfl = cumsum(pdf.(uncset.lower))
    cdfu = cumsum(pdf.(uncset.upper))
    m = Model(solver = ClpSolver())
    @variable(m, 0 <= p[i = 1:dim] <= 1)
    @objective(m, Min, dot(α, p .* pt))
    @constraint(m, sum(p) == 1)
    @constraint(m, cdfl[1] <= p[1] <= cdfu[1])
    for i = 2:dim
        pl = cdfl[i] - cdfu[i-1]
        pu = cdfu[i] - cdfl[i-1]
        @constraint(m, pl <= p[i] <= pu)
    end
    JuMP.solve(m)
    argmin = getvalue(p)
    obj = getobjectivevalue(m)
end
