# Write problem dynamics to .csv files
using DataFrames, CSV

function writedynamics(path::String, fn::String, dyn::Array{Float64,4})
    (_, nz, ns, na) = size(dyn)
    for si = 1:ns, ai = 1:na
        fnslice = string(fn,"_",si,"_",ai,".csv")
        slice = DataFrame(dyn[:, :, si, ai])
        CSV.write(joinpath(path, fnslice), slice, header = false)
    end
end

function writedynamics(path::String, fn::String, dyn::Tuple{Array{Float64,4},Array{Float64,4}})
    dyn_lower, dyn_upper = dyn
    (_, nz, ns, na) = size(dyn_lower)
    for si = 1:ns, ai = 1:na
        fnslice = string(fn,"_lower_",si,"_",ai,".csv")
        slice = DataFrame(dyn_lower[:, :, si, ai])
        CSV.write(joinpath(path, fnslice), slice, header = false)
        fnslice = string(fn,"_upper_",si,"_",ai,".csv")
        slice = DataFrame(dyn_upper[:, :, si, ai])
        CSV.write(joinpath(path, fnslice), slice, header = false)
    end
end

function writetransition(path::String, fn::String, problem::Union{POMDP,IPOMDP})
    ns = n_states(problem)
    na = n_actions(problem)
    tarray = Array{Float64,3}(ns, na, ns)
    for (si,s) in enumerate(states(problem)), (ai,a) in enumerate(actions(problem)), (spi,sp) in enumerate(states(problem))
        tarray[si, ai, spi] = transition(problem, s, a, sp)
    end
    for ai in 1:na
        slice = DataFrame(tarray[:,ai,:])
        fnslice = string(fn,"_",ai,".csv")
        CSV.write(joinpath(path, fnslice), slice, header = false)
    end
end

function writeobservation(path::String, fn::String, problem::Union{POMDP,IPOMDP})
    ns = n_states(problem)
    na = n_actions(problem)
    nz = n_observations(problem)
    zarray = Array{Float64,3}(na, ns, nz)
    for (ai,a) in enumerate(actions(problem)), (spi,sp) in enumerate(states(problem)), (zi,z) in enumerate(observations(problem))
        zarray[ai, spi, zi] = observation(problem, a, sp, z)
    end
    for ai in 1:na
        slice = DataFrame(zarray[ai,:,:])
        fnslice = string(fn,"_",ai,".csv")
        CSV.write(joinpath(path, fnslice), slice, header = false)
    end
end

function writetransition(path::String, fn::String, problem::Union{RPOMDP,RIPOMDP})
    ns = n_states(problem)
    na = n_actions(problem)
    tarray_lower = Array{Float64,3}(ns, na, ns)
    tarray_upper = Array{Float64,3}(ns, na, ns)
    for (si,s) in enumerate(states(problem)), (ai,a) in enumerate(actions(problem)), (spi,sp) in enumerate(states(problem))
        t_lower, t_upper = transition(problem, s, a, sp)
        tarray_lower[si, ai, spi] = t_lower
        tarray_upper[si, ai, spi] = t_upper
    end
    for ai in 1:na
        slice = DataFrame(tarray_lower[ai,:,:])
        fnslice = string(fn,"_lower_",ai,".csv")
        CSV.write(joinpath(path, fnslice), slice, header = false)
        slice = DataFrame(tarray_upper[ai,:,:])
        fnslice = string(fn,"_upper_",ai,".csv")
        CSV.write(joinpath(path, fnslice), slice, header = false)
    end
end

function writeobservation(path::String, fn::String, problem::Union{RPOMDP,RIPOMDP})
    ns = n_states(problem)
    na = n_actions(problem)
    nz = n_observations(problem)
    zarray_lower = Array{Float64,3}(na, ns, nz)
    zarray_upper = Array{Float64,3}(na, ns, nz)
    for (ai,a) in enumerate(actions(problem)), (spi,sp) in enumerate(states(problem)), (zi,z) in enumerate(observations(problem))
        z_lower, z_upper = observation(problem, a, sp, z)
        zarray_lower[ai, spi, zi] = z_lower
        zarray_upper[ai, spi, zi] = z_upper
    end
    for ai in 1:na
        slice = DataFrame(zarray_lower[ai,:,:])
        fnslice = string(fn,"_lower_",ai,".csv")
        CSV.write(joinpath(path, fnslice), slice, header = false)
        slice = DataFrame(zarray_upper[ai,:,:])
        fnslice = string(fn,"_upper_",ai,".csv")
        CSV.write(joinpath(path, fnslice), slice, header = false)
    end
end

disc = 0.9
ip = CyberIPOMDP(disc)
rip = CyberRIPOMDP(disc)
tip = CyberTestIPOMDP(disc)

dynamics_ip = dynamics(ip)
dynamics_rip = dynamics(rip)
dynamics_tip = dynamics(tip)


path = joinpath(homedir(),".julia\\v0.6\\RPOMDPModels\\data\\cyber_assessment")

writedynamics(path, "dynamics_nominal", dynamics_ip)
writedynamics(path, "dynamics_robust", dynamics_rip)
writedynamics(path, "dynamics_offnominal", dynamics_tip)

writetransition(path, "transition_nominal", ip)
writetransition(path, "transition_robust", rip)
writetransition(path, "transition_offnominal", tip)

writeobservation(path, "observation_nominal", ip)
writeobservation(path, "observation_robust", rip)
writeobservation(path, "observation_offnominal", tip)
