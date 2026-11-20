include(joinpath("/home/claude/sim","src","ReplySims.jl")); using .ReplySims, StableRNGs, Statistics
fs=100; dt=1/fs; tv=collect(0:dt:2.0-dt)
pt(; kw...) = rng -> begin r,c,cpp = kelly_port_trial(rng, tv, dt; d=6.0, kw...); (r,cpp) end
res(tr)= begin v=map(1:3) do s; e,ev=simulate_continuous(StableRNG(s),tr;N=500,fs); m=analyse(e,ev,fs); (m["deconv"]/m["epoch"], m["residual"]/m["epoch"]) end; round.((mean(first.(v)),mean(last.(v)));digits=2) end
for (n,s,m) in (("0/0",0.0,0.0),("realistic",0.015,0.025),("0.1/0.1",0.1,0.1))
  println(n," port ",res(pt(Terz=sd_to_width(s),st=sd_to_width(m))),"  port Sz=0 ",res(pt(Terz=sd_to_width(s),st=sd_to_width(m),Sz=0.0)),
     "  port Sz=0,Z=0 ",res(pt(Terz=sd_to_width(s),st=sd_to_width(m),Sz=0.0,Z=0.0)))
end
# seed sensitivity of deconv at 0/0 for KellyModel defaults
v=[ (e=simulate_continuous(StableRNG(s), rng->begin r,t=ReplySims.UnfoldSim.SSM_Simulate(rng,kelly_model(sens_sd=0.0,motor_sd=0.0,urgency=true),fs,200);(r/fs,t) end; N=500,fs); m=analyse(e[1],e[2],fs); round(m["deconv"]/m["epoch"],digits=2)) for s in 1:10]
println("KellyModel defaults σ0/0 deconv res_norm, 10 seeds: ", v)
v=[ (e=simulate_continuous(StableRNG(s), rng->begin r,t=ReplySims.UnfoldSim.SSM_Simulate(rng,kelly_model(sens_sd=0.015,motor_sd=0.025,urgency=true),fs,200);(r/fs,t) end; N=500,fs); m=analyse(e[1],e[2],fs); (round(m["deconv"]/m["epoch"],digits=2),round(m["residual"]/m["epoch"],digits=2))) for s in 1:10]
println("KellyModel defaults realistic (deconv,residual), 10 seeds: ", v)
