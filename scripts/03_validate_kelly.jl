# Compare UnfoldSim's KellyModel with the Kelly & O'Connell implementation (O'Connell et al. 2025
# MATLAB code / Frömer et al. 2024 Julia port): behaviour (RT quantiles, accuracy) and mean CPP
# traces under (i) each implementation's own defaults and (ii) the settings used in the reply.
include(joinpath(@__DIR__, "..", "src", "ReplySims.jl")); using .ReplySims
using Distributions, UnfoldSim, StableRNGs, Statistics, DataFrames, CSV, CairoMakie
fs = 500; dt = 1 / fs; tv = collect(0:dt:1.6 - dt); L = length(tv); N = 2000
q(x) = round.(quantile(filter(!isnan, x), [0.1, 0.3, 0.5, 0.7, 0.9]); digits = 3)
rows = []; traces = Dict{String,Vector{Float64}}()
function port(label; kw...)
    rng = StableRNG(1); rt = Float64[]; ch = Int[]; M = zeros(L)
    for i in 1:N
        r, c, cpp = kelly_port_trial(rng, tv, dt; kw...); push!(rt, r); push!(ch, c); M .+= cpp
    end
    traces[label] = M ./ N
    push!(rows, (; model = label, accuracy = mean(ch .== 1), rt_q = string(q(rt)), rt_sd = round(std(filter(!isnan, rt)); digits = 3)))
end
function kelly(label, m)
    rng = StableRNG(1); rt = Float64[]; M = zeros(L)
    for i in 1:N
        r, tr = UnfoldSim.SSM_Simulate(rng, m, fs, L); push!(rt, r / fs); M .+= tr[1:L]
    end
    traces[label] = M ./ N
    push!(rows, (; model = label, accuracy = 1.0, rt_q = string(q(rt)), rt_sd = round(std(rt); digits = 3)))
end
port("Kelly/O'Connell code, O'Connell 2025 settings (d=6, U=1, Terz=st=0.1 width)"; d = 6.0)
port("Kelly/O'Connell code, NHB 2024 settings (d=6, U=0.5, Terz=st=0.2 width)"; d = 6.0, U = 0.5, Terz = 0.2, st = 0.2)
kelly("UnfoldSim KellyModel defaults (σ sens = σ motor = 0.1, motor mean 0.4)", UnfoldSim.KellyModel())
kelly("UnfoldSim KellyModel matched means/spreads (motor mean 0.1, uniform widths 0.1, no start point var)",
      kelly_model(sens_sd = width_to_sd(0.1), motor_sd = width_to_sd(0.1), dist = :uniform, motor_delay = 0.1,
                  start_point_variability = 0.0, start_point_distribution = truncated(Normal(0, 1e-9), upper = 1.0),
                  post_accumulation_distribution = Uniform(0.1, 0.3)))
df = DataFrame(rows); show(stdout, df; allcols = true, truncate = 200); println()
CSV.write(joinpath(@__DIR__, "..", "results", "validation_behaviour.csv"), df)
fig = Figure(size = (900, 420)); ax = Axis(fig[1, 1], xlabel = "time from stimulus (s)", ylabel = "mean CPP (a.u.)",
    title = "Stimulus-locked mean CPP: Kelly/O'Connell implementation vs UnfoldSim KellyModel")
for (k, v) in sort(collect(traces); by = first)
    lines!(ax, tv, v; label = k)
end
Legend(fig[2, 1], ax; nbanks = 1, framevisible = false, labelsize = 10)
save(joinpath(@__DIR__, "..", "figures", "validation_cpp_traces.png"), fig, px_per_unit = 2)
