# (1) Stimulus- and response-locked fast vs slow averages for example cells, for epoch / deconv / residual.
# (2) One-factor-at-a-time ablation: which KellyModel-vs-Kelly/O'Connell differences change res_norm.
include(joinpath(@__DIR__, "..", "src", "ReplySims.jl")); using .ReplySims
using UnfoldSim, Unfold, Distributions, StableRNGs, DataFrames, Statistics, CSV, CairoMakie
fs = 100; τS = (-0.2, 1.6); τR = (-1.6, 0.4); L = 200
root = joinpath(@__DIR__, "..")

kelly_trial(model) = rng -> begin r, tr = UnfoldSim.SSM_Simulate(rng, model, fs, L); (r / fs, tr) end
port_trial(; kw...) = rng -> begin
    dt = 1 / fs; tv = collect(0:dt:(2.0 - dt))
    r, c, cpp = kelly_port_trial(rng, tv, dt; d = 6.0, kw...); (r, cpp)
end

function curves(eeg, evts)
    out = Dict{Tuple{String,String,String},Tuple{Vector{Float64},Vector{Float64}}}()  # (method, event, group) => (t, y)
    function ep(signal, label)
        for (ev, τ) in (("S", τS), ("R", τR))
            tbl = evts[evts.event .== ev, :]
            e, t = Unfold.epoch(data = signal, tbl = tbl, τ = τ, sfreq = fs)
            ok = .!vec(any(ismissing.(e), dims = (1, 2))); e = coalesce.(e[:, :, ok], 0.0); tb = tbl[ok, :]
            for g in ("fast", "slow")
                out[(label, ev, g)] = (collect(t), vec(mean(e[1, :, tb.rtgroup .== g], dims = 2)))
            end
        end
    end
    ep(eeg, "epoch")
    f = @formula(0 ~ 1 + rtgroup)
    m = fit(UnfoldModel, ["S" => (f, firbasis(τ = τS, sfreq = fs)), "R" => (f, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    eff = effects(Dict(:rtgroup => ["fast", "slow"]), m)
    for ev in ("S", "R"), g in ("fast", "slow")
        e = eff[(eff.eventname .== ev) .& (eff.rtgroup .== g), :]
        out[("deconv", ev, g)] = (e.time, e.yhat)
    end
    f0 = @formula(0 ~ 1)
    m0 = fit(UnfoldModel, ["S" => (f0, firbasis(τ = τS, sfreq = fs)), "R" => (f0, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    ep(vec(Unfold.residuals(m0, eeg)), "residual")
    out
end

# ---------- (1) example cells
cells = [
    ("A: KellyModel, urgency, σ 0/0 (deconv 3.0, residual 0.27; original ≈0.7 / ≈0.05)", kelly_trial(kelly_model(sens_sd = 0.0, motor_sd = 0.0, urgency = true))),
    ("B: KellyModel, no urgency, σ 0/0 (deconv 1.6, residual 0.07; original ≈0.3 / ≈0.45)", kelly_trial(kelly_model(sens_sd = 0.0, motor_sd = 0.0, urgency = false))),
    ("C: KellyModel, urgency, σ 0.1/0.1 (deconv 0.79, residual 0.40; original ≈0.8 / ≈0.6)", kelly_trial(kelly_model(sens_sd = 0.1, motor_sd = 0.1, urgency = true))),
    ("D: KellyModel, urgency, realistic σ 0.015/0.025", kelly_trial(kelly_model(sens_sd = 0.015, motor_sd = 0.025, urgency = true))),
    ("E: Kelly/O'Connell code, urgency, realistic σ 0.015/0.025", port_trial(Terz = sd_to_width(0.015), st = sd_to_width(0.025))),
]
for (k, (label, trial)) in enumerate(cells)
    eeg, evts = simulate_continuous(StableRNG(1), trial; N = 500, fs)
    c = curves(eeg, evts)
    fig = Figure(size = (900, 820))
    Label(fig[0, 1:2], label; fontsize = 13, tellwidth = false)
    for (r, meth) in enumerate(("epoch", "deconv", "residual")), (col, ev) in enumerate(("S", "R"))
        ax = Axis(fig[r, col]; title = "$(meth) — $(ev == "S" ? "stimulus" : "response")-locked", xlabel = "time (s)",
                  ylabel = "a.u.", titlesize = 12)
        for (g, colr) in (("fast", :firebrick), ("slow", :steelblue))
            t, y = c[(meth, ev, g)]; lines!(ax, t, y; color = colr, label = g)
        end
        vlines!(ax, [0]; color = :gray, linestyle = :dash)
        r == 1 && col == 1 && axislegend(ax; position = :lt, framevisible = false)
    end
    linkyaxes!(contents(fig[1:3, 1:2])...)
    save(joinpath(root, "figures", "erp_example_$(Char('A' + k - 1)).png"), fig, px_per_unit = 2)
end

# ---------- (2) ablation, urgency on, 3 seeds, two cells
dt0 = 1 / fs
matched(; sens, motor, kw...) = kelly_model(; sens_sd = sens, motor_sd = motor, urgency = true, dist = :uniform, motor_delay = 0.1,
    start_point_distribution = truncated(Normal(0, 1e-9), upper = 1.0), post_accumulation_distribution = Uniform(0.1, 0.3), kw...)
function res(trial)
    v = map(1:3) do seed
        eeg, evts = simulate_continuous(StableRNG(seed), trial; N = 500, fs); m = analyse(eeg, evts, fs)
        (m["deconv"] / m["epoch"], m["residual"] / m["epoch"])
    end
    (round(mean(first.(v)); digits = 2), round(mean(last.(v)); digits = 2))
end
rows = []
for (cname, s, m) in (("σ 0/0", 0.0, 0.0), ("realistic σ 0.015/0.025", 0.015, 0.025), ("σ 0.1/0.1", 0.1, 0.1))
    variants = [
        "Kelly/O'Connell code" => port_trial(Terz = sd_to_width(s), st = sd_to_width(m)),
        "KellyModel matched (uniform delays, motor 0.1, no start-point var, post-acc U(.1,.3))" => kelly_trial(matched(sens = s, motor = m)),
        "  + bound 0.7 (≈ Kelly effective bound)" => kelly_trial(matched(sens = s, motor = m, boundary = 0.7)),
        "  + start-point variability σ 0.2 in evidence (KellyModel default)" => kelly_trial(matched(sens = s, motor = m, start_point_distribution = truncated(Normal(0, 0.2), upper = 1.0))),
        "  + motor mean 0.4 (KellyModel default)" => kelly_trial(matched(sens = s, motor = m, motor_delay = 0.4)),
        "  + post-acc N(0.1, 0.001) (KellyModel default)" => kelly_trial(matched(sens = s, motor = m, post_accumulation_distribution = truncated(Normal(0.1, 0.001), lower = 0))),
        "  + Normal delays (KellyModel default)" => kelly_trial(kelly_model(sens_sd = s, motor_sd = m, urgency = true, dist = :normal, motor_delay = 0.1,
              start_point_distribution = truncated(Normal(0, 1e-9), upper = 1.0), post_accumulation_distribution = Uniform(0.1, 0.3))),
        "KellyModel defaults (as in reply Supp. Fig. 2)" => kelly_trial(kelly_model(sens_sd = s, motor_sd = m, urgency = true)),
    ]
    for (vname, tr) in variants
        d, r = res(tr); push!(rows, (; cell = cname, variant = vname, deconv = d, residual = r)); println(cname, " | ", vname, " | ", d, " | ", r)
    end
end
CSV.write(joinpath(root, "results", "ablation.csv"), DataFrame(rows))
