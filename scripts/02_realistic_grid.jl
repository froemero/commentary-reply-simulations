# Same analysis on a finer grid around empirically realistic delay variabilities
# (visual encoding SD ≈ 15 ms, ≤ ~30 ms; EMG motor time SD ≈ 25 ms, 17–46 ms), for
# (a) KellyModel with Normal delays (as in the reply's Supp. Fig. 2),
# (b) KellyModel with uniform delays of the same SD,
# (c) the Kelly & O'Connell two-accumulator implementation (uniform delays, widths = SD·√12).
include(joinpath(@__DIR__, "..", "src", "ReplySims.jl")); using .ReplySims
using CSV, DataFrames
grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
seeds = 1:3
rows = []
for urg in (false, true), s in grid, m in grid, seed in seeds
    push!(rows, run_cell(seed; sens_sd = s, motor_sd = m, urgency = urg, dist = :normal))
    push!(rows, run_cell(seed; sens_sd = s, motor_sd = m, urgency = urg, dist = :uniform))
    push!(rows, run_cell_port(seed; sens_sd = s, motor_sd = m, urgency = urg))
end
CSV.write(joinpath(@__DIR__, "..", "results", "realistic_grid.csv"), DataFrame(rows))
println("done")
