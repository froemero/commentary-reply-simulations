# Reproduce the grid of Supplementary Figure 2 (reply): KellyModel delay variabilities given as
# Normal SDs of 0, 0.1, 0.2, 0.3 s, with and without urgency.
include(joinpath(@__DIR__, "..", "src", "ReplySims.jl")); using .ReplySims
using CSV, DataFrames
grid = [0.0, 0.1, 0.2, 0.3]
seeds = 1:3
rows = []
for urg in (false, true), s in grid, m in grid, seed in seeds
    r = run_cell(seed; sens_sd = s, motor_sd = m, urgency = urg)
    push!(rows, r)
end
CSV.write(joinpath(@__DIR__, "..", "results", "suppfig2_grid.csv"), DataFrame(rows))
