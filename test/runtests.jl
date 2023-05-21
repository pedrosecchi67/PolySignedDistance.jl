using PolySDF
using PolySDF: RayTracing

using Random: seed!
seed!(42)

include("ray_trace.jl")
include("sdf.jl")
