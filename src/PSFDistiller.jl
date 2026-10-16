module PSFDistiller

using IndexFunArrays
using ImageFiltering: findlocalmaxima # lightweight; the Images.jl umbrella is not needed
using NearestNeighbors # to discard them in PSF extraction
using InverseModeling # for gauss_fit
using FourierTools # for shift
using FFTW # for rfft
using NDTools # for select_region
using StatsBase

include("psf_extraction.jl")
include("utils.jl")

end # module
