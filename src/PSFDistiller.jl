module PSFDistiller

using IndexFunArrays
using NearestNeighbors # to discard them in PSF extraction
using Optim: Optim, optimize, LBFGS # for gauss_fit (see gauss_fit.jl)
using FourierTools # for shift
using AbstractFFTs # for rfft (the FFT library itself is loaded by the application)
using NDTools # for select_region
using StatsBase

include("local_maxima.jl")   # findlocalmaxima (as ImageFiltering's, without its FFTW dependency)
include("gauss_fit.jl")
include("psf_extraction.jl")
include("utils.jl")

end # module
