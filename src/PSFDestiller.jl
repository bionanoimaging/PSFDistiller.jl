module PSFDestiller

using IndexFunArrays
using Images # find_local_maxima
using NearestNeighbors # to discard them in PSF extraction
using InverseModeling # for gauss_fit
using FourierTools # for shift
using FFTW # for rfft
using NDTools # for select_region

include("psf_extraction.jl")
include("utils.jl")

end # module
