using PSFDistiller
using PointSpreadFunctions
using NDTools
using FourierTools
using InverseModeling

sz = (256,256,100)
sampling = (0.040,0.040,0.100)
# simulate a confocal PSF
pp_em = PSFParams(0.5,1.4,1.52; mode=ModeConfocal);
pp_ex = PSFParams(pp_em; λ=0.488, aplanatic=aplanatic_illumination);
p_conf = psf(sz,pp_em; pp_ex=pp_ex, pinhole=0.1, sampling=sampling);

beads = zeros(sz)
N = 10
beads[rand(1:prod(sz),N)] .= 1.0

perfect_img = conv_psf(beads, p_conf)

apsf, rois, positions, selected, params, fwd = distille_PSF(perfect_img)
params2, fwd2, allp2 = PSFDistiller.gauss_fit(apsf)

sampling .* params[:FWHM]
sampling .* params2[:FWHM]
params2[:R2]
@vt fwd2 apsf (fwd2 .- apsf)
mpsf = sum(apsf)/prod(size(apsf))

1.0 - sum(abs2.(fwd2 .- apsf)) / sum(abs2.(apsf .- mpsf))

# visualize the result using View5D.jl
# @ve perfect_img selected selected
