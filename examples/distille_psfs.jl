using PSFDistiller
using PointSpreadFunctions
using NDTools
using FourierTools
using InverseModeling
using Noise

sz = (256,256,100)
sampling = (0.020,0.020,0.050)
# simulate a confocal PSF
pp_em = PSFParams(0.5,1.4,1.52; mode=ModeConfocal);

# aberrations = Aberrations();
aberrations = Aberrations([Zernike_HorizontalComa,Zernike_Tip],[0.2,0.2]);
pp_ex = PSFParams(pp_em; λ=0.488, method=MethodPropagateIterative, aplanatic=aplanatic_illumination, aberrations=aberrations);
p_conf = psf(sz,pp_em; pp_ex=pp_ex, pinhole=1.5, sampling=sampling);
params_c, fwd_c, optim_res = PSFDistiller.gauss_fit(p_conf)
@ve fwd_c p_conf  p_conf

n_photons = 10000
p_conf = n_photons .* p_conf ./ maximum(p_conf)

# create random delta peaks
beads = zeros(sz); N = 10
beads[rand(1:prod(sz),N)] .= 1.0

perfect_img = conv_psf(beads, p_conf)
n_img = poisson(perfect_img)

mypsf, rois, positions, selected, params, fwd = distille_PSF(perfect_img)
params2, fwd2, allp2 = PSFDistiller.gauss_fit(mypsf)

@show sampling .* params[:FWHMs]
@show sampling .* params2[:FWHMs]
@show params2[:R2]
# @vt fwd2 apsf (fwd2 .- mypsf)
@show mpsf = sum(mypsf)/prod(size(mypsf))

@show 1.0 - sum(abs2.(fwd2 .- mypsf)) / sum(abs2.(mypsf .- mpsf))

# visualize the result using View5D.jl
# @ve n_img selected 
