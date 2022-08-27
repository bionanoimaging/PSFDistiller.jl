using Test
using PSFDistiller

@testset "Testset distille_PSF" begin
sz = (256,256,100)
# create random delta peaks
beads = zeros(sz); N = 10
beads[rand(1:prod(sz),N)] .= 1.0

sigma = (2.3, 4.4, 3.5)
perfect_img = gaussf(beads, sigma)
# n_img = poisson(perfect_img)

mypsf, rois, positions, selected, params, fwd = distille_PSF(perfect_img)
@test all(abs.(params[:σ] .- sigma) .< 1e-4)

params2, fwd2, allp2 = PSFDistiller.gauss_fit(mypsf)
@test all(abs.(params2[:σ] .- sigma) .< 1e-4)

end
