# Least-squares fit of an axis-aligned N-dimensional Gaussian plus a constant
# offset. Self-contained (Optim L-BFGS with an analytic gradient) so that
# PSFDistiller does not need InverseModeling.jl: its dependency ComponentArrays
# makes Zygote-based code elsewhere in the same Julia session (e.g.
# DeconvOptim's L-BFGS deconvolution in DeconvGUI) crash with an access
# violation in Julia's method lookup.

"""
    gauss_fit(meas; pixelsize=nothing, iterations=500, verbose=false)

Fit `offset + i0 * exp(-Σ_d (x_d - μ_d)² / (2σ_d²))` to the N-dimensional array
`meas` (least squares).

# Returns
a tuple `(params, fwd, res)`:
+ `params`: named tuple `(offset, μ, σ, i0, FWHMs)` as in `InverseModeling.gauss_fit`:
  `μ` in pixels relative to the array centre (`size .÷ 2 .+ 1`), `σ` and `FWHMs`
  multiplied by `pixelsize` when given.
+ `fwd`: the fitted model evaluated on the grid of `meas`.
+ `res`: the Optim result.
"""
function gauss_fit(meas::AbstractArray{<:Real,N}; pixelsize=nothing, iterations=500, verbose=false) where {N}
    sz = size(meas)
    ctr = sz .÷ 2 .+ 1
    scale = Float64(maximum(abs, meas))
    scale = scale > 0 ? scale : 1.0
    m = Float64.(meas) ./ scale                    # normalized for a well-conditioned fit
    coords = ntuple(d -> reshape(Float64.(collect(1:sz[d]) .- ctr[d]), ntuple(k -> k == d ? sz[d] : 1, N)), N)

    # Start values from moments of the part above 20% of the peak (as
    # InverseModeling.gauss_start): offset from the border, i0 from the centre.
    border = trues(sz)
    border[(ntuple(d -> min(2, sz[d]):max(1, sz[d] - 1), N))...] .= false
    offset0 = any(border) ? sum(m[border]) / count(border) : minimum(m)
    peak = m[ntuple(d -> max(1, ctr[d] - 1):min(sz[d], ctr[d] + 1), N)...]
    i00 = sum(peak) / length(peak) - offset0
    w = max.(m .- offset0 .- 0.2 * i00, 0.0)
    sw = sum(w)
    μ0 = sw > 0 ? [sum(w .* coords[d]) / sw for d in 1:N] : zeros(N)
    σ0 = sw > 0 ? [max(0.7, sqrt(sum(w .* (coords[d] .- μ0[d]) .^ 2) / sw)) for d in 1:N] : ones(N)
    p0 = [offset0; max(i00, 1e-3); μ0; log.(σ0)]   # σ via its logarithm: stays positive

    function model(p)
        G = ones(Float64, sz)
        for d in 1:N
            G = G .* exp.(-(coords[d] .- p[2+d]) .^ 2 ./ (2 * exp(p[2+N+d])^2))
        end
        return G
    end
    f(p) = sum(abs2, p[1] .+ p[2] .* model(p) .- m) / 2
    function g!(Gr, p)
        G = model(p)
        r = p[1] .+ p[2] .* G .- m
        rag = r .* (p[2] .* G)
        Gr[1] = sum(r)
        Gr[2] = sum(r .* G)
        for d in 1:N
            σ2 = exp(p[2+N+d])^2
            xd = coords[d] .- p[2+d]
            Gr[2+d] = sum(rag .* xd) / σ2
            Gr[2+N+d] = sum(rag .* xd .^ 2) / σ2
        end
        return Gr
    end
    res = optimize(f, g!, p0, LBFGS(), Optim.Options(iterations=iterations))
    p = Optim.minimizer(res)

    σ = exp.(p[3+N:2+2N])
    if !isnothing(pixelsize)
        σ = σ .* collect(pixelsize .* ones(N))
    end
    FWHMs = σ .* (sqrt(log(2) * 2) * 2)
    fwd = eltype(meas) <: AbstractFloat ? eltype(meas).(scale .* (p[1] .+ p[2] .* model(p))) :
                                          scale .* (p[1] .+ p[2] .* model(p))
    verbose && println("FWHMs : $FWHMs, Iterations : $(Optim.iterations(res)), Norm: $(Optim.minimum(res))")
    params = (offset=p[1] * scale, μ=p[3:2+N], σ=σ, i0=p[2] * scale, FWHMs=FWHMs)
    return params, fwd, res
end
