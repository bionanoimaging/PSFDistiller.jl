export align_all, distille_PSF, nn_mutual, select_rois, clear_border, border_dist, remove_border, precise_extract

"""
    align_all(rois; shifts=nothing, optimizer=LBFGS(), iterations=500, verbose=false)

aligns a series of regions of interest `rois` of equal sizes to another. This is achieved by fitting a Gaussian to each of the ROIs and applying a Fourier-based shift
to align all of them to the center. The aligned raw data is then summed.
#
`positions`: If provided, these will be added to the determined shifts for the valid fits.
"""
function align_all(rois; shifts=nothing, positions=nothing, iterations=500, verbose=false)
    res_shifts = []
    shifted = []
    if length(rois) == 0
        return [], [], []
    end
    mysum = zeros(Float32, size(rois[1]))
    for n=1:length(rois)
        myshift, valid = let 
            if isnothing(shifts)
                to_center = rois[n] # gaussf(rois[n], 1.0)
                params, _, _ = gauss_fit(to_center, iterations=iterations, verbose=verbose)
                params[:μ], all(params[:FWHMs] .< size(rois[n])) && all(params[:FWHMs] .> 0.0)
            else
                shifts[n], true
            end
        end
        if !valid
            println("Warning fit failed (sigma .> roi_size). discarding bead $n")
        else
            if isnothing(positions)
                push!(res_shifts, myshift)
            else
                push!(res_shifts, myshift .+ positions[n])
            end                
            ashifted = shift(rois[n], .-myshift)
            push!(shifted, ashifted)
            mysum .+= ashifted
        end
    end
    println("Averaged $(length(shifted)) beads.")
    return mysum./length(shifted), shifted, res_shifts
end

"""
    nn_mutual(points)
    returns the mutual nearest neighbor distances and the index of the nearest neighbor, both in matrix form.
"""
function nn_mutual(points)
    kdtree = KDTree(points)
    nns, nndist = knn(kdtree, points, 2)
    nns = [mynn[1] for mynn in nns]
    nndist = [myd[1] for myd in nndist]
    return nndist, nns
end

"""
    select_rois(img::AbstractArray, roi_pos::Vector; valid=nothing, roi_size=Tuple(15 .* ones(Int, ndims(img))))

extracts a series of ROIs based from the array `img` on the positions supplied in `Vector`.
The latter can be a vector of Integer positions or a vector of `CartesionCoord` as obtained by
Image.local.

#arguments
+ `img`: image to extract ROIs from
+ `roi_pos`: a list of ND-indices where to extract. Alternatively also a matrix works, where the columns correspond to the positions.
+ `valid`: if provided a binary vector indicates which of the positions are valid and will be extracted
+ `roi_size`: the size of the region of interest to extract at each position via `NDTools.select_region()` 

#returns
a tuple of a vector of extracted regions of interest and a list of cartesian indices at which they were extracted.
"""
function select_rois(img::AbstractArray, roi_pos::Vector; valid=nothing, roi_size=Tuple(15 .* ones(Int, ndims(img))))
    rois = []
    cart_list = Vector{CartesianIndex{ndims(img)}}([])
    for n in 1:length(roi_pos)
        if isnothing(valid) || valid[n]
            pos = Tuple(round.(Int,roi_pos[n]))
            push!(rois, select_region(img, center=pos, new_size=roi_size))
            push!(cart_list, CartesianIndex(pos))
        end
    end
    return rois, cart_list
end

function select_rois(img::AbstractArray, roi_pos::Matrix; valid=nothing, roi_size=Tuple(15 .* ones(Int, ndims(img))))
    select_rois(img, mat_to_vec(roi_pos); valid=valid, roi_size=roi_size)
end

"""
    clear_border(img, roi_size; pad_value=zero(eltype(img)))

inserts `pad_value` into the border region.

"""
function clear_border(img, roi_size; pad_value=zero(eltype(img)))
    sz = size(img)
    return select_region(select_region(img, new_size=sz .- roi_size, pad_value=pad_value), new_size=sz, pad_value=pad_value)
end

"""
    border_dist(vec, sz)

calculates the closest distance to the border along each dimension, given a vector of coordinate vectors. 
"""
function border_dist(vec, sz, ndims=length(sz))
    [min.(Tuple(v[1:ndims]).-1, sz[1:ndims].-Tuple(v[1:ndims])) for v in vec]
end

"""
    remove_border(vec::Vector, roi_size)

removes all positions which are too close to the border in a vector of coordinates.
"""
function remove_border(vec::Vector, sz, roi_size; valid=nothing)
    res = []
    dist = border_dist(vec, sz, length(roi_size))
    for n=1:length(vec)
        if all(dist[n] .> roi_size) && (isnothing(valid) || valid[n])
            push!(res, vec[n])
        end
    end
    return res
end

function remove_border(mat::Matrix, sz, roi_size; valid=nothing)
    remove_border(mat_to_vec(mat), sz, roi_size; valid=valid)
end

"""
    precise_extract(img, positions, roi_size)

extracts a region of size `roi_size` from the ND-dataset `img` at each position centering each region precisely at the supixel cooredinates as specified by `positions`.

#returns
a tuple of `(extracted, cart_ids, mysum)` with `extracted` referring to the extracted and shifted regions and `mymean` being the average of all of these regions.
`cart_ids` refers to cartesian indices of the nearest pixel positions.
"""
function precise_extract(img, positions, roi_size)
    all_max_vec = [round.(Int, positions[n]) for n=1:length(positions)]
    shifts = [positions[n].- all_max_vec[n] for n=1:length(positions)]
    rois, cart_ids = select_rois(img, all_max_vec; roi_size=roi_size);
    mymean, extracted, shifts = align_all(rois, shifts=shifts);
    return extracted, cart_ids, mymean
end

"""
    distille_PSF(img, σ=1.3; positions=nothing, force_align=false, rel_thresh=0.1, min_dist=16.0, roi_size = Tuple(15 .* ones(Int, ndims(img))), upper_thresh=nothing, pixelsize=1.0)

automatically extracts multiple PSFs from one dataset, aligns and averages them. The input image `img` should contain a sparse set of PSF measurements
obtained by imaging beads, QDots or NV centers.
If you want to apply this to multicolor or multimode datasets, run it first on one channel and then again on the other channels using the `positions` argument.

#arguments
+ `img`: ND-image to distill the PSF from
+ `σ`: the size of the filtering kernel usde in the preprocessing step before finding the local maxima. This may be noise-dependent.
+ `positions`: if a list of (sub-pixel precision) positions is provided, these will be used instead of aligning them. 
+ `force_align`: If true, the subpixel-alignment will be done, even though positions are given.
+ `rel_trhesh`: the threshold specifying which local maxima are valid. This is a relative minimum brightness value compared to the maximum of the Gauss-filtered data.
+ `min_dist`: The minimum distance in pixels to the nearest other maximum.
+ `roi_size`: The size of the region of interest to extract. The default is 2D but this should also work for higher dimensions assuming the size of `img` for the other dimensions.
+ `upper_thresh`: if provided, also an upper relative threshold will be applied eliminating the very bright particles. If you have clusters, try `upper_thresh = 0.45`.
+ `pixelsize`: size of a pixel. Scales the FWHMs and σ in the result parameters.
+ `bg_alg`: Algorithm to use for background removal

#returns
a tuple of  `(mypsf, rois, positions, selected, params, fwd)`
+ `mypsf`: the distilled PSF
+ `rois`: the individual aligned beads as a vector of images
+ `positions`: the subpixels shifts as a vector of vectors
+ `selected`: a Float32 image with selection rings around each bead 
+ `params`: the resulting fit parameters of the fit of the final PSF as a named tuple
+ `fwd`: the (forward projected) fit results in the selected ROIs 

"""
function distille_PSF(img, σ=1.3; positions=nothing, force_align=false, rel_thresh=0.1, min_dist=nothing, roi_size=Tuple(15 .* ones(Int, ndims(img))), verbose=false, upper_thresh=nothing, pixelsize=1.0, preferred_z=nothing, bg_alg=GaussMin(σ))
    # may also upcast to Float32
    img, bg = remove_background(GaussMin(2 .* σ), img)
    println("Subtracted a background of $(bg)")
    if isnothing(min_dist)
        min_dist = maximum(roi_size)
    end
    all_max_vec=[]
    if isnothing(positions)
        gimg, roisz = let
            # decide whether to select only one slice for finding the beads
            if isnothing(preferred_z)
                gaussf(img, σ), roi_size
            else
                gaussf(img[:,:,preferred_z], σ), roi_size[1:2] #slice(img,3,preferred_z)
            end
        end
        gimg .*= (gimg .> maximum(gimg) .* rel_thresh)
        # gimg = clear_border(gimg, roi_size)
        all_max = findlocalmaxima(gimg)
        all_val = gimg[all_max]
        all_max_vec = cart_to_mat(all_max)
        println("Found $(length(all_max)) maxima to consider.")

        nndist, _ = nn_mutual(all_max_vec)
        valid_nn = nndist .> min_dist

        println("Found $(sum(valid_nn)) beads to consider with sufficient distance min_dist=$(min_dist).")
        if !isnothing(upper_thresh)
            max_b = maximum(all_val) .* upper_thresh
            valid_nn .*= (all_val .< max_b)
            println("Found $(sum(valid_nn)) beads to consider with correct brightness $(max_b).")
        end
        all_max_vec = remove_border(all_max_vec, size(gimg), roisz./2; valid=valid_nn)
        println("Removed border. $(length(all_max_vec)) beads remaining.")
        if length(all_max_vec) < 1
            error("No beads were found which are valid.")
        end

        all_max_vec = let
            # decide whether to select only one slice for finding the beads
            if isnothing(preferred_z)
                all_max_vec
            else
                [vcat(all_max_vec[n][1:2],preferred_z) for n=1:length(all_max_vec)]
            end
        end
        rois, cart_ids = select_rois(img, all_max_vec; roi_size=roi_size);
        force_align=true
    else
        rois, cart_ids, psf = precise_extract(img, positions, roi_size)
        all_max_vec = positions
    end
    if force_align
        println("Averaging $(length(rois)) regions of interest.")
        psf, rois, positions = align_all(rois, positions=all_max_vec, verbose=verbose);
    end
    selected = make_rings(size(img), cart_ids, expand_size(roi_size,size(img)))

    psf, _ = remove_background(bg_alg, psf)
    if !isempty(psf)
        params, fwd, allp = gauss_fit(psf, verbose=verbose, pixelsize=pixelsize);

        return (psf, rois, positions, selected, params, fwd)
    else
        return [],[],[],positions,[],[]
    end
end

function make_rings(sz, cart_ids, roi_size)
    res = zeros(sz)
    res[cart_ids] .= 1.0
    res = gaussf(res)
    tmp = Float32.((rr(sz, scale=2 ./roi_size) .> 0.9) .* (rr(sz, scale=2 ./roi_size) .< 1.1))
    return conv_psf(res,tmp)
end
