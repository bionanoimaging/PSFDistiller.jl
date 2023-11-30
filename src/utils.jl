export cart_to_mat, mat_to_vec, remove_background, gaussf

"""
    cart_to_mat(cart)

converts a list of cartesian indices to a matrix with the first dimension corresponding to the dimension direction.
    
"""
function cart_to_mat(cart)
    all_max_vec = zeros(length(cart[1]), length(cart))
    n=1
    for idx in cart
        all_max_vec[:,n] = [Tuple(idx)...] 
        n += 1
    end
    return all_max_vec
end

"""
    mat_to_vec(mat)

converts a 2D matrix to a vector of vectors, one vector for each column.

"""
function mat_to_vec(mat)
    [mat[:,n] for n=1:size(mat,2)]
end

"""
    gaussf(img::AbstractArray, sigma=1.0; dtype=Float32)
performs Gaussian filtering using FFTs. `sigma` denotes the kernel sizes along the various dimensions. 
#arguments
+ `img`: the image to apply the Gaussian filter to
+ ``
"""
function gaussf(img::AbstractArray{T}, sigma=1.0; dtype=Float32) where (T <: Real)
    dims=1:ndims(img)
    shiftdims = dims[2:end]
    img = dtype.(img)
    f = rfft(img, dims)
    return irfft(f .* ifftshift(gaussian(eltype(img), size(f), offset=CtrRFT, sigma=size(img) ./(2π .*sigma)), shiftdims), size(img, dims[1]), dims)
end

function gaussf(img::AbstractArray, sigma=1.0) 
    return (ifft(fft(img) .* ifftshift(gaussian(eltype(img), size(img), sigma=size(img) ./(2π*sigma)))))
end


"""
    remove_background(alg::BackgroundRemovalAlgorithm, img)

Remove the background from the input image using the specified background removal algorithm.

# Arguments
- `alg::BackgroundEstimationAlgorithm`: The background removal algorithm to use.
- `img`: The input image.

# Returns
- A tuple `(fg, bg)`, where `fg` is the image with the background removed and `bg` is the removed background.

# Extended help
- The background removal is implemented in a way where no zero-clipping is performed
# Examples
```jldoctest
julia> img = [1 2 3; 4 5 6; 7 8 9]
3×3 Array{Int64,2}:
 1  2  3
 4  5  6
 7  8  9

"""
abstract type BackgroundEstimationAlgorithm end
function remove_background(alg::BackgroundEstimationAlgorithm, img)
    bg = background(alg, img)
    return img .- bg, bg
end

Base.@kwdef struct GaussMin <: BackgroundEstimationAlgorithm
    σ = 5.0
end

# TODO: Make this a doc test <29-11-23> 
"""
    background(alg::GaussMin, img)

Estimate the background of `img` by applying a Gaussian filter to the data and taking the minimal value of the result

 # Extended help

```julia
julia> img = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> background(GaussMin(), img)
5.0f0

julia> fg, bg = remove_background(GaussMin(), img);

julia> fg
3×3 Matrix{Float32}:
 -4.0  -1.0  2.0
 -3.0   0.0  3.0
 -2.0   1.0  4.0
```
 """
background(alg::GaussMin, img) = minimum(gaussf(img, alg.σ))

struct Quantile{Q} <: BackgroundEstimationAlgorithm end
const Median = Quantile{0.5}
function Quantile(Q)
    0.0 < Q < 1.0 || throw(DomainError("quantile must be between 0 and 1"))
    Quantile{Q}()
end

"""
    background(alg::Quantile{Q}, img) where {Q}

Estimate the background of `img` using the `Q` quantile.

 # Extended help

```julia
julia> img = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> background(Quantile{0.2}(), img)
2.6

julia> fg,bg = remove_background(Quantile{0.2}(), img);

julia> fg
3×3 Matrix{Float64}:
 -1.6  1.4  4.4
 -0.6  2.4  5.4
  0.4  3.4  6.4

julia> Median
Quantile{0.5}

julia> background(Median(), img)
5.0

julia> fg,bg = remove_background(Median(), img);

julia> fg
3×3 Matrix{Float64}:
 -4.0  -1.0  2.0
 -3.0   0.0  3.0
 -2.0   1.0  4.0
```
"""
background(alg::Quantile{Q}, img) where {Q} = quantile(img[:], Q)
