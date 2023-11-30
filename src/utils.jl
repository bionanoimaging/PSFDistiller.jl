export cart_to_mat, mat_to_vec, remove_background, gaussf

"""
    cart_to_mat(cart)

Convert a list of cartesian indices to a matrix with the first dimension corresponding to the dimension direction.
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

Convert a 2D matrix to a vector of vectors, one vector for each column.
"""
function mat_to_vec(mat)
    [mat[:,n] for n=1:size(mat,2)]
end

"""
    gaussf(img::AbstractArray, sigma=1.0; dtype=Float32)

Perform Gaussian filtering using FFTs. `sigma` denotes the kernel sizes along the various dimensions. 

# Arguments
+ `img`: the image to apply the Gaussian filter to
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
    remove_background(img, sigma=5.0)

estimates the background by applying a Gaussian filter to the data and measuring for the minimum of this filtered data.
#arguments
+ `img`: the image to remove the background from
+ `sigma`: the kernelsize for the Gaussian filter
#returns
a tuple of the background-subtracted image and the background value.
Note that no zero-clipping is performed. The result type is by defaul Float32, if the image itself is not FLoat64 or Complex.
"""
function remove_background(img, sigma=5.0; dtype=Float32)
    bg = minimum(gaussf(img,sigma; dtype=dtype))
    return img .- bg, bg
end
