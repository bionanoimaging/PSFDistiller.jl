"""
    findlocalmaxima(img) -> Vector{CartesianIndex}

Coordinates of the elements that are strictly larger than all their neighbours (window 3 in
every dimension; neighbours outside the array are ignored, so edge elements are included).
Same result as `ImageFiltering.findlocalmaxima(img)` with its defaults; written out here so
that PSFDistiller does not need ImageFiltering (which pulls in the GPL-licensed FFTW).
"""
function findlocalmaxima(img::AbstractArray{T,N}) where {T,N}
    out = CartesianIndex{N}[]
    R = CartesianIndices(img)
    W = CartesianIndices(ntuple(_ -> -1:1, Val(N)))
    z = zero(CartesianIndex{N})
    @inbounds for i in R
        v = img[i]
        ismax = true
        for j in W
            j == z && continue
            k = i + j
            k in R || continue
            if !(v > img[k])
                ismax = false
                break
            end
        end
        ismax && push!(out, i)
    end
    return out
end
