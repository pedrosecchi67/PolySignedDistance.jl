module PointHashing

    export PointHash

    """
    ```
        struct PointHash
            point_dict::AbstractDict
            digits::Int
        end
    ```

    Struct containing a hashmap of points
    """
    struct PointHash
        point_dict::AbstractDict
        index_dict::AbstractDict
        digits::Int

        PointHash(
            ;
            digits::Int = 7,
        ) = new(
            Dict{Tuple, Int64}(),
            Dict{Int64, Tuple}(),
            digits
        )
    end

    """
    ```
        function Base.push!(
            hsh::PointHash,
            pt::AbstractVector
        )
    ```

    Add point to hash, return index and hash (tuple of coordinates)
    """
    function Base.push!(
        hsh::PointHash,
        pt::AbstractVector
    )

        pthash = tuple(
            map(
                x -> round(x; digits = hsh.digits),
                pt,
            )...
        )

        if haskey(hsh.point_dict, pthash)
            return (hsh.point_dict[pthash], pthash)
        end

        ind = hsh.point_dict.count + 1

        hsh.point_dict[pthash] = ind
        hsh.index_dict[ind] = pthash

        (ind, pthash)

    end

    """
    ```
        Base.getindex(
            hsh::PointHash,
            pt::AbstractVector
        )
    ```

    Find index for point
    """
    Base.getindex(
        hsh::PointHash,
        pt::AbstractVector
    ) = let pthash = map(
        x -> round(x; digits = hsh.digits),
        pt,
    )
        hsh.point_dict[pthash]
    end

    """
    ```
        Base.getindex(
            hsh::PointHash,
            i::Int,
        )
    ```

    Find point of index i
    """
    Base.getindex(
        hsh::PointHash,
        i::Int,
    ) = collect(
        hsh.index_dict[i]
    )

    """
    ```
        function points(hsh::PointHash)
    ```

    Get all points in the hashmap using an `(ndim, npoints)` matrix
    """
    function points(hsh::PointHash)

        X = hcat(
            map(collect, collect(keys(hsh.point_dict)))...
        )
        i = collect(values(hsh.point_dict))

        i = invperm(i)

        X[:, i]

    end

    """
    ```
        function PointHash(
            points::AbstractMatrix;
            digits::Int = 7,
        )
    ```

    Construct a point hash from a set of points in an `(ndim, npoints)` matrix
    """
    function PointHash(
        points::AbstractMatrix;
        digits::Int = 7,
    )

        hsh = PointHash(; digits = digits,)

        for pt in eachcol(points)
            push!(hsh, pt)
        end

        hsh

    end

    """
    ```
        function filter_cloud(
            pts::AbstractMatrix;
            digits::Int = 7,
        )
    ```

    Filter a cloud of points and merge points with the same representation up to 
    `digits`.

    Returns a vector of indices for the original points, and a matrix with the new ones
    """
    function filter_cloud(
        pts::AbstractMatrix;
        digits::Int = 7,
    )

        hsh = PointHash(; digits = digits,)
        inds = Vector{Int64}(undef, size(pts, 2))

        for (ipt, pt) in enumerate(eachcol(pts))
            i, _ = push!(hsh, pt)

            inds[ipt] = i
        end

        (
            inds, points(hsh)
        )
    
    end

    """
    ```
        @inline npoints(hsh::PointHash) = hsh.point_dict.count
    ```

    Get number of points in hash
    """
    @inline npoints(hsh::PointHash) = hsh.point_dict.count

end

#=
hsh = PointHashing.PointHash()

@show push!(hsh, [1.0, 1e-3, 1.0 + 1e-8])
@show push!(hsh, [1.0 + 1e-6, 1e-3, 1.0 + 1e-8])
@show push!(hsh, [1.0, 1e-3, 1.0 + 1e-8])

@show PointHashing.points(hsh)

hsh = PointHashing.PointHash(
    [
        1.0 (1.0 + 1e-6) 1.0;
        1e-3 1e-3 1e-3;
        (1.0 + 1e-8) (1.0 + 1e-8) (1.0 + 1e-8)
    ]
)

@show PointHashing.points(hsh)

@show hsh[[1.0, 1e-3, 1.0 + 1e-8]]
@show hsh[2]

@show PointHashing.filter_cloud( 
    [
        1.0 (1.0 + 1e-6) 1.0;
        1e-3 1e-3 1e-3;
        (1.0 + 1e-8) (1.0 + 1e-8) (1.0 + 1e-8)
    ]
)

@info "Performance test"

for i = 1:10
    pts = rand(3, 3)

    @time PointHashing.filter_cloud(pts; digits = 7,)
end
=#
