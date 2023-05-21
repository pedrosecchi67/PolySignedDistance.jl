module RayTracing

    using LinearAlgebra

    include("../bounding_box/bounding_boxes.jl") # adjust as necessary
    using .BoundingBox

    include("../point_merging/point_hashing.jl") # adjust as necessary
    using .PointHashing

    export Surface, isin, crossed_faces

    """
    Get simplex normal
    """
    function normal(simplex::AbstractMatrix)

        ϵ = eps(eltype(simplex))

        if size(simplex, 1) == 2
            v = simplex[:, 2] .- simplex[:, 1]

            return [
                - v[2], v[1]
            ] ./ (norm(v) + ϵ)
        end

        p0 = simplex[:, 1]

        n = cross(simplex[:, 2] .- p0, simplex[:, 3] .- p0)

        n ./ (norm(v) + ϵ)

    end

    """
    Find if a line connecting two points crosses a simplex
    """
    function crosses_simplex(
        simplex::AbstractMatrix,
        p1::AbstractVector,
        p2::AbstractVector,
        ϵ::Real = 0.0,
    )

        p0 = simplex[:, 1]

        nϵ = sqrt(eps(eltype(p0)))

        dp = (p2 .- p1)

        n = normal(simplex)
        dp .+= let p = n ⋅ dp
            n .* (
                p < 0.0 ?
                - nϵ :
                nϵ
            )
        end

        M = [(simplex[:, 2:end] .- p0) dp]

        M = pinv(M)

        ξ1 = M * (p1 .- p0)
        ξ2 = M * (p2 .- p0)

        if ξ1[end] * ξ2[end] > - ϵ
            return false
        end

        ξ1 = ξ1[1:(end - 1)]

        if any(
            x -> x < - ϵ,
            ξ1
        )
            return false
        end

        if sum(ξ1) > 1.0 + ϵ
            return false
        end

        true

    end

    """
    ```
        struct Surface
            points::AbstractMatrix
            simplices::AbstractMatrix
            bbox::Box

            ray_reference::AbstractVector
            reference_isin::Bool

            Surface(
                points::AbstractMatrix, # matrix (ndims, npts)
                simplices::AbstractMatrix; # matrix (ndims, nsimps)
                leaf_size::Int = 10, # leaf size for bounding box tree
                ray_reference = nothing, # default ray origin point
                reference_isin::Bool = false, # whether said origin is within the surface
                digits::Int = 0, # digits of precision for point merging. Not merged if zero
            )
        end
    ```

    Struct defining a surface
    """
    struct Surface
        points::AbstractMatrix
        simplices::AbstractMatrix
        bbox::Box

        ray_reference::AbstractVector
        reference_isin::Bool

        digits::Int

        function Surface(
            points::AbstractMatrix, # matrix (ndims, npts)
            simplices::AbstractMatrix; # matrix (ndims, nsimps)
            leaf_size::Int = 10, # leaf size for bounding box tree
            ray_reference = nothing, # default ray origin point
            reference_isin::Bool = false, # whether said origin is within the surface
            digits::Int = 12, # digits of precision for point merging. Not merged if zero
        )

            if digits > 0
                (inds, points) = PointHashing.filter_cloud(
                    points; digits = digits,
                )

                simplices = inds[simplices]

                simplices = hcat(
                    filter(
                        c -> length(unique(c)) == length(c),
                        collect(eachcol(simplices),),
                    )...
                )
            end

            let boxes = map(
                simp -> Box(points[:, simp]),
                eachcol(simplices)
            )

                if isnothing(ray_reference)
                    ray_reference = zeros(eltype(points), size(points, 1))

                    Lmax = maximum(
                        r -> maximum(abs.(r)),
                        eachrow(points)
                    ) * 2.0

                    ray_reference[1] = Lmax
                end

                new(
                    points, simplices,
                    Box(boxes; max_size = leaf_size,),
                    ray_reference,
                    reference_isin,
                    digits,
                )

            end
        end
    end

    """
    ```
        Surface(points::AbstractMatrix; kwargs...)
    ```

    Alternative constructor for a surface in two dimensions,
    which recieves a series of points in 2D space to be joined in a manifold.

    Only works for two dimensions!!
    """
    function Surface(points::AbstractMatrix; kwargs...)

        @assert size(points, 1) == 2 "Surface constructor must receive simplex matrix for 3 or higher-dimensional spaces"

        simplices = let inds = collect(1:size(points, 2))
            permutedims(
                [
                        inds (circshift(inds, -1))
                ]
            )
        end

        Surface(points, simplices; kwargs...)

    end


    """
    ```
        function isin(
            surf::Surface,
            point::AbstractVector,
            origin = nothing, # reference point. Defaults to the surface default (surf.ray_reference)
            origin_isin::Bool = false, # is reference within?
        )
    ```

    Get whether a point is within the surface using ray tracing
    """
    function isin(
        surf::Surface,
        point::AbstractVector,
        origin = nothing,
        origin_isin::Bool = false,
    )

        if isnothing(origin)
            origin = surf.ray_reference
            origin_isin = surf.reference_isin
        end

        fcs, _ = crossed_faces(
            surf, point, origin,
        )

        (length(fcs) % 2) != origin_isin

    end

    """
    Get intersection of line and simplex
    """
    function crossing_point(
        simplex::AbstractMatrix, p1::AbstractVector, p2::AbstractVector,
    )

        ϵ = eps(eltype(simplex))

        n = normal(simplex)

        p0 = simplex[:, 1]

        np1 = abs(n ⋅ (p1 .- p0)) + ϵ
        np2 = abs(n ⋅ (p2 .- p0)) + ϵ

        η = np1 / (np1 + np2)

        @. η * p2 + (1.0 - η) * p1

    end

    """
    ```
        function crossed_faces(
            surf::Surface,
            p1::AbstractVector,
            p2::AbstractVector,
        )
    ```

    Get a list of faces crossed by a line connecting two points.

    Returns vector of face indices and matrix of crossing points
    """
    function crossed_faces(
        surf::Surface,
        p1::AbstractVector,
        p2::AbstractVector,
    )

        line = [p1 p2]

        potential = find_potential_intersections(surf.bbox, line,)

        if length(potential) == 0
            return (potential, Matrix{Float64}(undef, length(p1), 0))
        end

        crossed = filter(
            s -> crosses_simplex(
                surf.points[:, surf.simplices[:, s]],
                p1, p2,
                sqrt(eps(eltype(p1)))
            ),
            potential
        )

        if length(crossed) == 0
            return (crossed, Matrix{Float64}(undef, length(p1), 0))
        end

        crossing_points = hcat(
            map(
                c -> let simp = surf.points[:, surf.simplices[:, c]]
                    crossing_point(simp, p1, p2)
                end, crossed,
            )...
        )

        inds, crossing_points = PointHashing.filter_cloud(crossing_points; digits = surf.digits,)

        ncrossed = Vector{Int64}(undef, size(crossing_points, 2))

        for (i, cr) in zip(inds, crossed)
            ncrossed[i] = cr
        end

        (ncrossed, crossing_points)

    end

end
