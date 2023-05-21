module PolySDF

    include("ray_tracing/ray_tracing.jl") # change as necessary
    include("simplex/simplex.jl")

    using AdaptiveKDTrees: KNN, RangeSearch

    using LinearAlgebra

    # using StaticArrays
    # using AdaptiveDistanceFields
    # using AdaptiveDistanceFields.RegionTrees

    export SDFTree # , adaptive_distance_field

    """
    ```
        struct SDFTree
            range_tree::RangeSearch.RangeTree
            nn_tree::KNN.KDTree
            surface::RayTracing.Surface
            points::AbstractMatrix
            simplices::AbstractMatrix

            function SDFTree(
                points::AbstractMatrix, # (ndims, npts)
                simplices::AbstractMatrix; # (ndims, nsimplices)
                leaf_size::Int = 10, # for search trees
                is_open::Bool = false, # if true, an open domain (distance is positive in the outside. Defs. to false)
            )

                # ...

            end
        end
    ```

    Struct to hold a signed distance function evaluation tree
    """
    struct SDFTree
        range_tree::RangeSearch.RangeTree
        nn_tree::KNN.KDTree
        surface::RayTracing.Surface
        points::AbstractMatrix
        simplices::AbstractMatrix

        function SDFTree(
            points::AbstractMatrix, # (ndims, npts)
            simplices::AbstractMatrix; # (ndims, nsimplices)
            leaf_size::Int = 10, # for search trees
            is_open::Bool = false, # if true, an open domain (distance is positive in the outside. Defs. to false)
        )

            simp_centers = mapslices(
                s -> vec(
                    sum(
                        points[:, s];
                        dims = 2,
                    )
                ) ./ length(s),
                simplices;
                dims = 1,
            )
            simp_maxdists = map(
                (s, c) -> maximum(
                    map(
                        norm,
                        eachcol(points[:, s] .- c)
                    )
                ),
                eachcol(simplices),
                eachcol(simp_centers)
            )

            range_tree = RangeSearch.RangeTree(
                simp_centers, simp_maxdists;
                leaf_size = leaf_size,
            )
            nn_tree = KNN.KDTree(
                points; leaf_size = leaf_size,
            )

            surface = RayTracing.Surface(
                points, simplices; 
                leaf_size = leaf_size,
                reference_isin = is_open,
            )

            new(
                range_tree,
                nn_tree,
                surface,
                copy(points),
                copy(simplices)
            )

        end
    end

    """
    ```
        SDFTree(points::AbstractMatrix; kwargs...)
    ```

    Alternative constructor for an SDFTree in two dimensions,
    which recieves a series of points in 2D space to be joined in a surface.

    Only works for two dimensions!!
    """
    function SDFTree(points::AbstractMatrix; kwargs...)

        @assert size(points, 1) == 2 "SDFTree constructor must receive simplex matrix for 3 or higher-dimensional spaces"

        simplices = let inds = collect(1:size(points, 2))
            permutedims(
                [
                        inds (circshift(inds, -1))
                ]
            )
        end

        SDFTree(points, simplices; kwargs...)

    end

    """
    ```
        function (sdft::SDFTree)(
            x::AbstractVector
        )
    ```

    Obtain signed distance to the surface of a triangulation, as represented by the SDF tree.

    Also returns the projection of the point upon the surface.
    """
    function (sdft::SDFTree)(
        x::AbstractVector
    )

        _, dmin = KNN.nn(sdft.nn_tree, x)

        possible_simps = RangeSearch.find_in_range(
            sdft.range_tree, x, dmin + eps(typeof(dmin))
        )

        simps = sdft.simplices[:, possible_simps]

        p, d = let pd = map(
            s -> Simplex.proj_and_dist(
                sdft.points[:, s],
                x;
                ϵ = eps(typeof(dmin))
            ),
            eachcol(simps)
        )
            _, ind = findmin(
                t -> t[2],
                pd
            )

            pd[ind]
        end

        if RayTracing.isin(sdft.surface, x)
            return (
                d, p
            )
        end

        (
            - d,
            p
        )
    
    end

    #=
    """
    ```
        function adaptive_distance_field(
            tree::SDFTree,
            origin::AbstractVector,
            widths::AbstractVector; # for containing hypercube
            center::Bool = false, # whether the origin is in the center of the domain
            atol::Real = 1e-2, # absolute tolerance
            rtol::Real = 1e-2, # relative tolerance
        )
    ```

    Generate adaptive distance field from an SDFTree.
    Results in an object from AdaptiveDistanceFields.jl.
    """
    function adaptive_distance_field(
        tree::SDFTree,
        origin::AbstractVector,
        widths::AbstractVector; # for containing hypercube
        center::Bool = false, # whether the origin is in the center of the domain
        atol::Real = 1e-2, # absolute tolerance
        rtol::Real = 1e-2, # relative tolerance
    )

        if center
            origin = origin .- widths ./ 2
        end

        origin = SVector(origin...)
        widths = SVector(widths...)

        AdaptiveDistanceField(
            x -> tree(x)[1],
            origin, widths,
            rtol, atol,
        )

    end

    """
    ```
        adaptive_distance_field(
            points::AbstractMatrix,
            simplices::AbstractMatrix,
            origin::AbstractVector,
            widths::AbstractVector; # for containing hypercube
            leaf_size::Int = 10, # for search trees
            is_open::Bool = false, # if true, an open domain (distance is positive in the outside. Defs. to false)
            center::Bool = false, # whether the origin is in the center of the domain
            atol::Real = 1e-2, # absolute tolerance
            rtol::Real = 1e-2, # relative tolerance
        ) = adaptive_distance_field(
            SDFTree(
                points, simplices; leaf_size = leaf_size, is_open = is_open,
            ),
            origin, widths;
            center = center,
            atol = atol,
            rtol = rtol,
        )
    ```

    Shortcut to create an adaptive distance field straight from points and simplices when the tree
    is not necessary
    """
    adaptive_distance_field(
        points::AbstractMatrix,
        simplices::AbstractMatrix,
        origin::AbstractVector,
        widths::AbstractVector; # for containing hypercube
        leaf_size::Int = 10, # for search trees
        is_open::Bool = false, # if true, an open domain (distance is positive in the outside. Defs. to false)
        center::Bool = false, # whether the origin is in the center of the domain
        atol::Real = 1e-2, # absolute tolerance
        rtol::Real = 1e-2, # relative tolerance
    ) = adaptive_distance_field(
        SDFTree(
            points, simplices; leaf_size = leaf_size, is_open = is_open,
        ),
        origin, widths;
        center = center,
        atol = atol,
        rtol = rtol,
    )
    =#

end
