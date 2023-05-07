module Simplex

    using LinearAlgebra

    export simplex_volume, normal, simplex_faces, crosses_simplex, crossing_point, proj_and_dist

    """
    ```
        function simplex_volume(
            simplex::AbstractMatrix,
        )
    ```

    Get the volume of a simplex.

    If the passed matrix has fewer dimensions than expected (being a boundary face),
    its area is computed instead. In this case, it is always positive
    """
    function simplex_volume(
        simplex::AbstractMatrix,
    )

        p0 = simplex[:, 1]
        M = simplex[:, 2:end] .- p0

        if size(M, 1) != size(M, 2)
            return sqrt(det(M' * M)) / factorial(size(M, 2))
        end

        det(M) / factorial(length(p0))

    end

    """
    ```
        function normal(face::AbstractMatrix; normalize::Bool = false,)
    ```

    Get the normal of a face
    """
    function normal(face::AbstractMatrix; normalize::Bool = false,)

        p0 = face[:, 1]
        M = face[:, 2:end] .- p0

        N = length(p0)

        v = map(
            i -> let v = 1:N .== i
                det([M v])
            end,
            1:N,
        ) 

        v ./ (
            normalize ?
            norm(v) :
            factorial(N - 1)
        )

    end

    """
    ```
        simplex_faces(simplex::AbstractVector)
    ```

    Get faces from simplex (vector/index version)
    """
    simplex_faces(simplex::AbstractVector) = map(
        i -> let isnot = 1:length(simplex) .!= i
            simplex[isnot]
        end,
        1:length(simplex),
    )

    """
    ```
        simplex_faces(simplex::AbstractMatrix)
    ```

    Get faces from simplex (matrix/point version)
    """
    simplex_faces(simplex::AbstractMatrix) = map(
        i -> let isnot = 1:size(simplex, 2) .!= i
            simplex[:, isnot]
        end,
        1:size(simplex, 2),
    )


    """
    ```
        function crossing_point(
            simplex::AbstractMatrix, p1::AbstractVector, p2::AbstractVector,
        )
    ```

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
        function crosses_simplex(
            simplex::AbstractMatrix,
            p1::AbstractVector,
            p2::AbstractVector,
            ϵ::Real = 0.0,
        )
    ```

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
    Get projection upon simplex and find whether the point at hand is within the simplex
    """
    function proj_and_dist(
        simplex::AbstractMatrix,
        pt::AbstractVector;
        ϵ::Real = 0.0,
    )

        p0 = simplex[:, 1]
        p = pt .- p0

        if size(simplex, 2) == 1
            return (p0, norm(p))
        end

        M = simplex[:, 2:end] .- p0

        ξ = M \ p

        if any(
            x -> x < - ϵ,
            ξ
        ) || sum(ξ) > 1.0 + ϵ

            rets = map(
                f -> proj_and_dist(
                    f, pt; ϵ = ϵ,
                ),
                simplex_faces(simplex),
            )

            _, ind = findmin(
                r -> r[2],
                rets
            )

            return rets[ind]

        end

        proj = p0 .+ M * ξ

        (
            proj, norm(proj .- pt)
        )

    end

end

#=
@show Simplex.simplex_faces(
    [1, 2, 3, 4]
)
@show Simplex.simplex_faces(
    [
        0.0 1.0 0.0 0.0;
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0
    ]
)

@show Simplex.simplex_volume(
    [
        0.0 1.0 0.0;
        0.0 0.0 1.0;
        0.0 0.0 0.0
    ]
)
@show Simplex.simplex_volume(
    [
        0.0 1.0 0.0 0.0;
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0
    ]
)

@show Simplex.normal(
    [
        0.0 1.0 0.0;
        0.0 0.0 1.0;
        0.0 0.0 0.0
    ],
)
@show Simplex.normal(
    [
        0.0 1.0 0.0;
        0.0 0.0 1.0;
        0.0 0.0 0.0
    ]; normalize = true,
)

@show Simplex.proj_and_dist(
    [
        0.0 1.0 0.0;
        0.0 0.0 1.0;
        0.0 0.0 0.0
    ],
    [0.5, -1.0, 1.0]
)
=#
