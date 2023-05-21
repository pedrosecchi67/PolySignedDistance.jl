begin

    using .RayTracing
    using LinearAlgebra

    points = [
        0.0 1.0 1.0 0.0;
        0.0 0.0 1.0 1.0
    ] .* 0.5
    #=
    simplices = [
        1 2 3 4;
        2 3 4 1
    ]
    =#

    surf = Surface(points) # , simplices) # now the two-dimensional version!

    @assert !RayTracing.crosses_simplex([0.5 0.0; 0.5 0.5], [-0.1, 0.2], [1.1, 0.2])

    @assert isin(surf, [0.2, 0.2])
    @assert !isin(surf, [1.2, 0.2])
    @assert !isin(surf, [0.25, 0.60])

    inds, pts = crossed_faces(surf, [-0.2, 0.2], [1.2, 0.2])
    @assert Set(inds) == Set([2, 4])

    @info "Performance test"

    θ = collect(LinRange(0.0, 2 * π, 10000))[1:(end - 1)]

    points = [
        cos.(θ)';
        sin.(θ)'
    ]
    #=
    simplices = let i = collect(1:(length(θ) - 1))
        [
            i';
            (i .+ 1)'
        ]
    end
    =#

    surf = Surface(points; digits = 7,)

    for nit = 1:10
        x = randn(2)

        @time isin(surf, x)
    end

    @info "Performance test without intersection"

    for nit = 1:10
        x = randn(2) .* 0.01 .+ 0.9

        @time isin(surf, x)
    end

    pts = rand(2, 300) .* 2 .- 1.0
    pts = [pts zeros(2)]

    @assert isin(surf, zeros(2))

    isincirc = map(
        pt -> isin(surf, pt),
        eachcol(pts)
    )
    exact_isincirc = map(
        pt -> norm(pt) < 1,
        eachcol(pts)
    )

    isval = map(
        pt -> abs(norm(pt) - 1) > 1e-2,
        eachcol(pts)
    )

    @assert isincirc[isval] ≈ exact_isincirc[isval]

end
