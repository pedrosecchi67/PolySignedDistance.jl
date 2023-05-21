begin
    using .PolySDF
    using LinearAlgebra

    θ = collect(LinRange(0.0, 2 * π, 10000))[1:(end - 1)]

    points = [
        cos.(θ)';
        sin.(θ)'
    ]
    simplices = let i = collect(1:length(θ))
        [
            i';
            (circshift(i, -1))'
        ]
    end

    tree = SDFTree(points) # , simplices)

    @info "Tree creation - $(length(θ)) simplices on circumpherence"

    @time tree = SDFTree(points) # , simplices)
    @time tree = SDFTree(points) # , simplices)
    @time tree = SDFTree(points) # , simplices)
    @time tree = SDFTree(points) # , simplices)

    X = rand(2, 10)
    h = 0.01

    @info "Performance test - single point query"

    for x in eachcol(X)

        @time d, p = tree(x)

        ad = 1.0 - norm(x)

        if abs(d) > h
            @assert abs(d - ad) < h
        end

    end

    origin = fill(-2.0, 2)
    widths = fill(4.0, 2)

    tols = (
        rtol = 0.2,
        atol = 1e-6,
    )

    @info "Adaptive distance field with $tols"

    @time adf = adaptive_distance_field(
        tree,
        origin, widths;
        tols...
    )
    @time adf = adaptive_distance_field(
        tree,
        origin, widths;
        tols...
    )
    @time adf = adaptive_distance_field(
        tree,
        origin, widths;
        tols...
    )
    @time adf = adaptive_distance_field(
        tree,
        origin, widths;
        tols...
    )

    @info "Adaptive distance field with $tols - shortcut"

    @time adf = adaptive_distance_field(
        points, simplices,
        origin, widths;
        tols...
    )
    @time adf = adaptive_distance_field(
        points, simplices,
        origin, widths;
        tols...
    )
    @time adf = adaptive_distance_field(
        points, simplices,
        origin, widths;
        tols...
    )
    @time adf = adaptive_distance_field(
        points, simplices,
        origin, widths;
        tols...
    )

    _test_in_tol = (a, b; atol = 1e-2, rtol = 1e-2,) -> abs(
        a - b
    ) < atol + rtol * max(abs(a), abs(b))

    @info "Testing ASDF values"

    for x in eachcol(X)

        @time d, p = tree(x)

        ad = adf(x)

        if abs(d) > h
            @assert _test_in_tol(
                ad, d;
                tols...
            )
        end

    end
end
