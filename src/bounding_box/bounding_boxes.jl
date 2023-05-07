module BoundingBox

    using LinearAlgebra

    using Statistics

    export Box, find_potential_intersections

    """
    ```
        mutable struct Box
    ```

    Struct defining a Box
    """
    mutable struct Box
        boxes::Union{Vector{Box}, Nothing}
        indices::Union{AbstractVector, Nothing} # of indices
        center::AbstractVector
        width::AbstractVector
        sub_boxes::AbstractVector # vector of boxes
    end

    """
    ```
        function Box(
            points::AbstractMatrix
        )
    ```

    Create a bounding box encapsulating a set of points (matrix, `(ndims, npoints)`)
    """
    function Box(
        points::AbstractMatrix
    )

        axmin = map(
            minimum,
            eachrow(points)
        )
        axmax = map(
            maximum,
            eachrow(points)
        )

        center = @. (axmin + axmax) / 2
        width = @. (axmax - axmin)

        Box(nothing, nothing, center, width, [])

    end

    """
    ```
        function Box(
            boxes::Vector{Box}, indices = nothing;
            max_size::Int = 10,
        )
    ```

    Create a bounding box tree encapsulating a set of
    elementary bounding boxes and defining them hierarchically in space.

    If indices aren't provided, the boxes are numbered as in the input vector.

    `max_size` specifies a leaf maximum size for queries
    """
    function Box(
        boxes::Vector{Box}, indices = nothing;
        max_size::Int = 10,
    )

        indices = (
            isnothing(indices) ?
            collect(1:length(boxes)) :
            indices
        )

        ndim = length(boxes[1].center)

        pmin = map(
            i -> minimum(
                bb -> bb.center[i] - bb.width[i] / 2,
                boxes
            ),
            1:ndim
        )
        pmax = map(
            i -> maximum(
                bb -> bb.center[i] + bb.width[i] / 2,
                boxes
            ),
            1:ndim
        )

        center = @. (pmin + pmax) / 2
        width = pmax .- pmin

        b = Box(
            boxes, indices, center, width, []
        )

        split!(b, max_size)

        b

    end

    # split box in two sub-boxes by the median of the most broadly variating dimension of the 
    # sub-box centers
    function split!(
        b::Box, max_size::Int,
    )

        if length(b.boxes) <= max_size
            return
        end

        points = mapreduce(
            bb -> bb.center,
            hcat,
            b.boxes,
        )

        _, dim = findmax(
            r -> maximum(r) - minimum(r),
            eachrow(points)
        )

        r = @view points[dim, :]

        med = median(r)

        isleft = @. r <= med
        isright = @. !isleft

        if all(isleft) || all(isright)
            return
        end

        boxes = b.boxes
        b.boxes = nothing

        indices = b.indices
        b.indices = nothing

        b.sub_boxes = [
            Box(
                boxes[isleft], indices[isleft];
                max_size = max_size,
            ),
            Box(
                boxes[isright], indices[isright];
                max_size = max_size,
            ),
        ]; # return nothing

    end

    # minimum distance between boxes
    mindist(b1::Box, b2::Box) = norm(
        (
            @. max(
                0.0,
                abs(b1.center - b2.center) - abs(b1.width + b2.width) / 2
            )
        )
    )

    # center of the hypothetical box encapsulating a line
    function line_data(line::AbstractMatrix)

        dmin = map(minimum, eachrow(line))
        dmax = map(maximum, eachrow(line))

        center = @. (dmin + dmax) / 2
        width = @. (dmax - dmin)

        (center, width)

    end

    # projection of point upon line
    function line_projection(line::AbstractMatrix, point::AbstractVector)

        p1 = line[:, 1]
        p2 = line[:, 2]

        v = p2 .- p1

        ksi = v \ (point .- p1)

        p1 .+ v .* ksi

    end

    # minimum distance estimate between box and line
    mindist(b1::Box, line::AbstractMatrix) = let (center, width) = line_data(line) 

        proj = line_projection(line, b1.center)
        R = norm(b1.width) / 2

        max(
            norm(
                (
                    @. max(
                        0.0,
                        abs(b1.center - center) - abs(b1.width + width) / 2
                    )
                )
            ),
            norm(b1.center .- proj) - R,
            0.0,
        )

    end

    # is the box a leaf?
    isleaf(b::Box) = (length(b.sub_boxes) == 0)

    # I'm not sure I'm even using this but I'm now afraid to erase it
    split_line(
        btarg::AbstractMatrix,
    ) = let (p1, p2) = (btarg[:, 1], btarg[:, 2])
        mid = @. (p1 + p2) / 2

        (
            [p1 mid],
            [mid p2],
        )
    end

    #=
    function simplify_closest(
        b::Box,
        btarg::AbstractMatrix,
    )

        if size(btarg, 2) != 2
            return (false, btarg)
        end

        p1 = btarg[:, 1]
        p2 = btarg[:, 2]

        mid = @. (p1 + p2) / 2

        t1 = [p1 mid]
        t2 = [mid p2]

        epsilon = eps(eltype(btarg))

        i1 = mindist(b, t1) <= epsilon
        i2 = mindist(b, t2) <= epsilon

        if i1
            if i2
                return (false, btarg)
            else
                return (true, t1)
            end
        end

        if i2
            return (true, t2)
        end

        (false, btarg) # will conclude the iteration anyway

    end

    simplify_closest(
        b::Box,
        btarg::Box,
    ) = (false, btarg)
    =#

    # visit a box and store its indices if potentially within range
    function visit(
        b::Box,
        btarg, # ::Box,
        inds::AbstractVector = Int64[],
    )

        epsilon = eps(eltype(b.center))

        db = mindist(b, btarg)

        if db > epsilon
            return inds
        end

        if isleaf(b)
            if db <= epsilon
                inds = [inds; b.indices]
            end

            return inds
        end

        for sb in b.sub_boxes
            inds = visit(sb, btarg, inds)
        end

        inds

    end

    #=
    function visit(
        b::Box,
        btarg::AbstractMatrix,
        inds::AbstractVector = Int64[],
    )

        epsilon = eps(eltype(b.center))

        db = mindist(b, btarg)

        if db > epsilon
            return inds
        end

        if isleaf(b)
            if db <= epsilon
                inds = [inds; b.indices]
            end

            return inds
        end

        l1, l2 = split_line(btarg)

        for sb in b.sub_boxes
            inds = visit(sb, l1, inds)
            inds = visit(sb, l2, inds)
        end

        inds

    end
    =#

    """
    ```
        find_potential_intersections(
            b::Box,
            btarg,
        )
    ```

    Find a vector with indices of bounding boxes that potentially intersect with object `btarg`.

    `btarg` may be:

    * A line (matrix shape `(ndims, 2)`); or
    * Another bounding box.
    """
    function find_potential_intersections(
        b::Box,
        btarg,
    )

        if isnothing(b.indices) && length(b.sub_boxes) == 0
            throw(
                error(
                    "The provided bounding box is not hierarchically defined. See constructors for Box for more information"
                )
            )
        end

        if isa(btarg, AbstractMatrix)
            if size(btarg, 2) != 2
                btarg = Box(btarg)
            end
        end

        visit(b, btarg)

    end

end

#=
using .BoundingBox

θs = collect(LinRange(0.0, 2 * π, 100000))

simplices = [
    begin
        [
            cos(θ) cos(θp1);
            sin(θ) sin(θp1)
        ]
    end for (θ, θp1) in zip(
        θs[1:(end - 1)], θs[2:end]
    )
]

boxes = map(
    Box, 
    simplices
)

b = Box(boxes)
=#

#=
@time b = Box(boxes)
@time b = Box(boxes)
@time b = Box(boxes)
@time b = Box(boxes)

using ProfileView

@profview b = Box(boxes)
@profview b = Box(boxes)
=#

#=
h = 1.0 / length(simplices)
L = 100.0

line = [
    (L + h) (- L - h);
    (L + h) (- L - h)
]

pot = find_potential_intersections(b, line)

@time pot = find_potential_intersections(b, line)
@time pot = find_potential_intersections(b, line)
@time pot = find_potential_intersections(b, line)
@time pot = find_potential_intersections(b, line)

@show length(pot)
@show simplices[pot]

using ProfileView

@profview for i = 1:100
    pot = find_potential_intersections(b, line)
end
@profview for i = 1:100
    pot = find_potential_intersections(b, line)
end
@profview for i = 1:100
    pot = find_potential_intersections(b, line)
end
=#
