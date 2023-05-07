# PolySDF.jl

A package for ray tracing/point-in-polygon queries and SDF/Approximate-SDF computation in Julia using N-dimensional triangulated surfaces.

## Ray Tracing

To build a surface for ray tracing, one may use:

```
using PolySDF: RayTracing

surf = RayTracing.Surface(
    points, # matrix (ndims, npts)
    simplices; # matrix (ndims, nsimps)
    leaf_size = 10,
    ray_reference = nothing,
    reference_isin = false,
    digits = 12
)
```

* `points` indicates surface points;
* `simplices` indicates simplices in the triangulated surface;
* `leaf_size` indicates a max. leaf size for the bounding box tree;
* `ray_reference` indicates a standard reference (origin) point for the ray tracing. Defaults to a point far away from the surface if `nothing`;
* `reference_isin`: whether said ray reference is within the surface;
* `digits`: number of digits for point merging precision. We advise the user not to change this.

For a point-in-polygon query, you can use `RayTracing.isin`:

```
flag = isin(
    surf, point
) # true if within solid
```

To optimize the queries, one may use a custom ray tracing origin closer to the query point as:

```
flag = isin(
    surf, point;
    origin = [-1, 2, 0], # example
    origin_isin = true # for reference
)
```

To find a list of intersected simplex indices and the corresponding intersection points, you may also use:

```
face_inds, int_points = crossed_faces(surf, p1, p2)

@show size(int_points)
# (ndims, npoints)
```

## SDFs and Approximate SDFs

For a signed distance function calculation, one may use:

```
using PolySDF

tree = SDFTree(
    points,
    simplices;
    leaf_size = 10, # for search trees
    is_open = false, # if true, an open domain (distance is positive in the outside. Defs. to false)
)

x = rand(size(points, 1)) # random point

dist, proj = tree(x)

# dist: distance to surface (signed)
# proj: projection upon surface
```

To optimize further, we also provide a function that, using AdaptiveDistanceFields.jl, produces an `AdaptiveDistanceField` object from the tree:

```
adf = adaptive_distance_field(
    tree,
    [0.0, 0.0, 0.0], # hypercube origin
    [1.0, 1.0, 1.0]; # hypercube widths
    center = false, # consider the origin is not in the center of the hypercube
    atol = 1e-2,
    rtol = 1e-2
)

dist = adf(x)

# or, alternatively, to obtain it straight from the triangulation:

adf = adaptive_distance_field(
    points, simplices,
    origin, widths
)
```