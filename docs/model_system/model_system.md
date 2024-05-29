# `ModelSystem`

!!! warning
    This page is still under construction.

## Distributions of geometric properties

### Schema structure
These distributions are stored under `AtomicCell`, with different specializations as repeating subsections of `GeometryDistribution` or its child classes.
This separation simplifies filtering to the `results` section and helps with indexing, as bin distributions tend to follow the default.
Other information, such as cutoff distances used to construct the neighbor list, are also stored here.

`GeometryDistribution` objects are effectively histograms that specialize further in sections for elemental pairs / triples / quadruples.
This is a choice to facilitate searches and visualizations over the distribution themselves, which are normalized to reproduce the same frequency for primitive cells as supercells.
Additional advantages include a limited storage consumption.
It is, however, not suitable for extracting the exacting distances / angles / dihedral angles by a given value.

!!! note
    Triples / quadruples are defined around a specific geometric primitive, i.e. a point / arrow vector.
    The elements making up these primitives are stored under `central_atom_labels`.

### Object initialization
To keep state within `AtomicCell` simple and exert control over the stages in calculation, we use pure Python helper classes.
Each class tackles a specific stage, in line with the _Single Responsibility Principle_:

    - `DistributionFactory` scans the combinatorial space of elements and generates their `Distributions`.
    - `Distribution` leverages `ase` for computing the distances / (dihedral) angles for the elemental combo provided.
    It can also instantiate a `DistributionHistogram` of itself.
    - `DistributionHistogram` contains the histogram version of their distribution.

While only `DistributionHistogram` will eventually be written out to `GeometryDistribution`, the rest of the data artifacts are retained under `AtomicCell` during run-time.
As such, once computed, they can be used in other computation, or data analysis.

!!! info
    Conversion to `pandas.DataFrame` is a planned feature.

Inclusion of the actual computation pipeline, meanwhile, is toggled via the boolean `AtomicCell.analyze_geometry`.
The execution itself is handled by the normalizer.