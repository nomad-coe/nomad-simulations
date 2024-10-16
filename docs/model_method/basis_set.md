# Basis Sets

The following lays down the schema annotation for several families of basis sets.
We start off genercially before running over specific examples.
The aim is not to introduce the full theory behind every basis set, but just enough to understand its main concepts and how they relate.

## General Structure

Basis sets are used by codes to represent various kinds of electronic structures, e.g. wavefunctions, densities, exchange densities, etc.
Each electronic structure is therefore described by an individual `BasisSetContainer` in the schema.

Basis sets may be partitioned by various regions, spanning either physical / reciprocal space, energy (i.e. core vs. valence), or potential / Hamiltonian.
We will cover the partitions per example below.
Each `BasisSetContainer` is then constructed out of several `basis_set_components` matching a single region.
Sometimes, a region is defined in terms of another schema section, e.g. an atom center (`species_scope`) or Hamiltonian terms (`hamiltonian_scope`).

Note that typically, different kinds of regions also have different mathematical formulations.
Each formulation has its own dedicated section, to facilitate their reuse.
These are all derived from the abstract section `BasisSetComponent`, so that `basis_set_components: list[BasisSetComponent]`.

Generically, `BasisSetComponent` will allude to the the formula at large and just focus on capturing the _subtype_, as well as relevant _parameters_.
The most relevant ones are those that most commonly listed in the Method section of an article.
These typically also influence the _precision_ most.
Extra, code-specific subtypes and parameters can be added by their respective parsers.

This then coalesces into the following diagram:

```
ModelMethod
└── NumericalSettings[0]
└── ...
└── NumericalSettings[n] = BasisSetContainer
                            └── BasisSetComponent[1]
                            └── ...
                            └── BasisSetComponent[n]
                                └──> AtomsState
                                └──> BaseModelMethod
```

## Plane-waves

Plane-wave basis sets start from the description of a free electron and use Fourier to construct the representations of bound electrons.
In reciprocal space, the basis set can thus be thought of as vectors in a Cartesian* grid enclosed within a sphere.

The main parameter is the spherical radius, i.e. the _cutoff_, which corresponds to the highest frequency representable Fourier frequency.
By convention, the radius is typically expressed in terms of the kinetic energy for the matching free-electron wave.
`PlaneWaveBasisSet` allows storing either `cutoff_radius` and `cutoff_energy`.
It can even derive the former from the latter via normalization.

### Pseudopotentials

Under construction...

## LAPW

The family of linearized augmented plane-waves is one of the best examples of region partitioning:

- first it partitions the physical space into regions surrounding the atomic nuclei, i.e. the _muffin-tin spheres_, and the rest, i.e. the _interstitial region_.
- it then further partitions the muffin tins by energy, i.e. core versus valence.
Note that unlike with pseudpotentials, the electrons are not abstracted away here.
They are instead explicitly accounted for and relaxed, just via a different representation.
Hence, LAPW is a _full-electron approach_.

The interstitial region, covering mostly loose bonding, is described by plane-waves (`APWPlaneWaveBasisSet`). [1]
The valence electrons in the muffin tin (`MuffinTinRegion`), meanwhile, are represented by the spherically symmetric Schrödigner equation. [1]
They follow the additional constraint of having to match the plane-wave description.
In that sense, where the plane-wave description becomes too expensive, it is "augmented" by the muffin-tin description.
This results in a lower plane-wave cutoff.

The spherically symmetric Schrödigner equation decomposes into an angular and radial part.
In traditional APW (not supported in NOMAD), the angular and radial part are coupled in a non-linear fashion via the radial energy (at the boundary).
All versions of LAPW simplify the coupling by parametrizing this radial energy. [1]

The representation vector is then developed in terms of the angular basis vectors, i.e. $l$-channels, each with their corresponding radial energy parameter.
This approach is -confusingly- also called _APW_.
It is typically not found standalone, though.
Instead, the linearization introduces a secondary representation via the first-order derivative of the basis vector (function).
Both vectors are typically developed together.
This technique is called linearized APW (LAPW). [1]

Other formulas have been experimented with too.
For example, the use of even higher-order derivatives, i.e. superlinearized APW (SLAPW). [2, 3]
All of these types are captured by `APWOrbital`, where `type` distinguishes between APW, LAPW, or SLAPW.
The `name` quantity

Another option is to stay with APW (or LAPW) and add standalone vectors targeting specific atomic states, e.g. high-energy core states, valence states, etc.
These are called _local orbitals_ (lo) and bear other constraints.
Some authors distinguish different vector sums with different kinds of local orbitals, e.g. lo, LO, high-dimensional LO (HDLO). [2, 4]
Since there is no community-wide consensus on the use of these abbreviations, we only utilize `lo` via `APWLocalOrbital`.

In summary, a generic LAPW basis set can thus be summarized as follows:

```
LAPW+lo
├── 1 x plane-wave basis set
└── n x muffin-tin regions
    └── l_max x l-channels
        ├── orbitals
        └── local orbitals ?
```

or in terms of the schema:

```
BasisSetContainer(name: LAPW+lo)
├── APWPlaneWaveBasisSet
├── MuffinTinRegion(atoms_state: atom A)
├── ...
└── MuffinTinRegion(atoms_state: atom N)
    ├── channel 0
    ├── ...
    └── channel l_max
        ├── APWOrbital(type: lapw)
        └── APWLocalOrbital ?
```

[1]: D. J. Singh and L. Nordström, \"INTRODUCTION TO THE LAPW METHOD,\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006.

[2]: A. Gulans, S. Kontur, et al., exciting: a full-potential all-electron package implementing density-functional theory and many-body perturbation theory, _J. Phys.: Condens. Matter_ **26** (363202), 2014. DOI: 10.1088/0953-8984/26/36/363202

[3]: J. VandeVondele, M. Krack, et al., WIEN2k: An APW+lo program for calculating the properties of solids, _J. Chem. Phys._ **152**(074101), 2020. DOI: 10.1063/1.5143061

[4]: D. Singh and H. Krakauer, H-point phonon in molybdenum: Superlinearized augmented-plane-wave calculations, _Phys. Rev. B_ **43**(1441), 1991. DOI: 10.1103/PhysRevB.43.1441

## Gaussian-Planewaves (GPW)

The CP2K code introduces an algorithm called QuickStep that partitions by Hamiltonian, describing

- the kinetic and Coulombic electron-nuclei interaction terms of a Gaussian-type orbital (GTO).
- the electronic Hartree energy via plane-waves.

This GPW choice is to increase performance. [1]
In the schema, we would write:

```
BasisSetContainer(name: GPW)
├── PlaneWaveBasisSet(hamiltonian_scope: [`/path/to/kinetic_term/hamiltonian`, `/path/to/e-n_term/hamiltonian`])
└── AtomCenteredBasisSet(name: GTO, hamiltonian_scope: [`/path/to/hartree_term/hamiltonian`])
```

For further details on the schema, see the CP2K parser documentation.

[1]: J. VandeVondele, M. Krack, et al., Quickstep: Fast and accurate density functional calculations using a mixed Gaussian and plane waves approach,
_Comp. Phys. Commun._ **167**(2), 103-128, 2005. DOI: 10.1016/j.cpc.2004.12.014.
