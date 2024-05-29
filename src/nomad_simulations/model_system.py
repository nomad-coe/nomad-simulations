#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import ase.geometry
import numpy as np
import ase
from ase import neighborlist as ase_nl
from ase.geometry import analysis as ase_as
from typing import Optional, Callable
from structlog.stdlib import BoundLogger
import pint

from matid import SymmetryAnalyzer, Classifier  # pylint: disable=import-error
from matid.classification.classifications import (
    Class0D,
    Atom,
    Class1D,
    Class2D,
    Material2D,
    Surface,
    Class3D,
)  # pylint: disable=import-error

from nomad import config
from nomad.units import ureg
from nomad.atomutils import Formula, get_normalized_wyckoff, search_aflow_prototype

from nomad.metainfo import Quantity, SubSection, SectionProxy, MEnum, Section, Context
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.basesections import Entity, System
from nomad.datamodel.metainfo.annotations import ELNAnnotation

from .atoms_state import AtomsState
from .utils import get_sibling_section, is_not_representative


class GeometricSpace(Entity):
    """
    A base section used to define geometrical spaces and their entities.
    """  # TODO: allow input in of the lattice vectors directly

    length_vector_a = Quantity(
        type=np.float64,
        unit='meter',
        description="""
        Length of the first basis vector.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    length_vector_b = Quantity(
        type=np.float64,
        unit='meter',
        description="""
        Length of the second basis vector.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    length_vector_c = Quantity(
        type=np.float64,
        unit='meter',
        description="""
        Length of the third basis vector.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    angle_vectors_b_c = Quantity(
        type=np.float64,
        unit='radian',
        description="""
        Angle between second and third basis vector.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    angle_vectors_a_c = Quantity(
        type=np.float64,
        unit='radian',
        description="""
        Angle between first and third basis vector.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    angle_vectors_a_b = Quantity(
        type=np.float64,
        unit='radian',
        description="""
        Angle between first and second basis vector.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    volume = Quantity(
        type=np.float64,
        unit='meter ** 3',
        description="""
        Volume of a 3D real space entity.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    surface_area = Quantity(
        type=np.float64,
        unit='meter ** 2',
        description="""
        Surface area of a 3D real space entity.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    area = Quantity(
        type=np.float64,
        unit='meter ** 2',
        description="""
        Area of a 2D real space entity.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    length = Quantity(
        type=np.float64,
        unit='meter',
        description="""
        Total length of a 1D real space entity.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    coordinates_system = Quantity(
        type=MEnum('cartesian', 'polar', 'cylindrical', 'spherical'),
        default='cartesian',
        description="""
        Coordinate system used to determine the geometrical information of a shape in real
        space. Default to 'cartesian'.
        """,
    )

    origin_shift = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Vector `p` from the origin of a custom coordinates system to the origin of the
        global coordinates system. Together with the matrix `P` (stored in transformation_matrix),
        the transformation between the custom coordinates `x` and global coordinates `X` is then
        given by:
            `x` = `P` `X` + `p`.
        """,
    )

    transformation_matrix = Quantity(
        type=np.float64,
        shape=[3, 3],
        description="""
        Matrix `P` used to transform the custom coordinates system to the global coordinates system.
        Together with the vector `p` (stored in origin_shift), the transformation between
        the custom coordinates `x` and global coordinates `X` is then given by:
            `x` = `P` `X` + `p`.
        """,
    )

    def get_geometric_space_for_atomic_cell(self, logger: BoundLogger) -> None:
        """
        Get the real space parameters for the atomic cell using ASE.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        atoms = self.to_ase_atoms(logger)  # function defined in AtomicCell
        cell = atoms.get_cell()
        self.length_vector_a, self.length_vector_b, self.length_vector_c = (
            cell.lengths() * ureg.angstrom
        )
        self.angle_vectors_b_c, self.angle_vectors_a_c, self.angle_vectors_a_b = (
            cell.angles() * ureg.radians
        )
        self.volume = cell.volume * ureg.angstrom**3

    def normalize(self, archive: ArchiveSection, logger: BoundLogger) -> None:
        # Skip normalization for `Entity`
        try:
            self.get_geometric_space_for_atomic_cell(logger)
        except Exception:
            logger.warning(
                'Could not extract the geometric space information from ASE Atoms object.',
            )
            return


class Cell(GeometricSpace):
    """
    A base section used to specify the cell quantities of a system at a given moment in time.
    """

    name = Quantity(
        type=str,
        description="""
        Name of the specific cell section. This is typically used to easy identification of the
        `Cell` section. Possible values: "AtomicCell".
        """,
    )

    type = Quantity(
        type=MEnum('original', 'primitive', 'conventional'),
        description="""
        Representation type of the cell structure. It might be:
            - 'original' as in originally parsed,
            - 'primitive' as the primitive unit cell,
            - 'conventional' as the conventional cell used for referencing.
        """,
    )

    n_objects = Quantity(
        type=np.int32,
        description="""
        Number of cell points.
        """,
    )

    # geometry and analysis
    positions = Quantity(
        type=np.float64,
        shape=['n_cell_points', 3],
        unit='meter',
        description="""
        Positions of all the atoms in absolute, Cartesian coordinates.
        """,
    )

    velocities = Quantity(
        type=np.float64,
        shape=['n_cell_points', 3],
        unit='meter / second',
        description="""
        Velocities of the atoms defined with respect to absolute, Cartesian coordinate frame
        centered at the respective atom position.
        """,
    )

    # box
    lattice_vectors = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='meter',
        description="""
        Lattice vectors of the simulated cell in Cartesian coordinates. The first index runs
        over each lattice vector. The second index runs over the $x, y, z$ Cartesian coordinates.
        """,
    )

    periodic_boundary_conditions = Quantity(
        type=bool,
        shape=[3],
        default=[False, False, False],  # when is the default value set?
        description="""
        If periodic boundary conditions are applied to each direction of the crystal axes.
        """,
    )

    supercell_matrix = Quantity(
        type=np.int32,
        shape=[3, 3],
        description="""
        Specifies the matrix that transforms the primitive unit cell into the supercell in
        which the actual calculation is performed. In the easiest example, it is a diagonal
        matrix whose elements multiply the lattice_vectors, e.g., [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        is a $3 x 3 x 3$ superlattice.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class GeometryDistribution(ArchiveSection):
    """
    A base section used to specify the geometry distributions of the system.
    These include frequency counts of inter-atomic distances and angles.
    """

    name = Quantity(
        type=str,
        description="""
        Elements of which the geometry distribution is described.
        The name is a hyphen-separated and sorted alphabetically.
        """,
    )

    type = Quantity(
        type=str,
        description="""
        Type of the geometry distribution described.
        """,
    )

    extremity_atom_labels = Quantity(
        type=str,
        shape=[2],
        description="""
        Elements for which the geometry distribution is defined.
        (Note: the order of the elements is not algorithmically enforced.)
        """,
    )  # ? should we enforce this algorithmically. this also has implications for the distribution

    n_cutoff_specifications = Quantity(
        type=np.int32,
        description="""
        Number of cutoff specifications used to analyze the geometry.
        """,
    )

    element_cutoff_selection = Quantity(
        type=str,
        shape=['n_cutoff_specifications'],
        description="""
        Elements for which cutoffs are defined. The order of the elements corresponds to those in `distance_cutoffs`.
        """,
    )

    distance_cutoffs = Quantity(
        type=np.float64,
        unit='meter',
        shape=['n_cutoff_specifications'],
        description="""
        Cutoff radii within which atomic geometries are analyzed. This incidentally defines the upper-bound for distances.
        Defaults to 3x the covalent radius denoted by the [`ase` package](https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html).
        """,
    )

    bins = Quantity(
        type=np.float64,
        shape=['n_bins'],
        description="""
        Binning used to create the histogram.
        """,
    )

    n_bins = Quantity(
        type=np.int32,
        description="""
        Number of bins used to create the histogram.
        """,
    )

    frequency = Quantity(
        type=np.float64,
        shape=['n_bins'],
        description="""
        Frequency count of the histogram.
        """,
    )

    def get_bins(self) -> np.ndarray:
        """Return the distribution bins in their SI units.
        Fall back to a default value if none are present."""
        return self.bins

    def sort_cutoffs(self, sort_key: Callable = lambda x: ase.Atom(x).number):
        """Sort the cutoffs and the elements accordingly.
        Args:
            sort_key (Callable): A function to sort the elements. Defaults to the atomic number.
        Returns:
            GeometryDistribution: The sorted geometry distribution object itself.
        """
        sorted_indices = np.argsort(
            [sort_key(elem) for elem in self.element_cutoff_selection]
        )  # get the indices of the atomic numbers
        self.distance_cutoffs = self.distance_cutoffs[sorted_indices]
        self.element_cutoff_selection = [
            self.element_cutoff_selection[i] for i in sorted_indices
        ]
        return self

    def normalize(self, archive, logger: BoundLogger):
        super().normalize(archive, logger)
        return self


class DistanceGeometryDistribution(GeometryDistribution):
    bins = GeometryDistribution.bins.m_copy()
    bins.unit = 'meter'

    def normalize(self, archive, logger: BoundLogger):
        super().normalize(archive, logger)
        self.name = '-'.join(self.extremity_atom_labels)
        self.type = 'distances'
        return self


class AngleGeometryDistribution(GeometryDistribution):
    name = GeometryDistribution.name.m_copy()
    name.description += """
    The center element is placed at the center of the name.
    """

    central_atom_labels = Quantity(
        type=str,
        shape=[1],
        description="""
        Element at the center of the angle.
        """,
    )

    bins = GeometryDistribution.bins.m_copy()
    bins.unit = 'radians'

    def get_bins(self) -> np.ndarray:
        """Return the distribution bins in their SI units, i.e. radians.
        Fall back to a default value, i.e. [0 ... 0.01 ... 180] degrees, if none are present."""
        if not self.bins:
            return (np.arange(0, 180, 0.01) * ureg.degrees).to('radians')
        return self.bins

    def normalize(self, archive, logger: BoundLogger):
        super().normalize(archive, logger)
        self.name = '-'.join(
            [self.extremity_atom_labels[0]]
            + self.central_atom_labels
            + [self.extremity_atom_labels[1]]
        )
        self.type = 'angles'
        return self


class DihedralGeometryDistribution(GeometryDistribution):
    name = (
        GeometryDistribution.name.m_copy()
    )  # ? forego storage of constant values for @property
    name.description += """
    The central axis is placed in the middle, with the same ordering.
    """

    central_atom_labels = Quantity(
        type=str,
        shape=[2],
        description="""
        Elements along which the axis of the dihedral angle is aligned.
        The direction vector runs from element 1 to 2.
        (Note: the order of the elements is not algorithmically enforced.)
        """,
    )  # ? should we enforce this algorithmically. this also has implications for the distribution

    bins = GeometryDistribution.bins.m_copy()
    bins.unit = 'radians'

    def get_bins(self) -> np.ndarray:
        """Return the distribution bins in their SI units, i.e. radians.
        Fall back to a default value, i.e. [0 ... 0.01 ... 180] degrees, if none are present."""
        if not self.bins:
            return (np.arange(0, 180, 0.01) * ureg.degrees).to('radians')
        return self.bins

    def normalize(self, archive, logger: BoundLogger):
        super().normalize(archive, logger)
        self.name = '-'.join(
            [self.extremity_atom_labels[0]]
            + self.central_atom_labels
            + [self.extremity_atom_labels[1]]
        )
        self.type = 'dihedrals'
        return self


class DistributionHistogram:
    """
    A conversion class to compute the histogram of a distribution.
    It can also store the histogram in a NOMAD `GeometryDistribution` section via `produce_nomad_distribution`.
    """

    def __init__(
        self,
        combo: list[str],
        distribution_values: pint.Quantity,
        bins: pint.Quantity,
        neighbor_list: pint.Quantity,
    ) -> None:
        """The `combo` is a list of element symbols, `distribution_values` is a list of distances or angles,
        `bins` is the binning of the histogram and sets the final units,
        and `neighbor_list` is the cutoffs for the distribution (in the same order as combo)."""
        self.combo = combo
        self.n_elem = len(combo)
        self.bins = bins
        self.neighbor_list = neighbor_list
        self.compute_histogram(distribution_values)

    def compute_histogram(self, distribution_values: pint.Quantity) -> None:
        """Compute the histogram of the distribution values,
        based on `self.bins` and `distribution_values`.
        Results are stored under `self.counts`. Returns the object itself."""

        # compute the histogram
        self.counts, _ = np.histogram(
            distribution_values.to(self.bins.u).magnitude, bins=self.bins.magnitude
        )  # have the bins determine the units and range

        # normalize so the minimum is 1
        if len(nonzero_counts := self.counts[np.where(self.counts > 0)]) > 0:
            self.frequency = self.counts / np.min(nonzero_counts)
        else:
            self.frequency = self.counts

    def produce_nomad_distribution(self) -> GeometryDistribution:
        # select the constructor based on the type
        constructor_mapping = {
            2: DistanceGeometryDistribution,
            3: AngleGeometryDistribution,
            3: DihedralGeometryDistribution,
        }
        if self.n_elem in constructor_mapping:
            constructor = constructor_mapping[self.n_elem]
        else:
            raise ValueError('Cannot assign a constructor based on `self.combos`.')

        # instantiate the NOMAD distribution
        geom_dist = constructor(
            extremity_atom_labels=[self.combo[0], self.combo[-1]],
            frequency=self.frequency,
            n_cutoff_specifications=len(self.neighbor_list),
            element_cutoff_selection=self.combo,  # this is enforced by `DistributionFactory`
            distance_cutoffs=self.neighbor_list,
        )

        # distances require explicit binning data
        if self.n_elem == 2:
            geom_dist.n_bins = len(self.bins)
            geom_dist.bins = self.bins

        # add the central atom labels if they exist
        if self.n_elem > 2:
            geom_dist.central_atom_labels = self.combo[1:-1]
        return geom_dist


class Distribution:
    """
    A helper class to compute the distribution of a given geometry type, i.e., distances or (dihedral) angles.
    Also generates a histogram class via `produce_histogram`.
    """

    def __init__(
        self,
        combo: list[str],
        ase_atoms: ase.Atoms,
        neighbor_list: pint.Quantity,
    ) -> None:
        # initialize object attributes
        self.combo = combo
        self.n_elem = len(combo)
        self.neighbor_list = neighbor_list

        geom_analysis = ase_as.Analysis(
            ase_atoms, neighbor_list.to('angstrom').magnitude
        )
        if self.n_elem == 2:  # distances
            if len((matches := geom_analysis.get_bonds(*combo, unique=True))[0]) > 0:
                self.values = geom_analysis.get_values(matches)
            else:
                self.values = np.array([])
            self.values *= ureg.angstrom
        elif self.n_elem in (3, 4):  # (dihedral) angles
            if len((matches := geom_analysis.get_angles(*combo, unique=True))[0]) > 0:
                self.values = np.array(geom_analysis.get_values(matches)) * ureg.degrees
                self.values = np.abs(
                    np.where(
                        self.values > 180 * ureg.degrees,
                        360 * ureg.degrees - self.values,
                        self.values,
                    )
                )  # restrict to [0, 180] degrees
            else:
                self.values = np.array([])
            self.values *= ureg.degrees
        else:
            raise ValueError('Could not infer distribution type from `combo`.')

    def produce_histogram(self, bins: pint.Quantity) -> DistributionHistogram:
        return DistributionHistogram(self.combo, self.values, bins, self.neighbor_list)


class DistributionFactory:
    """
    Factory class for exhaustively generating distributions of geometrical observables
    for all element paris or triples. It will produce a list of `Distribution` objects via `produce_distributions`.
    """

    def __init__(self, ase_atoms: ase.Atoms, neighbor_list: pint.Quantity) -> None:
        """The neighbor list should follow the same order as `ase_atoms`."""
        if len(ase_atoms) != len(neighbor_list):
            raise ValueError('Length of `ase_atoms` and `neighbor_list` do not match.')

        self.ase_atoms = ase_atoms
        self.cutoff_mapping = dict(zip(ase_atoms.get_chemical_symbols(), neighbor_list))
        self._elements = sorted(
            list(set(self.ase_atoms.get_chemical_symbols())),
            key=lambda x: ase.Atom(x).number,
        )  # sort by atomic number

    @property
    def get_elemental_pairs(self) -> list[tuple[str, str]]:
        """Generate all possible pairs of element symbols.
        Permutations are not considered, i.e., (B, A) is converted into (A, B)."""
        return [
            (atom_1, atom_2)
            for i, atom_1 in enumerate(self._elements)
            for atom_2 in self._elements[i:]
        ]

    @property
    def get_elemental_triples_centered(self) -> list[tuple[str, str, str]]:
        """Generate all possible triples of element symbols with the center element first.
        Permutations between the outer elements are not considered, i.e., (A, C, B) is converted into (A, B, C).
        """
        return [
            (atom_1, atom_c, atom_2)
            for atom_c in self._elements
            for i, atom_1 in enumerate(self._elements)
            for atom_2 in self._elements[i:]
        ]  # matching the order of the `ase` package: https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/analysis.html#Analysis.get_angles

    @property
    def get_elemental_quadruples_centered(self) -> list[tuple[str, str, str, str]]:
        """Generate all possible quadruples of element symbols with the center elements first.
        Permutations between the outer elements are not considered, i.e., (A, D, B, C) is converted into (A, B, C, D).
        """
        return [
            (atom_1, atom_c, atom_d, atom_2)
            for atom_c in self._elements
            for atom_d in self._elements
            for i, atom_1 in enumerate(self._elements)
            for atom_2 in self._elements[i:]
        ]  # matching the order of the `ase` package: https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/analysis.html#Analysis.get_dihedrals

    def produce_distributions(self) -> list[Distribution]:
        """Generate all possible distributions of distances and angles
        for all element pairs and triples, respectively."""
        return [
            Distribution(
                list(combos),
                self.ase_atoms,
                [self.cutoff_mapping[atom] for atom in combos],
            )
            for combos in [
                *self.get_elemental_pairs,
                *self.get_elemental_triples_centered,
            ]
        ]


class AtomicCell(Cell):
    """
    A base section used to specify the atomic cell information of a system.
    """

    atoms_states = SubSection(
        sub_section=AtomsState.m_def,
        repeats=True,
    )

    # geometry distributions
    ## I separated the distributions into different categories
    ## to facilitate indexing when copied over to `results` section
    distance_distributions = SubSection(
        sub_section=DistanceGeometryDistribution.m_def,
        repeats=True,
    )

    angle_distributions = SubSection(
        sub_section=AngleGeometryDistribution.m_def,
        repeats=True,
    )

    geometry_distributions = SubSection(
        sub_section=GeometryDistribution.m_def,
        repeats=True,
    )

    # symmetry
    equivalent_atoms = Quantity(
        type=np.int32,
        shape=['n_objects'],
        description="""
        List of equivalent atoms as defined in `atoms`. If no equivalent atoms are found,
        then the list is simply the index of each element, e.g.:
            - [0, 1, 2, 3] all four atoms are non-equivalent.
            - [0, 0, 0, 3] three equivalent atoms and one non-equivalent.
        """,
    )

    # ! improve description and clarify whether this belongs to `Symmetry` with @lauri-codes
    wyckoff_letters = Quantity(
        type=str,
        shape=['n_objects'],
        description="""
        Wyckoff letters associated with each atom.
        """,
    )

    # topology
    bond_list = Quantity(
        type=np.int32,
        description="""
        List of index pairs corresponding to (covalent) bonds, e.g., as defined by a force field.
        """,
    )

    def to_ase_atoms(self, logger: BoundLogger) -> Optional[ase.Atoms]:
        """
        Generate an `ase.Atoms` object from `AtomicCell`.
        """
        # check all prerequisites
        attributes = ('atoms_states', 'lattice_vectors', 'positions')
        for attribute in attributes:
            if getattr(self, attribute) is None:
                logger.warning(f'Attribute {attribute} not found in AtomicCell.')
                return None

        if len(self.atoms_states) != len(positions := self.positions):
            logger.warning(
                'Number of atoms states and positions do not match.',
                atoms_states=len(self.atoms_states),
                positions=len(positions),
            )
            return None

        # if all is well, generate the `ase.Atoms` object
        ase_atoms = ase.Atoms(
            symbols=[atom_state.chemical_symbol for atom_state in self.atoms_states],
            positions=self.positions.to('angstrom').magnitude,
            cell=self.lattice_vectors.to('angstrom').magnitude,
            pbc=self.periodic_boundary_conditions,
        )
        return ase_atoms

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name
        self.analyze_geometry = kwargs.get('analyze_geometry', False)
        self.ase_atoms = None

    def normalize(self, archive, logger: BoundLogger):
        """When `analyze_geometry` is `true`, perform the geometry analysis.
        All intermediate artifacts are stored in the `AtomicCell` section.
        Only `GeometryDistribution` sections are retained in the archive written to disk."""
        super().normalize(archive, logger)
        self.name = self.m_def.name if self.name is None else self.name

        if self.analyze_geometry:
            # set up distributions
            cutoffs = ase_nl.natural_cutoffs(self.ase_atoms, mult=3.0) * ureg.angstrom
            self.plain_distributions = DistributionFactory(
                self.ase_atoms,
                ase_nl.build_neighbor_list(
                    self.ase_atoms,
                    cutoffs.magnitude,  # ase works in angstrom
                    self_interaction=False,
                ),
            ).produce_distributions()

            # write distributions to the archive
            (
                self.histogram_distributions,
                self.distance_distributions,
                self.angle_distributions,
            ) = [], [], []
            for distr in self.plain_distributions:
                # set binning
                if distr.n_elem == 2:
                    bins = (
                        np.arange(0, max(cutoffs), 0.001) * ureg.angstrom
                    )  # so binning may deviate
                    hist = distr.produce_histogram(bins)
                    self.distance_distributions.append(
                        hist.produce_nomad_distribution()
                    )
                elif distr.n_elem in (3, 4):
                    bins = np.arange(0, 180, 0.01) * ureg.degrees
                    hist = distr.produce_histogram(bins)
                    self.angle_distributions.append(hist.produce_nomad_distribution())
                self.histogram_distributions.append(hist)
        return self


class Symmetry(ArchiveSection):
    """
    A base section used to specify the symmetry of the `AtomicCell`.

    Note: this information can be extracted via normalization using the MatID package, if `AtomicCell`
    is specified.
    """

    bravais_lattice = Quantity(
        type=str,
        description="""
        Bravais lattice in Pearson notation.

        The first lowercase letter identifies the
        crystal family: a (triclinic), b (monoclinic), o (orthorhombic), t (tetragonal),
        h (hexagonal), c (cubic).

        The second uppercase letter identifies the centring: P (primitive), S (face centered),
        I (body centred), R (rhombohedral centring), F (all faces centred).
        """,
    )

    hall_symbol = Quantity(
        type=str,
        description="""
        Hall symbol for this system describing the minimum number of symmetry operations
        needed to uniquely define a space group. See https://cci.lbl.gov/sginfo/hall_symbols.html.
        Examples:
            - `F -4 2 3`,
            - `-P 4 2`,
            - `-F 4 2 3`.
        """,
    )

    point_group_symbol = Quantity(
        type=str,
        description="""
        Symbol of the crystallographic point group in the Hermann-Mauguin notation. See
        https://en.wikipedia.org/wiki/Crystallographic_point_group. Examples:
            - `-43m`,
            - `4/mmm`,
            - `m-3m`.
        """,
    )

    space_group_number = Quantity(
        type=np.int32,
        description="""
        Specifies the International Union of Crystallography (IUC) space group number of the 3D
        space group of this system. See https://en.wikipedia.org/wiki/List_of_space_groups.
        Examples:
            - `216`,
            - `123`,
            - `225`.
        """,
    )

    space_group_symbol = Quantity(
        type=str,
        description="""
        Specifies the International Union of Crystallography (IUC) space group symbol of the 3D
        space group of this system. See https://en.wikipedia.org/wiki/List_of_space_groups.
        Examples:
            - `F-43m`,
            - `P4/mmm`,
            - `Fm-3m`.
        """,
    )

    strukturbericht_designation = Quantity(
        type=str,
        description="""
        Classification of the material according to the historically grown and similar crystal
        structures ('strukturbericht'). Useful when using altogether with `space_group_symbol`.
        Examples:
            - `C1B`, `B3`, `C15b`,
            - `L10`, `L60`,
            - `L21`.

        Extracted from the AFLOW encyclopedia of crystallographic prototypes.
        """,
    )

    prototype_formula = Quantity(
        type=str,
        description="""
        The formula of the prototypical material for this structure as extracted from the
        AFLOW encyclopedia of crystallographic prototypes. It is a string with the chemical
        symbols:
            - https://aflowlib.org/prototype-encyclopedia/chemical_symbols.html
        """,
    )

    prototype_aflow_id = Quantity(
        type=str,
        description="""
        The identifier of this structure in the AFLOW encyclopedia of crystallographic prototypes:
            http://www.aflowlib.org/prototype-encyclopedia/index.html
        """,
    )

    atomic_cell_ref = Quantity(
        type=AtomicCell,
        description="""
        Reference to the AtomicCell section that the symmetry refers to.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    def resolve_analyzed_atomic_cell(
        self, symmetry_analyzer: SymmetryAnalyzer, cell_type: str, logger: BoundLogger
    ) -> Optional[AtomicCell]:  # ! consider using an index list and a filter function
        """
        Resolves the `AtomicCell` section from the `SymmetryAnalyzer` object and the cell_type
        (primitive or conventional).

        Args:
            symmetry_analyzer (SymmetryAnalyzer): The `SymmetryAnalyzer` object used to resolve.
            cell_type (str): The type of cell to resolve, either 'primitive' or 'conventional'.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[AtomicCell]): The resolved `AtomicCell` section or None if the cell_type
            is not recognized.
        """
        if cell_type not in ['primitive', 'conventional']:
            logger.error(
                "Cell type not recognized, only 'primitive' and 'conventional' are allowed."
            )
            return None
        wyckoff = getattr(symmetry_analyzer, f'get_wyckoff_letters_{cell_type}')()
        equivalent_atoms = getattr(
            symmetry_analyzer, f'get_equivalent_atoms_{cell_type}'
        )()
        system = getattr(symmetry_analyzer, f'get_{cell_type}_system')()

        positions = system.get_scaled_positions()
        cell = system.get_cell()
        atomic_numbers = system.get_atomic_numbers()
        labels = system.get_chemical_symbols()

        atomic_cell = AtomicCell(type=cell_type)
        atomic_cell.lattice_vectors = cell * ureg.angstrom
        for label, atomic_number in zip(labels, atomic_numbers):
            atom_state = AtomsState(chemical_symbol=label, atomic_number=atomic_number)
            atomic_cell.m_add_sub_section(AtomicCell.atoms_state, atom_state)
        atomic_cell.positions = (
            positions * ureg.angstrom
        )  # ? why do we need to pass units
        atomic_cell.wyckoff_letters = wyckoff
        atomic_cell.equivalent_atoms = equivalent_atoms
        atomic_cell.get_geometric_space_for_atomic_cell(logger)
        return atomic_cell

    def resolve_bulk_symmetry(
        self, original_atomic_cell: AtomicCell, logger: BoundLogger
    ) -> tuple[Optional[AtomicCell], Optional[AtomicCell]]:
        """
        Resolves the symmetry of the material being simulated using MatID and the
        originally parsed data under original_atomic_cell. It generates two other
        `AtomicCell` sections (the primitive and standardized cells) and populates
        the `Symmetry` section.

        Args:
            original_atomic_cell (AtomicCell): The `AtomicCell` section that the symmetry
            uses to in MatID.SymmetryAnalyzer().
            logger (BoundLogger): The logger to log messages.
        Returns:
            primitive_atomic_cell (Optional[AtomicCell]): The primitive `AtomicCell` section.
            conventional_atomic_cell (Optional[AtomicCell]): The standardized `AtomicCell` section.
        """
        symmetry = {}
        try:
            ase_atoms = original_atomic_cell.set_ase_atoms(logger)
            symmetry_analyzer = SymmetryAnalyzer(
                ase_atoms,
                symmetry_tol=config.normalize.symmetry_tolerance,  # ! @JosePizarro: it cannot find this import
            )
        except ValueError as e:
            logger.debug(
                'Symmetry analysis with MatID is not available.', details=str(e)
            )
            return None, None
        except Exception as e:
            logger.warning('Symmetry analysis with MatID failed.', exc_info=e)
            return None, None

        # We store symmetry_analyzer info in a dictionary
        symmetry['bravais_lattice'] = symmetry_analyzer.get_bravais_lattice()
        symmetry['hall_symbol'] = symmetry_analyzer.get_hall_symbol()
        symmetry['point_group_symbol'] = symmetry_analyzer.get_point_group()
        symmetry['space_group_number'] = symmetry_analyzer.get_space_group_number()
        symmetry['space_group_symbol'] = (
            symmetry_analyzer.get_space_group_international_short()
        )
        symmetry['origin_shift'] = symmetry_analyzer._get_spglib_origin_shift()
        symmetry['transformation_matrix'] = (
            symmetry_analyzer._get_spglib_transformation_matrix()
        )

        # Populating the originally parsed AtomicCell wyckoff_letters and equivalent_atoms information
        original_wyckoff = symmetry_analyzer.get_wyckoff_letters_original()
        original_equivalent_atoms = symmetry_analyzer.get_equivalent_atoms_original()
        original_atomic_cell.wyckoff_letters = original_wyckoff
        original_atomic_cell.equivalent_atoms = original_equivalent_atoms

        # Populating the primitive AtomState information
        primitive_atomic_cell = self.resolve_analyzed_atomic_cell(
            symmetry_analyzer, 'primitive', logger
        )

        # Populating the conventional AtomState information
        conventional_atomic_cell = self.resolve_analyzed_atomic_cell(
            symmetry_analyzer, 'conventional', logger
        )

        # Getting `prototype_formula`, `prototype_aflow_id`, and strukturbericht designation from
        # standardized Wyckoff numbers and the space group number
        if (
            symmetry.get('space_group_number')
            and conventional_atomic_cell.atoms_state is not None
        ):
            # Resolve atomic_numbers and wyckoff letters from the conventional cell
            conventional_num = [
                atom_state.atomic_number
                for atom_state in conventional_atomic_cell.atoms_state
            ]
            conventional_wyckoff = conventional_atomic_cell.wyckoff_letters
            # Normalize wyckoff letters
            norm_wyckoff = get_normalized_wyckoff(
                conventional_num, conventional_wyckoff
            )
            aflow_prototype = search_aflow_prototype(
                symmetry.get('space_group_number'), norm_wyckoff
            )
            if aflow_prototype:
                strukturbericht = aflow_prototype.get('Strukturbericht Designation')
                strukturbericht = (
                    re.sub('[$_{}]', '', strukturbericht)
                    if strukturbericht != 'None'
                    else None
                )
                prototype_aflow_id = aflow_prototype.get('aflow_prototype_id')
                prototype_formula = aflow_prototype.get('Prototype')
                # Adding these to the symmetry dictionary for later assignement
                symmetry['strukturbericht_designation'] = strukturbericht
                symmetry['prototype_aflow_id'] = prototype_aflow_id
                symmetry['prototype_formula'] = prototype_formula

        # Populating Symmetry section
        for key, val in self.m_def.all_quantities.items():
            self.m_set(val, symmetry.get(key))

        return primitive_atomic_cell, conventional_atomic_cell

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        atomic_cell = get_sibling_section(
            section=self, sibling_section_name='cell', logger=logger
        )
        if self.m_parent.type == 'bulk':
            # Adding the newly calculated primitive and conventional cells to the ModelSystem
            (
                primitive_atomic_cell,
                conventional_atomic_cell,
            ) = self.resolve_bulk_symmetry(atomic_cell, logger)
            self.m_parent.m_add_sub_section(ModelSystem.cell, primitive_atomic_cell)
            self.m_parent.m_add_sub_section(ModelSystem.cell, conventional_atomic_cell)
            # Reference to the standardized cell, and if not, fallback to the originally parsed one
            self.atomic_cell_ref = self.m_parent.cell[-1]


class ChemicalFormula(ArchiveSection):
    """
    A base section used to store the chemical formulas of a `ModelSystem` in different formats.
    """

    descriptive = Quantity(
        type=str,
        description="""
        The chemical formula of the system as a string to be descriptive of the computation.
        It is derived from `elemental_composition` if not specified, with non-reduced integer
        numbers for the proportions of the elements.
        """,
    )

    reduced = Quantity(
        type=str,
        description="""
        Alphabetically sorted chemical formula with reduced integer chemical proportion
        numbers. The proportion number is omitted if it is 1.
        """,
    )

    iupac = Quantity(
        type=str,
        description="""
        Chemical formula where the elements are ordered using a formal list based on
        electronegativity as defined in the IUPAC nomenclature of inorganic chemistry (2005):

            - https://en.wikipedia.org/wiki/List_of_inorganic_compounds

        Contains reduced integer chemical proportion numbers where the proportion number
        is omitted if it is 1.
        """,
    )

    hill = Quantity(
        type=str,
        description="""
        Chemical formula where Carbon is placed first, then Hydrogen, and then all the other
        elements in alphabetical order. If Carbon is not present, the order is alphabetical.
        """,
    )

    anonymous = Quantity(
        type=str,
        description="""
        Formula with the elements ordered by their reduced integer chemical proportion
        number, and the chemical species replaced by alphabetically ordered letters. The
        proportion number is omitted if it is 1.

        Examples: H2O becomes A2B and H2O2 becomes AB. The letters are drawn from the English
        alphabet that may be extended by increasing the number of letters: A, B, ..., Z, Aa, Ab
        and so on. This definition is in line with the similarly named `OPTIMADE` definition.
        """,
    )

    def resolve_chemical_formulas(self, formula: Formula) -> None:
        """
        Resolves the chemical formulas of the `ModelSystem` in different formats.

        Args:
            formula (Formula): The Formula object from NOMAD `atomutils` containing the chemical formulas.
        """
        self.descriptive = formula.format('descriptive')
        self.reduced = formula.format('reduced')
        self.iupac = formula.format('iupac')
        self.hill = formula.format('hill')
        self.anonymous = formula.format('anonymous')

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        atomic_cell = get_sibling_section(
            section=self, sibling_section_name='cell', logger=logger
        )
        if atomic_cell is None:
            logger.warning('Could not resolve the sibling `AtomicCell` section.')
            return
        ase_atoms = atomic_cell.to_ase_atoms(logger)
        formula = None
        try:
            formula = Formula(ase_atoms.get_chemical_formula())
            # self.chemical_composition = ase_atoms.get_chemical_formula(mode="all")
        except ValueError as e:
            logger.warning(
                'Could not extract the chemical formulas information.',
                exc_info=e,
                error=str(e),
            )
        if formula:
            self.resolve_chemical_formulas(formula)
            self.m_cache['elemental_composition'] = formula.elemental_composition()


class ModelSystem(System):
    """
    Model system used as an input for simulating the material.

    Definitions:
        - `name` refers to all the verbose and user-dependent naming in ModelSystem,
        - `type` refers to the type of the ModelSystem (atom, bulk, surface, etc.),
        - `dimensionality` refers to the dimensionality of the ModelSystem (0, 1, 2, 3),

    If the ModelSystem `is_representative`, proceeds with normalization. The time evolution of the
    ModelSystem is stored in a `list` format under `Simulation`, and for each element of that list,
    `time_step` can be defined.

    It is composed of the sub-sections:
        - `AtomicCell` containing the information of the atomic structure,
        - `Symmetry` containing the information of the (conventional) atomic cell symmetry
        in bulk ModelSystem,
        - `ChemicalFormula` containing the information of the chemical formulas in different
        formats.

    This class nest over itself (with the section proxy in `model_system`) to define different
    parent-child system trees. The quantities `branch_label`, `branch_depth`, `atom_indices`,
    and `bond_list` are used to define the parent-child tree.

    The normalization is run in the following order:
        1. `OrbitalsState.normalize()` in atoms_state.py under `AtomsState`
        2. `CoreHoleState.normalize()` in atoms_state.py under `AtomsState`
        3. `HubbardInteractions.normalize()` in atoms_state.py under `AtomsState`
        4. `AtomsState.normalize()` in atoms_state.py
        5. `AtomicCell.normalize()` in atomic_cell.py
        6. `Symmetry.normalize()` in this class
        7. `ChemicalFormula.normalize()` in this class
        8. `ModelSystem.normalize()` in this class

    Note: `normalize()` can be called at any time for each of the classes without being re-triggered
    by the NOMAD normalization.

    Examples for the parent-child hierarchical trees:

        - Example 1, a crystal Si has: 3 AtomicCell sections (named 'original', 'primitive',
        and 'conventional'), 1 Symmetry section, and 0 nested ModelSystem trees.

        - Example 2, an heterostructure Si/GaAs has: 1 parent ModelSystem section (for
        Si/GaAs together) and 2 nested child ModelSystem sections (for Si and GaAs); each
        child has 3 AtomicCell sections and 1 Symmetry section. The parent ModelSystem section
        could also have 3 AtomicCell and 1 Symmetry section (if it is possible to extract them).

        - Example 3, a solution of C800H3200Cu has: 1 parent ModelSystem section (for
        800*(CH4)+Cu) and 2 nested child ModelSystem sections (for CH4 and Cu); each child
        has 1 AtomicCell section.

        - Example 4, a passivated surface GaAs-CO2 has --> similar to the example 2.

        - Example 5, a passivated heterostructure Si/(GaAs-CO2) has: 1 parent ModelSystem
        section (for Si/(GaAs-CO2)), 2 child ModelSystem sections (for Si and GaAs-CO2),
        and 2 additional children sections in one of the children (for GaAs and CO2). The number
        of AtomicCell and Symmetry sections can be inferred using a combination of example
        2 and 3.
    """

    normalizer_level = 0

    name = Quantity(
        type=str,
        description="""
        Any verbose naming referring to the ModelSystem. Can be left empty if it is a simple
        crystal or it can be filled up. For example, an heterostructure of graphene (G) sandwiched
        in between hexagonal boron nitrides (hBN) slabs could be named 'hBN/G/hBN'.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    # TODO work on improving and extending this quantity and the description
    type = Quantity(
        type=MEnum(
            'atom',
            'active_atom',
            'molecule / cluster',
            '1D',
            'surface',
            '2D',
            'bulk',
            'unavailable',
        ),
        description="""
        Type of the system (atom, bulk, surface, etc.) which is determined by the normalizer.
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    dimensionality = Quantity(
        type=np.int32,
        description="""
        Dimensionality of the system: 0, 1, 2, or 3 dimensions. For atomistic systems this
        is automatically evaluated by using the topology-scaling algorithm:

            https://doi.org/10.1103/PhysRevLett.118.106101.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    # TODO improve on the definition and usage
    is_representative = Quantity(
        type=bool,
        default=False,
        description="""
        If the model system section is the one representative of the computational simulation.
        Defaults to False and set to True by the `Computation.normalize()`. If set to True,
        the `ModelSystem.normalize()` function is ran (otherwise, it is not).
        """,
    )

    # ? Check later when implementing `Outputs` if this quantity needs to be extended
    time_step = Quantity(
        type=np.int32,
        description="""
        Specific time snapshot of the ModelSystem. The time evolution is then encoded
        in a list of ModelSystems under Computation where for each element this quantity defines
        the time step.
        """,
    )

    cell = SubSection(sub_section=Cell.m_def, repeats=True)

    symmetry = SubSection(sub_section=Symmetry.m_def, repeats=True)

    chemical_formula = SubSection(sub_section=ChemicalFormula.m_def, repeats=False)

    branch_label = Quantity(
        type=str,
        description="""
        Label of the specific branch in the hierarchical `ModelSystem` tree.
        """,
    )

    branch_depth = Quantity(
        type=np.int32,
        description="""
        Index referring to the depth of a branch in the hierarchical `ModelSystem` tree.
        """,
    )

    # ? Move to `AtomicCell`?
    atom_indices = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        Indices of the atoms in the child with respect to its parent. Example:

        We have SrTiO3, where `AtomicCell.labels = ['Sr', 'Ti', 'O', 'O', 'O']`.
        - If we create a `model_system` child for the `'Ti'` atom only, then in that child
            `ModelSystem.model_system.atom_indices = [1]`.
        - If now we want to refer both to the `'Ti'` and the last `'O'` atoms,
            `ModelSystem.model_system.atom_indices = [1, 4]`.
        """,
    )

    model_system = SubSection(sub_section=SectionProxy('ModelSystem'), repeats=True)

    def resolve_system_type_and_dimensionality(
        self, ase_atoms: ase.Atoms, logger: BoundLogger
    ) -> tuple[str, int]:
        """
        Resolves the `ModelSystem.type` and `ModelSystem.dimensionality` using `MatID` classification analyzer:

            - https://singroup.github.io/matid/tutorials/classification.html

        Args:
            ase.Atoms: The ASE Atoms structure to analyse.
        Returns:
            system_type (str): The system type as determined by MatID.
            dimensionality (str): The system dimensionality as determined by MatID.
        """
        classification = None
        system_type, dimensionality = self.type, self.dimensionality
        if (
            len(ase_atoms)
            <= config.normalize.system_classification_with_clusters_threshold  # ! @JosePizarro: it cannot find this import
        ):
            try:
                classifier = Classifier(
                    radii='covalent',
                    cluster_threshold=config.normalize.cluster_threshold,  # ! @JosePizarro: it cannot find this import
                )
                classification = classifier.classify(ase_atoms)
            except Exception as e:
                logger.warning(
                    'MatID system classification failed.', exc_info=e, error=str(e)
                )
                return system_type, dimensionality

            if isinstance(classification, Class3D):
                system_type = 'bulk'
                dimensionality = 3
            elif isinstance(classification, Atom):
                system_type = 'atom'
                dimensionality = 0
            elif isinstance(classification, Class0D):
                system_type = 'molecule / cluster'
                dimensionality = 0
            elif isinstance(classification, Class1D):
                system_type = '1D'
                dimensionality = 1
            elif isinstance(classification, Surface):
                system_type = 'surface'
                dimensionality = 2
            elif isinstance(classification, (Class2D, Material2D)):
                system_type = '2D'
                dimensionality = 2
        else:
            logger.info(
                'ModelSystem.type and dimensionality analysis not run due to large system size.'
            )

        return system_type, dimensionality

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # We don't need to normalize if the system is not representative
        if is_not_representative(self, logger):
            return

        # Extracting ASE Atoms object from the originally parsed AtomicCell section
        if self.cell is None or len(self.cell) == 0:
            logger.warning(
                'Could not find the originally parsed atomic system. `Symmetry` and `ChemicalFormula` extraction is thus not run.'
            )
            return
        if self.cell[0].name == 'AtomicCell':
            self.cell[0].type = 'original'
            ase_atoms = self.cell[0].to_ase_atoms(logger)
            if not ase_atoms:
                return

            # Resolving system `type`, `dimensionality`, and Symmetry section (if this last
            # one does not exists already)
            original_atom_positions = self.cell[0].positions
            if original_atom_positions is not None:
                self.type = 'unavailable' if not self.type else self.type
                (
                    self.type,
                    self.dimensionality,
                ) = self.resolve_system_type_and_dimensionality(ase_atoms, logger)
                # Creating and normalizing Symmetry section
                if self.type == 'bulk' and self.symmetry is not None:
                    sec_symmetry = self.m_create(Symmetry)
                    sec_symmetry.normalize(archive, logger)

        # Creating and normalizing ChemicalFormula section
        # TODO add support for fractional formulas (possibly add `AtomicCell.concentrations` for each species)
        sec_chemical_formula = self.m_create(ChemicalFormula)
        sec_chemical_formula.normalize(archive, logger)
        if sec_chemical_formula.m_cache:
            self.elemental_composition = sec_chemical_formula.m_cache.get(
                'elemental_composition', []
            )
