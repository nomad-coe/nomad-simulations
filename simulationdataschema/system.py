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
#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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
import numpy as np
import ase

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

from nomad.datamodel.data import ArchiveSection

from nomad.metainfo import Quantity, SubSection, SectionProxy, MEnum
from nomad.datamodel.metainfo.basesections import System, RealSpace
from nomad.datamodel.metainfo.annotations import ELNAnnotation


class AtomicCell(RealSpace):
    """
    A base section used to specify the atomic cell quantities (labels, positions) of a system
    at a given moment in time.
    """

    name = Quantity(
        type=MEnum("original", "primitive", "standard"),
        description="""
        Name to identify the cell structure. It might be:
            - 'original' as in orignally parsed,
            - 'primitive' as the primitive unit cell,
            - 'standard' as the standarized cell used for referencing.
        """,
    )

    n_atoms = Quantity(
        type=np.int32,
        description="""
        The total number of atoms in the system.
        """,
    )

    labels = Quantity(
        type=str,
        shape=["n_atoms"],
        description="""
        List containing the labels of the atomic species in the system at the different positions
        of the structure. It refers to a chemical element as defined in the periodic table,
        e.g., 'H', 'O', 'Pt'. This quantity is equivalent to `atomic_numbers`.
        """,
    )

    atomic_numbers = Quantity(
        type=np.int32,
        shape=["n_atoms"],
        description="""
        List of atomic numbers Z. This quantity is equivalent to `labels`.
        """,
    )

    positions = Quantity(
        type=np.float64,
        shape=["n_atoms", 3],
        unit="meter",
        description="""
        Positions of all the atoms in Cartesian coordinates.
        """,
    )

    lattice_vectors = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="meter",
        description="""
        Lattice vectors of the simulated cell in Cartesian coordinates. The first index runs
        over each lattice vector. The second index runs over the $x, y, z$ Cartesian coordinates.
        """,
    )

    lattice_vectors_reciprocal = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="1/meter",
        description="""
        Reciprocal lattice vectors of the simulated cell, in Cartesian coordinates and
        including the $2 pi$ pre-factor. The first index runs over each lattice vector. The
        second index runs over the $x, y, z$ Cartesian coordinates.
        """,
    )

    periodic_boundary_conditions = Quantity(
        type=bool,
        shape=[3],
        description="""
        If periodic boundary conditions are applied to each direction of the crystal axes.
        """,
    )

    velocities = Quantity(
        type=np.float64,
        shape=["n_atoms", 3],
        unit="meter / second",
        description="""
        Velocities of the atoms. It is the change in cartesian coordinates of the atom position
        with time.
        """,
    )

    supercell_matrix = Quantity(
        type=np.int32,
        shape=[3, 3],
        description="""
        Specifies the matrix that transforms the primitive unit cell into the supercell in
        which the actual calculation is performed. In the easiest example, it is a diagonal
        matrix whose elements multiply the `lattice_vectors`, e.g., [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        is a $3 x 3 x 3$ superlattice.
        """,
    )

    equivalent_atoms = Quantity(
        type=np.int32,
        shape=["n_atoms"],
        description="""
        List of equivalent atoms as defined in `labels`. If no equivalent atoms are found,
        then the list is simply the index of each element, e.g.:
            - [0, 1, 2, 3] all four atoms are non-equivalent.
            - [0, 0, 0, 3] three equivalent atoms and one non-equivalent.
        """,
    )

    wyckoff_letters = Quantity(
        type=str,
        shape=["n_atoms"],
        # TODO improve description
        description="""
        Wyckoff letters associated with each atom position.
        """,
    )

    def normalize(self, archive, logger) -> None:
        # Check if AtomicCell section exists
        if self is None:
            logger.error("Could not find the basic System.atomic_cell information.")

        # Resolving atom_labels (either directly or from atomic_numbers)
        atom_labels = self.labels
        atomic_numbers = self.atomic_numbers
        if atom_labels is None:
            if atomic_numbers is None:
                logger.error(
                    "System.atomic_cell has neither labels nor atomic_numbers defined."
                )
                return
            try:
                atom_labels = [
                    ase.data.chemical_symbols[number] for number in atomic_numbers
                ]
            except IndexError:
                logger.error(
                    "System.atomic_cell has atomic_numbers that are out of range of the periodic table."
                )
                return
            self.labels = atom_labels
        self.n_atoms = len(atom_labels)

        # Using ASE functionalities to write atomic_numbers
        ase_atoms = ase.Atoms(symbols=atom_labels)
        chemical_symbols = ase_atoms.get_chemical_symbols()
        if atom_labels != list(chemical_symbols):
            logger.warning(
                "Chemical symbols in System.atomic_cell.labels are ambigous and cannot be "
                "recognized by ASE."
            )
        if atomic_numbers is None:
            atomic_numbers = ase_atoms.get_atomic_numbers()
        else:
            if atomic_numbers != list(ase_atoms.get_atomic_numbers()):
                logger.info(
                    "The parsed System.atomic_cell.atomic_numbers do not coincide with "
                    "the ASE extracted numbers from the labels. We will rewrite the parsed data."
                )
                atomic_numbers = ase_atoms.get_atomic_numbers()
        self.atomic_numbers = atomic_numbers

        # Periodic boundary conditions
        pbc = self.periodic_boundary_conditions
        if pbc is None:
            pbc = [False, False, False]
            logger.info(
                "Could not find System.atomic_cell.periodic_boundary_conditions information. "
                "Setting them to False."
            )
            self.periodic_boundary_conditions = pbc
        ase_atoms.set_pbc(pbc)

        # Atom positions
        atom_positions = self.positions
        if atom_positions is None or len(atom_positions) == 0:
            logger.error("Could not find System.atomic_cell.positions.")
            return
        if len(atom_positions) != len(atom_labels):
            logger.error(
                "Length of System.atomic_cell.positions does not coincide with the length "
                "of the System.atomic_cell.labels."
            )
            return
        ase_atoms.set_positions(atom_positions.to("angstrom").magnitude)

        # Lattice vectors and reciprocal lattice vectors
        lattice_vectors = self.lattice_vectors
        if lattice_vectors is None:
            logger.info("Could not find System.atomic_cell.lattice_vectors.")
        else:
            ase_atoms.set_cell(lattice_vectors.to("angstrom").magnitude)
            lattice_vectors_reciprocal = self.lattice_vectors_reciprocal
            if lattice_vectors_reciprocal is None:
                self.lattice_vectors_reciprocal = (
                    2 * np.pi * ase_atoms.get_reciprocal_cell() / ureg.angstrom
                )

        # Store temporarily the ase.Atoms object to use in the ModelSystem.normalizer()
        self.m_cache["ase_atoms"] = ase_atoms

    def to_ase_atoms(self, nomad_atomic_cell) -> ase.Atoms:
        """
        Generates a ASE Atoms object with the most basic information from the parsed AtomicCell
        section (labels, positions, and lattice_vectors).
        """
        ase_atoms = ase.Atoms(symbols=nomad_atomic_cell.labels)
        ase_atoms.set_positions(nomad_atomic_cell.positions.to("angstrom").magnitude)
        ase_atoms.set_cell(nomad_atomic_cell.lattice_vectors.to("angstrom").magnitude)
        return ase_atoms


class Symmetry(ArchiveSection):
    """
    A base section used to specify the symmetry of the AtomicCell. This information can
    be extracted via normalization using the MatID package, if `AtomicCell` is specified.
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
        Hall symbol for this system describing the minimum number of symmetry operations,
        in the form of Seitz matrices, needed to uniquely define a space group. See
        https://cci.lbl.gov/sginfo/hall_symbols.html. Examples:
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
        a_eln=ELNAnnotation(component="ReferenceEditQuantity"),
    )


class ModelSystem(System):
    """
    Model system used as an input for the computation. It inherits from `System` where a set
    of sub-sections for the `elemental_composition` is defined. We also define `name` to
    refer to all the verbose and user-dependent naming in ModelSystem.

    It is composed of the sub-sections: AtomicCell (atomic structure quantities), Symmetry
    (symmetry of the ModelSystem and which always references the 'standard' AtomicCell).

    This class nest over itself (with the proxy in `model_system`) to define different
    parent-child system trees.

    The time evolution of the system is encoded on the fact that ModelSystem is a list under
    computation, and for each element of that list, `time_step` can be defined.

    The normalization is ran in the following order:
        1. `AtomicCell.normalize()` from `atomic_cell`,
        2. `ModelSystem.normalize()` in this class.

    Examples:

        - Example 1, a crystal Si has: 3 AtomicCell sections (named 'original', 'primitive',
        and 'standard'), 1 Symmetry, and 0 nested ModelSystem trees.

        - Example 2, an heterostructure Si/GaAs has: 1 parent ModelSystem (for Si/GaAs together)
        and 2 nested child ModelSystem sections (for Si and GaAs); each child has 3 AtomicCell
        sections and 1 Symmetry section. The parent ModelSystem could also have 3 AtomicCell
        and 1 Symmetry section (if it is possible to extract them).

        - Example 3, a solution of C800H3200Cu has: 1 parent ModelSystem (for 800*(CH4)+Cu)
        and 2 nested child ModelSystem sections (for CH4 and Cu); each child has 1 AtomicCell
        section.

        - Example 4, a passivated surface GaAs-CO2 has --> similar to the example 2.

        - Example 5, a passivated heterostructure Si/(GaAs-CO2) has: 1 parent ModelSystem
        (for Si/(GaAs-CO2)), 2 child ModelSystems (for Si and GaAs-CO2), and 2 additional
        children in one of the childs (for GaAs and CO2). The number of AtomicCell and Symmetry
        sections can be inferred using a combination of example 2 and 3.
    """

    name = Quantity(
        type=str,
        description="""
        Any verbose naming refering to the ModelSystem. Can be left empty if it is a simple
        crystal or it can be filled up. For example, an heterostructure of graphene (G) sandwiched
        in between hexagonal boron nitrides (hBN) slabs could be named 'hBN/G/hBN'.
        """,
        a_eln=ELNAnnotation(component="StringEditQuantity"),
    )

    # TODO work on improving and extending this quantity and the description
    type = Quantity(
        type=MEnum(
            "atom",
            "molecule / cluster",
            "bulk",
            "surface",
            "2D",
            "1D",
            "active_atom",
            "unavailable",
        ),
        description="""
        Type of the system (atom, bulk, surface, etc.) which is determined by the normalizer.
        """,
    )

    dimensionality = Quantity(
        type=MEnum("0D", "1D", "2D", "3D", "unavailable"),
        description="""
        Dimensionality of the system. For atomistic systems this is automatically evaluated
        by using the topology-scaling algorithm:

            https://doi.org/10.1103/PhysRevLett.118.106101.

        | Value | Description |
        | --------- | ----------------------- |
        | `'0D'` | Points in the space |
        | `'1D'` | Periodi in one dimension |
        | `'2D'` | Periodic in two dimensions |
        | `'3D'` | Periodic in three dimensions |
        """,
    )

    time_step = Quantity(
        type=np.int32,
        description="""
        Specific time snapshot of the ModelSystem. The time evolution is then encoded
        in a list of ModelSystems under Computation where for each element this quantity defines
        the time step.
        """,
    )

    chemical_formula_descriptive = Quantity(
        type=str,
        description="""
        The chemical formula of the system as a string to be descriptive of the computation.
        It is derived from `elemental_composition` if not specified, with non-reduced integer
        numbers for the proportions of the elements.
        """,
    )

    chemical_formula_reduced = Quantity(
        type=str,
        description="""
        Alphabetically sorted chemical formula with reduced integer chemical proportion
        numbers. The proportion number is omitted if it is 1.
        """,
    )

    chemical_formula_iupac = Quantity(
        type=str,
        description="""
        Chemical formula where the elements are ordered using a formal list based on
        electronegativity as defined in the IUPAC nomenclature of inorganic chemistry (2005):

            - https://en.wikipedia.org/wiki/List_of_inorganic_compounds

        Contains reduced integer chemical proportion numbers where the proportion number
        is omitted if it is 1.
        """,
    )

    chemical_formula_hill = Quantity(
        type=str,
        description="""
        Chemical formula where Carbon is placed first, then Hydrogen, and then all the other
        elements in alphabetical order. If Carbon is not present, the order is alphabetical.
        """,
    )

    chemical_formula_anonymous = Quantity(
        type=str,
        description="""
        Formula with the elements ordered by their reduced integer chemical proportion
        number, and the chemical species replaced by alphabetically ordered letters. The
        proportion number is omitted if it is 1.

        Examples: H2O becomes A2B and H2O2 becomes AB. The letters are drawn from the English
        alphabet that may be extended by increasing the number of letters: A, B, ..., Z, Aa, Ab
        and so on. This definition is in line with the similarly named OPTIMADE definition.
        """,
    )

    atomic_cell = SubSection(sub_section=AtomicCell.m_def, repeats=True)

    symmetry = SubSection(sub_section=Symmetry.m_def, repeats=True)

    is_representative = Quantity(
        type=bool,
        default=False,
        description="""
        If the model system section is the one representative of the computational simulation.
        Defaults to False and set to True by the `Computation.normalize()`. If set to True,
        the `ModelSystem.normalize()` function is ran (otherwise, it is not).
        """,
    )

    # TODO what about `branch_label`?
    tree_label = Quantity(
        type=str,
        shape=[],
        description="""
        Label of the specific branch in the system tree.
        """,
    )

    # TODO what about `branch_index`?
    tree_index = Quantity(
        type=np.int32,
        description="""
        Index refering to the depth of a branch in the system tree.
        """,
    )

    atom_indices = Quantity(
        type=np.int32,
        shape=["*"],
        description="""
        Indices of the atoms in the child with respect to its parent. Example:
            - We have SrTiO3, where `AtomicCell.labels = ['Sr', 'Ti', 'O', 'O', 'O']`. If
            we create a `model_system` child for the `'Ti'` atom only, then in that child
            `ModelSystem.model_system.atom_indices = [1]`. If now we want to refer both to
            the `'Ti'` and the last `'O'` atoms, `ModelSystem.model_system.atom_indices = [1, 4]`.
        """,
    )

    bond_list = Quantity(
        type=np.int32,
        # TODO improve description and add an example using the case in atom_indices
        description="""
        List of pairs of atom indices corresponding to bonds (e.g., as defined by a force field)
        within this atoms_group.
        """,
    )

    model_system = SubSection(sub_section=SectionProxy("ModelSystem"), repeats=True)

    def _resolve_system_type_and_dimensionality(self, ase_atoms: ase.Atoms) -> str:
        """
        Determine the ModelSystem.type and ModelSystem.dimensionality using MatID classification analyzer:

            - https://singroup.github.io/matid/tutorials/classification.html

        Args:
            ase.Atoms: The ASE Atoms structure to analyse
        """
        classification = None
        system_type, dimensionality = self.type, self.dimensionality
        if (
            len(ase_atoms)
            <= config.normalize.system_classification_with_clusters_threshold
        ):
            try:
                classifier = Classifier(
                    radii="covalent",
                    cluster_threshold=config.normalize.cluster_threshold,
                )
                cls = classifier.classify(ase_atoms)
            except Exception as e:
                self.logger.warning(
                    "MatID system classification failed.", exc_info=e, error=str(e)
                )
                return system_type, dimensionality

            classification = type(cls)
            if classification == Class3D:
                system_type = "bulk"
                dimensionality = "3D"
            elif classification == Atom:
                system_type = "atom"
                dimensionality = "3D"
            elif classification == Class0D:
                system_type = "molecule / cluster"
                dimensionality = "0D"
            elif classification == Class1D:
                system_type = "1D"
                dimensionality = "1D"
            elif classification == Surface:
                system_type = "surface"
                dimensionality = "2D"
            elif classification == Material2D or classification == Class2D:
                system_type = "2D"
                dimensionality = "2D"
        else:
            self.logger.info(
                "ModelSystem.type and dimensionality analysis not run due to large system size."
            )

        return system_type, dimensionality

    def _resolve_bulk_symmetry(self, ase_atoms: ase.Atoms) -> None:
        """
        Analyze the symmetry of the material being simulated using MatID and the parsed data
        stored under ModelSystem and AtomicCell. Only available for bulk materials.

        Args:
            ase.Atoms: The ASE Atoms structure to analyse
        """
        symmetry = {}
        try:
            symmetry_analyzer = SymmetryAnalyzer(
                ase_atoms, symmetry_tol=config.normalize.symmetry_tolerance
            )

            symmetry["bravais_lattice"] = symmetry_analyzer.get_bravais_lattice()
            symmetry["hall_symbol"] = symmetry_analyzer.get_hall_symbol()
            symmetry["point_group_symbol"] = symmetry_analyzer.get_point_group()
            symmetry["space_group_number"] = symmetry_analyzer.get_space_group_number()
            symmetry[
                "space_group_symbol"
            ] = symmetry_analyzer.get_space_group_international_short()
            symmetry["origin_shift"] = symmetry_analyzer._get_spglib_origin_shift()
            symmetry[
                "transformation_matrix"
            ] = symmetry_analyzer._get_spglib_transformation_matrix()

            # Originally parsed cell
            original_wyckoff = symmetry_analyzer.get_wyckoff_letters_original()
            original_equivalent_atoms = (
                symmetry_analyzer.get_equivalent_atoms_original()
            )

            # Primitive cell
            primitive_wyckoff = symmetry_analyzer.get_wyckoff_letters_primitive()
            primitive_equivalent_atoms = (
                symmetry_analyzer.get_equivalent_atoms_primitive()
            )
            primitive_sys = symmetry_analyzer.get_primitive_system()
            primitive_pos = primitive_sys.get_scaled_positions()
            primitive_cell = primitive_sys.get_cell()
            primitive_num = primitive_sys.get_atomic_numbers()
            primitive_labels = primitive_sys.get_chemical_symbols()

            # Standarized (or conventional) cell
            standard_wyckoff = symmetry_analyzer.get_wyckoff_letters_conventional()
            standard_equivalent_atoms = (
                symmetry_analyzer.get_equivalent_atoms_conventional()
            )
            standard_sys = symmetry_analyzer.get_conventional_system()
            standard_pos = standard_sys.get_scaled_positions()
            standard_cell = standard_sys.get_cell()
            standard_num = standard_sys.get_atomic_numbers()
            standard_labels = standard_sys.get_chemical_symbols()
        except ValueError as e:
            self.logger.debug(
                "Symmetry analysis with MatID is not available.", details=str(e)
            )
            return
        except Exception as e:
            self.logger.warning("Symmetry analysis with MatID failed.", exc_info=e)
            return

        # Populating the originally parsed AtomicCell wyckoff_letters and equivalent_atoms information
        sec_original_atoms = self.atomic_cell[0]
        sec_original_atoms.wyckoff_letters = original_wyckoff
        sec_original_atoms.equivalent_particles = original_equivalent_atoms

        # Populating the primitive AtomicCell information
        sec_primitive_atoms = self.m_create(AtomicCell)
        sec_primitive_atoms.name = "primitive"
        sec_primitive_atoms.lattice_vectors = primitive_cell * ureg.angstrom
        sec_primitive_atoms.n_atoms = len(primitive_labels)
        sec_primitive_atoms.positions = primitive_pos * ureg.angstrom
        sec_primitive_atoms.labels = primitive_labels
        sec_primitive_atoms.atomic_numbers = primitive_num
        sec_primitive_atoms.wyckoff_letters = primitive_wyckoff
        sec_primitive_atoms.equivalent_atoms = primitive_equivalent_atoms

        # Populating the standarized Atoms information
        sec_standard_atoms = self.m_create(AtomicCell)
        sec_standard_atoms.name = "standard"
        sec_standard_atoms.lattice_vectors = standard_cell * ureg.angstrom
        sec_standard_atoms.n_atoms = len(standard_labels)
        sec_standard_atoms.positions = standard_pos * ureg.angstrom
        sec_standard_atoms.labels = standard_labels
        sec_standard_atoms.atomic_numbers = standard_num
        sec_standard_atoms.wyckoff_letters = standard_wyckoff
        sec_standard_atoms.equivalent_atoms = standard_equivalent_atoms

        # Getting prototype_formula, prototype_aflow_id, and strukturbericht designation from
        # standarized Wyckoff numbers and the space group number
        if symmetry.get("space_group_number"):
            norm_wyckoff = get_normalized_wyckoff(standard_num, standard_wyckoff)
            aflow_prototype = search_aflow_prototype(
                symmetry.get("space_group_number"), norm_wyckoff
            )
            strukturbericht = aflow_prototype.get("Strukturbericht Designation")
            if strukturbericht == "None":
                strukturbericht = None
            else:
                strukturbericht = re.sub("[$_{}]", "", strukturbericht)
            prototype_aflow_id = aflow_prototype.get("aflow_prototype_id")
            prototype_formula = aflow_prototype.get("Prototype")
            # Adding these to the symmetry dictionary for later assignement
            symmetry["strukturbericht_designation"] = strukturbericht
            symmetry["prototype_aflow_id"] = prototype_aflow_id
            symmetry["prototype_formula"] = prototype_formula

        # Populating Symmetry section (try to reference the standarized cell, and if not,
        # fallback to the originally parsed one)
        sec_symmetry = self.m_create(Symmetry)
        for key, val in sec_symmetry.m_def.all_quantities.items():
            sec_symmetry.m_set(val, symmetry.get(key))
        sec_symmetry.atomic_cell_ref = self.atomic_cell[-1]

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.logger = logger

        # We don't need to normalize if the system is not representative
        # self.is_representative = True
        if not self.is_representative:
            return

        # Extracting ASE Atoms object from the originally parsed AtomicCell section
        if len(self.atomic_cell) == 0:
            self.logger.warning(
                "Could not find the originally parsed atomic system. "
                "Symmetry and Formula extraction is thus not run."
            )
            return
        self.atomic_cell[0].name = "original"
        ase_atoms = self.atomic_cell[0].m_cache.get("ase_atoms")
        if not ase_atoms:
            return

        # Resolving system `type`, `dimensionality`, and Symmetry section (if this last one does not exists already)
        original_atom_positions = self.atomic_cell[0].positions
        if original_atom_positions is not None:
            self.type = "unavailable" if not self.type else self.type
            self.dimensionality = (
                "unavailable" if not self.dimensionality else self.dimensionality
            )
            (
                self.type,
                self.dimensionality,
            ) = self._resolve_system_type_and_dimensionality(ase_atoms)
            if self.type == "bulk" and len(self.symmetry) == 0:
                self._resolve_bulk_symmetry(ase_atoms)
            # Extracting the cells parameters using the object Cell from ASE
            for atom_cell in self.atomic_cell:
                atoms = AtomicCell().to_ase_atoms(atom_cell)
                cell = atoms.get_cell()
                atom_cell.a, atom_cell.b, atom_cell.c = cell.lengths() * ureg.angstrom
                atom_cell.alpha, atom_cell.beta, atom_cell.gamma = (
                    cell.angles() * ureg.degree
                )
                atom_cell.volume = cell.volume * ureg.angstrom**3

        # Formulas
        # TODO add support for fractional formulas (possibly add `AtomicCell.concentrations` for each species)
        try:
            formula = Formula(ase_atoms.get_chemical_formula())
            self.chemical_composition = ase_atoms.get_chemical_formula(mode="all")
            self.chemical_formula_descriptive = formula.format("descriptive")
            self.chemical_formula_reduced = formula.format("reduced")
            self.chemical_formula_iupac = formula.format("iupac")
            self.chemical_formula_hill = formula.format("hill")
            self.chemical_formula_anonymous = formula.format("anonymous")
            self.elemental_composition = formula.elemental_composition()
        except ValueError as e:
            self.logger.warning(
                "Could not extract the chemical formulas information.",
                exc_info=e,
                error=str(e),
            )
