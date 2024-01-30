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
from nomad.datamodel.metainfo.basesections import System, GeometricSpace
from nomad.datamodel.metainfo.annotations import ELNAnnotation


def check_parent_and_atomic_cell(section, logger):
    """
    Checks if the parent of a section exists and whether it has a sub-section atomic_cell.
    This is useful for other sub-sections under ModelSystem. It returns then the corresponding
    AtomicCell section.

    Args:
        section (ArchiveSection): The section to check for its parent and AtomicCell.

    Returns:
        (AtomicCell): The AtomicCell section resolved from the parent.
    """
    if section.m_parent is None and section.m_parent.atomic_cell is None:
        logger.error(
            "Could not find m_parent ModelSystem and its AtomicCell section for "
            "Symmetry analyzer."
        )
        return
    return section.m_parent.atomic_cell[0]


class AtomicCell(GeometricSpace):
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

    labels = Quantity(
        type=str,
        shape=["*"],
        description="""
        List containing the labels of the atomic species in the system at the different positions
        of the structure. It refers to a chemical element as defined in the periodic table,
        e.g., 'H', 'O', 'Pt'. This quantity is equivalent to `atomic_numbers`.
        """,
    )

    atomic_numbers = Quantity(
        type=np.int32,
        shape=["*"],
        description="""
        List of atomic numbers Z. This quantity is equivalent to `labels`.
        """,
    )

    positions = Quantity(
        type=np.float64,
        shape=["*", 3],
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
        shape=["*", 3],
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
        shape=["*"],
        description="""
        List of equivalent atoms as defined in `labels`. If no equivalent atoms are found,
        then the list is simply the index of each element, e.g.:
            - [0, 1, 2, 3] all four atoms are non-equivalent.
            - [0, 0, 0, 3] three equivalent atoms and one non-equivalent.
        """,
    )

    wyckoff_letters = Quantity(
        type=str,
        shape=["*"],
        # TODO improve description
        description="""
        Wyckoff letters associated with each atom position.
        """,
    )

    def to_ase_atoms(self, logger):
        """
        Generates a ASE Atoms object with the most basic information from the parsed AtomicCell
        section (labels, periodic_boundary_conditions, positions, and lattice_vectors).

        Returns:
            ase.Atoms: The ASE Atoms object with the basic information from the AtomicCell.
        """
        # Initialize ase.Atoms object with labels
        ase_atoms = ase.Atoms(symbols=self.labels)

        # PBC
        if self.periodic_boundary_conditions is None:
            logger.info("Could not find AtomicCell.periodic_boundary_conditions.")
            self.periodic_boundary_conditions = [False, False, False]
        ase_atoms.set_pbc(self.periodic_boundary_conditions)

        # Positions
        if self.positions is not None:
            if len(self.positions) != len(self.labels):
                logger.error(
                    "Length of AtomicCell.positions does not coincide with the length "
                    "of the AtomicCell.labels."
                )
                return
            ase_atoms.set_positions(self.positions.to("angstrom").magnitude)
        else:
            logger.error("Could not find AtomicCell.positions.")
            return

        # Lattice vectors
        if self.lattice_vectors is not None:
            ase_atoms.set_cell(self.lattice_vectors.to("angstrom").magnitude)
            if self.lattice_vectors_reciprocal is None:
                self.lattice_vectors_reciprocal = (
                    2 * np.pi * ase_atoms.get_reciprocal_cell() / ureg.angstrom
                )
        else:
            logger.info("Could not find AtomicCell.lattice_vectors.")

        return ase_atoms

    def normalize(self, archive, logger):
        # Check if AtomicCell section exists
        if self is None:
            logger.error(
                "Could not find the basic ModelSystem.atomic_cell information."
            )
            return

        # If the labels and atomic_numbers are not specified, we return with an error
        if self.labels is None and self.atomic_numbers is None:
            logger.error(
                "Could not read parsed AtomicCell.labels or AtomicCell.atomic_positions."
            )
            return
        atomic_labels = self.labels
        atomic_numbers = self.atomic_numbers

        # Labels
        if atomic_labels is None and atomic_numbers is not None:
            try:
                atomic_labels = [
                    ase.data.chemical_symbols[number] for number in atomic_numbers
                ]
            except IndexError:
                logger.error(
                    "The AtomicCell.atomic_numbers are out of range of the periodic table."
                )
                return
        self.labels = atomic_labels

        # We will use ASE Atoms functionalities to extract information about the AtomicCell
        ase_atoms = self.to_ase_atoms(logger)

        # Atomic numbers
        if atomic_labels is not None and atomic_numbers is None:
            atomic_numbers = ase_atoms.get_atomic_numbers()
        self.atomic_numbers = atomic_numbers

        # We then normalize `GeometricSpace`
        super().normalize(archive, logger)


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

    def resolve_bulk_symmetry(self, original_atomic_cell, logger):
        """
        Resolves the symmetry of the material being simulated using MatID and the
        originally parsed data under `original_atomic_cell`. It generates two other
        AtomicCell sections (the primitive and standarized cells), as well as populating
        the Symmetry section.

        Args:
            original_atomic_cell (AtomicCell): The AtomicCell section that the symmetry
            uses to in MatID.SymmetryAnalyzer().
        Returns:
            primitive_atomic_cell (AtomicCell): The primitive AtomicCell section.
            standard_atomic_cell (AtomicCell): The standarized AtomicCell section.
        """
        symmetry = {}
        try:
            ase_atoms = original_atomic_cell.to_ase_atoms(logger)
            symmetry_analyzer = SymmetryAnalyzer(
                ase_atoms, symmetry_tol=config.normalize.symmetry_tolerance
            )
        except ValueError as e:
            logger.debug(
                "Symmetry analysis with MatID is not available.", details=str(e)
            )
            return
        except Exception as e:
            logger.warning("Symmetry analysis with MatID failed.", exc_info=e)
            return

        # We store symmetry_analyzer info in a dictionary
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

        # Populating the originally parsed AtomicCell wyckoff_letters and equivalent_atoms information
        original_wyckoff = symmetry_analyzer.get_wyckoff_letters_original()
        original_equivalent_atoms = symmetry_analyzer.get_equivalent_atoms_original()
        original_atomic_cell.wyckoff_letters = original_wyckoff
        original_atomic_cell.equivalent_particles = original_equivalent_atoms

        # Populating the primitive AtomicCell information
        primitive_wyckoff = symmetry_analyzer.get_wyckoff_letters_primitive()
        primitive_equivalent_atoms = symmetry_analyzer.get_equivalent_atoms_primitive()
        primitive_sys = symmetry_analyzer.get_primitive_system()
        primitive_pos = primitive_sys.get_scaled_positions()
        primitive_cell = primitive_sys.get_cell()
        primitive_num = primitive_sys.get_atomic_numbers()
        primitive_labels = primitive_sys.get_chemical_symbols()
        primitive_atomic_cell = AtomicCell()
        primitive_atomic_cell.name = "primitive"
        primitive_atomic_cell.lattice_vectors = primitive_cell * ureg.angstrom
        primitive_atomic_cell.n_atoms = len(primitive_labels)
        primitive_atomic_cell.positions = primitive_pos * ureg.angstrom
        primitive_atomic_cell.labels = primitive_labels
        primitive_atomic_cell.atomic_numbers = primitive_num
        primitive_atomic_cell.wyckoff_letters = primitive_wyckoff
        primitive_atomic_cell.equivalent_atoms = primitive_equivalent_atoms
        primitive_atomic_cell.get_geometric_space_for_atomic_cell(logger)

        # Populating the standarized Atoms information
        standard_wyckoff = symmetry_analyzer.get_wyckoff_letters_conventional()
        standard_equivalent_atoms = (
            symmetry_analyzer.get_equivalent_atoms_conventional()
        )
        standard_sys = symmetry_analyzer.get_conventional_system()
        standard_pos = standard_sys.get_scaled_positions()
        standard_cell = standard_sys.get_cell()
        standard_num = standard_sys.get_atomic_numbers()
        standard_labels = standard_sys.get_chemical_symbols()
        standard_atomic_cell = AtomicCell()
        standard_atomic_cell.name = "standard"
        standard_atomic_cell.lattice_vectors = standard_cell * ureg.angstrom
        standard_atomic_cell.n_atoms = len(standard_labels)
        standard_atomic_cell.positions = standard_pos * ureg.angstrom
        standard_atomic_cell.labels = standard_labels
        standard_atomic_cell.atomic_numbers = standard_num
        standard_atomic_cell.wyckoff_letters = standard_wyckoff
        standard_atomic_cell.equivalent_atoms = standard_equivalent_atoms
        standard_atomic_cell.get_geometric_space_for_atomic_cell(logger)

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

        # Populating Symmetry section
        for key, val in self.m_def.all_quantities.items():
            self.m_set(val, symmetry.get(key))

        return primitive_atomic_cell, standard_atomic_cell

    def normalize(self, archive, logger):
        atomic_cell = check_parent_and_atomic_cell(self, logger)
        if self.m_parent.type == "bulk":
            # Adding the newly calculated primitive and standard cells to the ModelSystem
            primitive_atomic_cell, standard_atomic_cell = self.resolve_bulk_symmetry(
                atomic_cell, logger
            )
            self.m_parent.m_add_sub_section(
                ModelSystem.atomic_cell, primitive_atomic_cell
            )
            self.m_parent.m_add_sub_section(
                ModelSystem.atomic_cell, standard_atomic_cell
            )
            # Reference to the standarized cell, and if not, fallback to the originally parsed one
            self.atomic_cell_ref = self.m_parent.atomic_cell[-1]


class ChemicalFormula(ArchiveSection):
    """
    A base section used to store the chemical formulas of a ModelSystem in different formats.
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
        and so on. This definition is in line with the similarly named OPTIMADE definition.
        """,
    )

    def resolve_chemical_formulas(self, formula):
        """
        Resolves the chemical formulas of the ModelSystem in different formats.

        Args:
            formula (Formula): The Formula object from NOMAD atomutils containing the
            chemical formulas.
        """
        self.descriptive = formula.format("descriptive")
        self.reduced = formula.format("reduced")
        self.iupac = formula.format("iupac")
        self.hill = formula.format("hill")
        self.anonymous = formula.format("anonymous")

    def normalize(self, archive, logger):
        atomic_cell = check_parent_and_atomic_cell(self, logger)
        ase_atoms = atomic_cell.to_ase_atoms(logger)
        formula = None
        try:
            formula = Formula(ase_atoms.get_chemical_formula())
            # self.chemical_composition = ase_atoms.get_chemical_formula(mode="all")
        except ValueError as e:
            logger.warning(
                "Could not extract the chemical formulas information.",
                exc_info=e,
                error=str(e),
            )
        if formula:
            self.resolve_chemical_formulas(formula)
            self.m_cache["elemental_composition"] = formula.elemental_composition()


class ModelSystem(System):
    """
    Model system used as an input for the computation. It inherits from `System` where a set
    of sub-sections for the `elemental_composition` is defined.

    We also defined:
        - `name` refers to all the verbose and user-dependent naming in ModelSystem,
        - `type` refers to the type of the ModelSystem (atom, bulk, surface, etc.),
        - `dimensionality` refers to the dimensionality of the ModelSystem (0D, 1D, 2D, 3D),

    If the ModelSystem `is_representative`, the normalization occurs. The time evolution of
    the system is encoded on the fact that ModelSystem is a list under Simulation, and for
    each element of that list, `time_step` can be defined.

    It is composed of the sub-sections:
        - `AtomicCell` containing the information of the atomic structure,
        - `Symmetry` containing the information of the (standarized) atomic cell symmetry
        in bulk ModelSystem,
        - `ChemicalFormula` containing the information of the chemical formulas in different
        formats.

    This class nest over itself (with the proxy in `model_system`) to define different
    parent-child system trees. The quantities `tree_label`, `tree_index`, `atom_indices`,
    and `bond_list` are used to define the parent-child tree.

    The normalization is ran in the following order:
        1. `AtomicCell.normalize()` from `atomic_cell`,
        2. `ModelSystem.normalize()` in this class,
        3. `Symmetry.normalize()` is called within this class normalization,
        4. `ChemicalFormula.normalize()` is called within this class normalization.

    Examples:

        - Example 1, a crystal Si has: 3 AtomicCell sections (named 'original', 'primitive',
        and 'standard'), 1 Symmetry section, and 0 nested ModelSystem trees.

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
        and 2 additional children sections in one of the childs (for GaAs and CO2). The number
        of AtomicCell and Symmetry sections can be inferred using a combination of example
        2 and 3.
    """

    normalizer_level = 0

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
            "active_atom",
            "molecule / cluster",
            "1D",
            "surface",
            "2D",
            "bulk",
            "unavailable",
        ),
        description="""
        Type of the system (atom, bulk, surface, etc.) which is determined by the normalizer.
        """,
        a_eln=ELNAnnotation(component="EnumEditQuantity"),
    )

    dimensionality = Quantity(
        type=np.int32,
        description="""
        Dimensionality of the system: 0, 1, 2, or 3 dimensions. For atomistic systems this
        is automatically evaluated by using the topology-scaling algorithm:

            https://doi.org/10.1103/PhysRevLett.118.106101.
        """,
        a_eln=ELNAnnotation(component="NumberEditQuantity"),
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

    time_step = Quantity(
        type=np.int32,
        description="""
        Specific time snapshot of the ModelSystem. The time evolution is then encoded
        in a list of ModelSystems under Computation where for each element this quantity defines
        the time step.
        """,
    )

    atomic_cell = SubSection(sub_section=AtomicCell.m_def, repeats=True)

    symmetry = SubSection(sub_section=Symmetry.m_def, repeats=True)

    chemical_formula = SubSection(sub_section=ChemicalFormula.m_def, repeats=False)

    # TODO what about `branch_label`?
    tree_label = Quantity(
        type=str,
        shape=[],
        description="""
        Label of the specific branch in the system tree.
        """,
    )

    # TODO what about `branch_index` or `branch_depth`?
    tree_index = Quantity(
        type=np.int32,
        description="""
        Index refering to the depth of a branch in the system tree.
        """,
    )

    # TODO add method to resolve labels and positions from the parent AtomicCell
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

    def resolve_system_type_and_dimensionality(self, ase_atoms):
        """
        Determine the ModelSystem.type and ModelSystem.dimensionality using MatID classification analyzer:

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
                dimensionality = 3
            elif classification == Atom:
                system_type = "atom"
                dimensionality = 0
            elif classification == Class0D:
                system_type = "molecule / cluster"
                dimensionality = 0
            elif classification == Class1D:
                system_type = "1D"
                dimensionality = 1
            elif classification == Surface:
                system_type = "surface"
                dimensionality = 2
            elif classification == Material2D or classification == Class2D:
                system_type = "2D"
                dimensionality = 2
        else:
            self.logger.info(
                "ModelSystem.type and dimensionality analysis not run due to large system size."
            )

        return system_type, dimensionality

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        self.logger = logger

        # We don't need to normalize if the system is not representative
        if not self.is_representative:
            return

        # Extracting ASE Atoms object from the originally parsed AtomicCell section
        if self.atomic_cell is None:
            self.logger.warning(
                "Could not find the originally parsed atomic system. "
                "Symmetry and ChemicalFormula extraction is thus not run."
            )
            return
        self.atomic_cell[0].name = "original"
        ase_atoms = self.atomic_cell[0].to_ase_atoms(logger)
        if not ase_atoms:
            return

        # Resolving system `type`, `dimensionality`, and Symmetry section (if this last
        # one does not exists already)
        original_atom_positions = self.atomic_cell[0].positions
        if original_atom_positions is not None:
            self.type = "unavailable" if not self.type else self.type
            (
                self.type,
                self.dimensionality,
            ) = self.resolve_system_type_and_dimensionality(ase_atoms)
            # Creating and normalizing Symmetry section
            if self.type == "bulk" and self.symmetry is not None:
                sec_symmetry = self.m_create(Symmetry)
                sec_symmetry.normalize(archive, logger)

        # Creating and normalizing ChemicalFormula section
        # TODO add support for fractional formulas (possibly add `AtomicCell.concentrations` for each species)
        sec_chemical_formula = self.m_create(ChemicalFormula)
        sec_chemical_formula.normalize(archive, logger)
        if sec_chemical_formula.m_cache:
            self.elemental_composition = sec_chemical_formula.m_cache.get(
                "elemental_composition", []
            )
