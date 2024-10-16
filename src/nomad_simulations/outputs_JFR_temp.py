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

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity, SubSection, SectionProxy, MEnum
from nomad.metainfo.metainfo import DirectQuantity, Dimension
from nomad.datamodel.metainfo.annotations import ELNAnnotation

from .outputs import Outputs
from .property import BaseProperty
from .atoms_state import AtomsState


from nomad.datamodel.metainfo.common import (
    FastAccess,
    PropertySection,
    ProvenanceTracker,
)
from nomad.metainfo import (
    Category,
    HDF5Reference,
    MCategory,
    MEnum,
    MSection,
    Package,
    Quantity,
    Reference,
    Section,
    SectionProxy,
    SubSection,
)

from .model_system import ModelSystem

class ScalarProperty(BaseProperty):
    """
    Generic section containing the values and information for any scalar property.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # TODO check that all variable and bin quantities are None

class AtomicProperty(BaseProperty):
    """
    Generic section containing the values and information reqarding an atomic quantity
    such as charges, forces, multipoles.
    """

    n_orbitals = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of orbitals used in the projection.
        """,
    )
    # ? Is "the projection" general/clear?

    # TODO we should make clear in these descriptions that it is preferred to reference the relevant sub-system if existing,
    # TODO otherwise these quantities can be populated "by hand"
    n_atoms = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of atoms used to determine/calculate the property.
        """,
    )

    atom_labels = Quantity(
        type=str,
        shape=[],
        description="""
        Labels of the atomic species corresponding to the atomic quantity.
        """,
    )

    atom_indices = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
        Index of the atomic species corresponding to the atomic quantity.
        """,
    )

    n_spin_channels = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of spin channels.
        """,
    )

    spin_channels = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
        Spin channels corresponding to the atomic quantity.
        """,
    )

    atoms_state_ref = Quantity(
        type=AtomsState,
        description="""
        Reference to the AtomsState section for the atoms involved in this property.
        """,
    )

    # ? below deprecated by atoms_state?
    # m_kind = Quantity(
    #     type=str,
    #     shape=[],
    #     description="""
    #     String describing what the integer numbers of $m$ lm mean used in orbital
    #     projections. The allowed values are listed in the [m_kind wiki page]
    #     (https://gitlab.rzg.mpg.de/nomad-lab/nomad-meta-info/wikis/metainfo/m-kind).
    #     """,
    # )

    # lm = Quantity(
    #     type=np.dtype(np.int32),
    #     shape=[2],
    #     description="""
    #     Tuples of $l$ and $m$ values for which the atomic quantity are given. For
    #     the quantum number $l$ the conventional meaning of azimuthal quantum number is
    #     always adopted. For the integer number $m$, besides the conventional use as
    #     magnetic quantum number ($l+1$ integer values from $-l$ to $l$), a set of
    #     different conventions is accepted (see the [m_kind wiki
    #     page](https://gitlab.rzg.mpg.de/nomad-lab/nomad-meta-info/wikis/metainfo/m-kind).
    #     The adopted convention is specified by m_kind.
    #     """,
    # )

    # orbital = Quantity(
    #     type=str,
    #     shape=[],
    #     description="""
    #     String representation of the of the atomic orbital.
    #     """,
    # )

class EnergyEntry(Atomic):
    """
    Section describing a type of energy or a contribution to the total energy.
    """

    reference = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the reference energy to be subtracted from value to obtain a
        code-independent value of the energy.
        """,
    )

    # TODO Can we remove reference to unit cell in this description to make more general?
    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the energy of the unit cell.
        """,
    )

    value_per_atom = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the energy normalized by the total number of atoms in the simulation
        cell.
        """,
    )

    # TODO rename this to value_atomic
    values_per_atom = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms'],
        unit='joule',
        description="""
        Value of the atom-resolved energies.
        """,
    )

    potential = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the potential energy.
        """,
    )

    kinetic = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the kinetic energy.
        """,
    )

    correction = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the correction to the energy.
        """,
    )

    short_range = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the short range contributions to the energy.
        """,
    )

    long_range = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the long range contributions to the energy.
        """,
    )


class Energy(ArchiveSection):
    """
    Section containing all energy types and contributions.
    """

    total = SubSection(
        sub_section=EnergyEntry.m_def,
        categories=[FastAccess],
        description="""
        Contains the value and information regarding the total energy of the system.
        """,
    )

    # TODO this should be removed and replaced by correction in EnergyEntry
    current = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the energy calculated with
        calculation_method_current. energy_current is equal to energy_total for
        non-perturbative methods. For perturbative methods, energy_current is equal to the
        correction: energy_total minus energy_total of the calculation_to_calculation_ref
        with calculation_to_calculation_kind = starting_point
        """,
    )

    zero_point = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the converged zero-point
        vibrations energy calculated using the method described in zero_point_method.
        """,
    )
    # this should be removed and replaced by electronic.kinetic
    kinetic_electronic = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the self-consistent electronic
        kinetic energy.
        """,
    )

    electronic = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the self-consistent electronic
        energy.
        """,
    )

    correlation = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the correlation energy calculated
        using the method described in XC_functional.
        """,
    )

    exchange = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the exchange energy calculated
        using the method described in XC_functional.
        """,
    )

    xc = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the exchange-correlation (XC)
        energy calculated with the functional stored in XC_functional.
        """,
    )

    # TODO Remove this should use xc.potential
    xc_potential = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the exchange-correlation (XC)
        potential energy: the integral of the first order derivative of the functional
        stored in XC_functional (integral of v_xc*electron_density), i.e., the component
        of XC that is in the sum of the eigenvalues. Value associated with the
        configuration, should be the most converged value..
        """,
    )

    electrostatic = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the total electrostatic energy
        (nuclei + electrons), defined consistently with calculation_method.
        """,
    )

    nuclear_repulsion = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the total nuclear-nuclear repulsion
        energy.
        """,
    )

    # TODO remove this or electrostatic
    coulomb = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the Coulomb energy.
        """,
    )

    madelung = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the Madelung energy.
        """,
    )

    # TODO I suggest ewald is moved to "long range" under electrostatic->energyentry, unless there is some other usage I am misunderstanding
    ewald = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the Ewald energy.
        """,
    )

    free = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the free energy (nuclei + electrons)
        (whose minimum gives the smeared occupation density calculated with
        smearing_kind).
        """,
    )

    sum_eigenvalues = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the sum of the eigenvalues of the
        Hamiltonian matrix.
        """,
    )

    total_t0 = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the total energy extrapolated to
        $T=0$, based on a free-electron gas argument.
        """,
    )

    van_der_waals = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the Van der Waals energy. A multiple
        occurence is expected when more than one van der Waals methods are defined. The
        van der Waals kind should be specified in Energy.kind
        """,
    )

    hartree_fock_x_scaled = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Scaled exact-exchange energy that depends on the mixing parameter of the
        functional. For example in hybrid functionals, the exchange energy is given as a
        linear combination of exact-energy and exchange energy of an approximate DFT
        functional; the exact exchange energy multiplied by the mixing coefficient of the
        hybrid functional would be stored in this metadata. Defined consistently with
        XC_method.
        """,
    )

    contributions = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains other energy contributions to the total energy not already defined.
        """,
        repeats=True,
    )

    types = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains other energy types not already defined.
        """,
        repeats=True,
    )

    enthalpy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the calculated enthalpy per cell i.e. energy_total + pressure * volume.
        """,
    )

    # TODO Shouldn't this be moved out of energy?
    entropy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule / kelvin',
        description="""
        Value of the entropy.
        """,
    )

    chemical_potential = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the chemical potential.
        """,
    )

    internal = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the internal energy.
        """,
    )

    double_counting = SubSection(
        sub_section=EnergyEntry.m_def,
        categories=[FastAccess],
        description="""
        Double counting correction when performing Hubbard model calculations.
        """,
    )

    # TODO remove this should be be entropy.correction
    correction_entropy = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Entropy correction to the potential energy to compensate for the change in
        occupation so that forces at finite T do not need to keep the change of occupation
        in account. Defined consistently with XC_method.
        """,
    )

    # TODO remove this should be in electrostatic.correction
    correction_hartree = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Correction to the density-density electrostatic energy in the sum of eigenvalues
        (that uses the mixed density on one side), and the fully consistent density-
        density electrostatic energy. Defined consistently with XC_method.
        """,
    )

    # TODO remove this should be in xc.correction
    correction_xc = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Correction to energy_XC.
        """,
    )

    change = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Stores the change of total energy with respect to the previous step.
        """,
        categories=[ErrorEstimateContribution, EnergyValue],
    )

    fermi = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Fermi energy (separates occupied from unoccupied single-particle states)
        """,
        categories=[EnergyTypeReference, EnergyValue],
    )

    highest_occupied = Quantity(
        type=np.dtype(np.float64),
        unit='joule',
        shape=[],
        description="""
        The highest occupied energy.
        """,
    )

    lowest_unoccupied = Quantity(
        type=np.dtype(np.float64),
        unit='joule',
        shape=[],
        description="""
        The lowest unoccupied energy.
        """,
    )

    kinetic = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the kinetic energy.
        """,
    )

    potential = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the potential energy.
        """,
    )

    pressure_volume_work = SubSection(
        sub_section=EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the instantaneous pV work.
        """,
    )

class ForcesEntry(Atomic):
    """
    Section describing a contribution to or type of atomic forces.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms', 3],
        unit='newton',
        description="""
        Value of the forces acting on the atoms. This is calculated as minus gradient of
        the corresponding energy type or contribution **including** constraints, if
        present. The derivatives with respect to displacements of nuclei are evaluated in
        Cartesian coordinates.  In addition, these are obtained by filtering out the
        unitary transformations (center-of-mass translations and rigid rotations for
        non-periodic systems, see value_raw for the unfiltered counterpart).
        """,
    )

    value_raw = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms', 3],
        unit='newton',
        description="""
        Value of the forces acting on the atoms **not including** such as fixed atoms,
        distances, angles, dihedrals, etc.""",
    )


class Forces(MSection):
    """
    Section containing all forces types and contributions.
    """

    m_def = Section(validate=False)

    total = SubSection(
        sub_section=ForcesEntry.m_def,
        description="""
        Contains the value and information regarding the total forces on the atoms
        calculated as minus gradient of energy_total.
        """,
    )

    free = SubSection(
        sub_section=ForcesEntry.m_def,
        description="""
        Contains the value and information regarding the forces on the atoms
        corresponding to the minus gradient of energy_free. The (electronic) energy_free
        contains the information on the change in (fractional) occupation of the
        electronic eigenstates, which are accounted for in the derivatives, yielding a
        truly energy-conserved quantity.
        """,
    )

    t0 = SubSection(
        sub_section=ForcesEntry.m_def,
        description="""
        Contains the value and information regarding the forces on the atoms
        corresponding to the minus gradient of energy_T0.
        """,
    )

    contributions = SubSection(
        sub_section=ForcesEntry.m_def,
        description="""
        Contains other forces contributions to the total atomic forces not already
        defined.
        """,
        repeats=True,
    )

    types = SubSection(
        sub_section=ForcesEntry.m_def,
        description="""
        Contains other types of forces not already defined.
        """,
        repeats=True,
    )


class StressEntry(Atomic):
    """
    Section describing a contribution to or a type of stress.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 3],
        unit='joule/meter**3',
        description="""
        Value of the stress on the simulation cell. It is given as the functional
        derivative of the corresponding energy with respect to the deformation tensor.
        """,
    )

    values_per_atom = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3, 3],
        unit='joule/meter**3',
        description="""
        Value of the atom-resolved stresses.
        """,
    )


class Stress(MSection):
    """
    Section containing all stress types and contributions.
    """

    m_def = Section(validate=False)

    total = SubSection(
        sub_section=StressEntry.m_def,
        description="""
        Contains the value and information regarding the stress on the simulation cell
        and the atomic stresses corresponding to energy_total.
        """,
    )

    contributions = SubSection(
        sub_section=StressEntry.m_def,
        description="""
        Contains contributions for the total stress.
        """,
        repeats=True,
    )

    types = SubSection(
        sub_section=StressEntry.m_def,
        description="""
        Contains other types of stress.
        """,
        repeats=True,
    )


class ChargesValue(AtomicValues):
    """
    Contains information on the charge on an atom or projected onto an orbital.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='coulomb',
        description="""
        Value of the charge projected on atom and orbital.
        """,
    )

    n_electrons = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Value of the number of electrons projected on atom and orbital.
        """,
    )

    spin_z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Value of the azimuthal spin projected on atom and orbital.
        """,
    )


class Charges(Atomic):
    """
    Section describing the charges on the atoms obtained through a given analysis
    method. Also contains information on the orbital projection of charges.
    """

    m_def = Section(validate=False)

    analysis_method = Quantity(
        type=str,
        shape=[],
        description="""
        Analysis method employed in evaluating the atom and partial charges.
        """,
    )

    value = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms'],
        unit='coulomb',
        description="""
        Value of the atomic charges calculated through analysis_method.
        """,
    )

    n_electrons = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms'],
        description="""
        Value of the number of electrons on the atoms.
        """,
    )

    # TODO should this be on a separate section magnetic_moments or charges should be
    # renamed population
    spins = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms'],
        description="""
        Value of the atomic spins.
        """,
    )

    total = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='coulomb',
        description="""
        Value of the total charge of the system.
        """,
    )

    spin_projected = SubSection(sub_section=ChargesValue.m_def, repeats=True)

    orbital_projected = SubSection(sub_section=ChargesValue.m_def, repeats=True)



class DosValues(AtomicValues):
    """
    Section containing information regarding the values of the density of states (DOS).
    """

    m_def = Section(validate=False)

    phonon_mode = Quantity(
        type=str,
        shape=[],
        description="""
        Phonon mode corresponding to the DOS used for phonon projections.
        """,
    )

    normalization_factor = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Normalization factor for DOS values to get a cell-independent intensive DOS,
        defined as the DOS integral from the lowest energy state to the Fermi level for a neutrally charged system.
        """,
    )

    value = Quantity(
        type=np.dtype(np.float64),
        shape=['n_energies'],
        unit='1/joule',
        description="""
        Values of DOS, i.e. number of states for a given energy. The set of discrete
        energy values is given in energies.
        """,
    )

    value_integrated = Quantity(
        type=np.dtype(np.float64),
        shape=['n_energies'],
        description="""
        A cumulative DOS starting from the mimunum energy available up to the energy level specified in `energies`.
        """,
    )


class Dos(Atomic):
    """
    Section containing information of an electronic-energy or phonon density of states
    (DOS) evaluation per spin channel.

    It includes the total DOS and the projected DOS values. We differentiate `species_projected` as the
    projected DOS for same atomic species, `atom_projected` as the projected DOS for different
    atoms in the cell, and `orbital_projected` as the projected DOS for the orbitals of each
    atom. These are hierarchically connected as:

        atom_projected = sum_{orbitals} orbital_projected

        species_projected = sum_{atoms} atom_projected

        total = sum_{species} species_projected
    """

    m_def = Section(validate=False)

    n_energies = Quantity(
        type=int,
        shape=[],
        description="""
        Gives the number of energy values for the DOS, see energies.
        """,
    )

    energies = Quantity(
        type=np.float64,
        shape=['n_energies'],
        unit='joule',
        description="""
        Contains the set of discrete energy values for the DOS.
        """,
    )

    energy_fermi = Quantity(
        type=np.float64,
        unit='joule',
        shape=[],
        description="""
        Fermi energy.
        """,
    )

    energy_ref = Quantity(
        type=np.float64,
        unit='joule',
        shape=[],
        description="""
        Energy level denoting the origin along the energy axis, used for comparison and visualization.
        It is defined as the energy_highest_occupied and does not necessarily coincide with energy_fermi.
        """,
    )

    spin_channel = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Spin channel of the corresponding DOS. It can take values of 0 or 1.
        """,
    )

    # TODO total is neither repeated, nor inheriting from AtomicValues; we have to change this when overhauling.
    total = SubSection(sub_section=DosValues.m_def, repeats=True)

    species_projected = SubSection(sub_section=DosValues.m_def, repeats=True)

    atom_projected = SubSection(sub_section=DosValues.m_def, repeats=True)

    orbital_projected = SubSection(sub_section=DosValues.m_def, repeats=True)

    fingerprint = SubSection(sub_section=DosFingerprint.m_def, repeats=False)

    # TODO deprecate this subsection
    band_gap = SubSection(sub_section=BandGapDeprecated.m_def, repeats=True)



class MultipolesValues(AtomicValues):
    """
    Section containing the values of the multipoles projected unto an atom or orbital.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Value of the multipole.
        """,
    )


class MultipolesEntry(Atomic):
    """
    Section describing a multipole term. The unit of the values are given by C * m ^ n,
    where n = 1 for dipole, 2 for quadrupole, etc.
    """

    m_def = Section(validate=False)

    origin = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        unit='meter',
        description="""
        Origin in cartesian space.
        """,
    )

    n_multipoles = Quantity(
        type=int,
        shape=[],
        description="""
        Number of multipoles.
        """,
    )

    value = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms', 'n_multipoles'],
        description="""
        Value of the multipoles projected unto the atoms.
        """,
    )

    total = Quantity(
        type=np.dtype(np.float64),
        shape=['n_multipoles'],
        description="""
        Total value of the multipoles.
        """,
    )

    orbital_projected = SubSection(sub_section=MultipolesValues.m_def, repeats=True)


class Enthalpy(ScalarProperty):
    """
    Section containing the enthalpy (i.e. energy_total + pressure * volume.) of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.value_unit = 'joule'

class Entropy(ScalarProperty):
    """
    Section containing the entropy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.value_unit = 'joule'

    entropy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule / kelvin',
        description="""
        Value of the entropy.
        """,
    )

    chemical_potential = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the chemical potential.
        """,
    )

    kinetic_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the kinetic energy.
        """,
    )

    potential_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the potential energy.
        """,
    )

    internal_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the internal energy.
        """,
    )

    vibrational_free_energy_at_constant_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the vibrational free energy per cell unit at constant volume.
        """,
    )

    pressure = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='pascal',
        description="""
        Value of the pressure of the system.
        """,
    )

    temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='kelvin',
        description="""
        Value of the temperature of the system at which the properties are calculated.
        """,
    )

    volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description="""
        Value of the volume of the system at which the properties are calculated.
        """,
    )

    heat_capacity_c_v = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule / kelvin',
        description="""
        Stores the heat capacity per cell unit at constant volume.
        """,
    )

    heat_capacity_c_p = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule / kelvin',
        description="""
        Stores the heat capacity per cell unit at constant pressure.
        """,
    )

    time_step = Quantity(
        type=int,
        shape=[],
        description="""
        The number of time steps with respect to the start of the calculation.
        """,
    )


class RadiusOfGyrationValues(AtomicGroupValues):
    """
    Section containing information regarding the values of
    radius of gyration (Rg).
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m',
        description="""
        Value of Rg.
        """,
    )


class RadiusOfGyration(AtomicGroup):
    """
    Section containing information about the calculation of
    radius of gyration (Rg).
    """

    m_def = Section(validate=False)

    radius_of_gyration_values = SubSection(
        sub_section=RadiusOfGyrationValues.m_def, repeats=True
    )

class BaseCalculation(ArchiveSection):
    """
    Contains computed properties of a configuration as defined by the corresponding
    section system and with the simulation method defined by section method. The
    references to the system and method sections are given by system_ref and method_ref,
    respectively.

    Properties derived from a group of configurations are not included in this section but
    can be accessed in section workflow.
    """

    m_def = Section(validate=False)

    system_ref = Quantity(
        type=Reference(System.m_def),
        shape=[],
        description="""
        Links the calculation to a section system.
        """,
        categories=[FastAccess],
    )

    method_ref = Quantity(
        type=Reference(Method.m_def),
        shape=[],
        description="""
        Links the calculation to a section method.
        """,
        categories=[FastAccess],
    )

    starting_calculation_ref = Quantity(
        type=Reference(SectionProxy('Calculation')),
        shape=[],
        description="""
        Links the current section calculation to the starting calculation.
        """,
        categories=[FastAccess],
    )

    n_references = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
         Number of references to the current section calculation.
        """,
    )

    calculations_ref = Quantity(
        type=Reference(SectionProxy('Calculation')),
        shape=['n_references'],
        description="""
        Links the current section calculation to other section calculations. Such a link
        is necessary for example if the referenced calculation is a self-consistent
        calculation that serves as a starting point or a calculation is part of a domain
        decomposed simulation that needs to be connected.
        """,
        categories=[FastAccess],
    )

    calculations_path = Quantity(
        type=str,
        shape=['n_references'],
        description="""
        Links the current section calculation to other section calculations. Such a link
        is necessary for example if the referenced calculation is a self-consistent
        calculation that serves as a starting point or a calculation is part of a domain
        decomposed simulation that needs to be connected.
        """,
    )

    calculation_converged = Quantity(
        type=bool,
        shape=[],
        description="""
        Indicates whether a the calculation is converged.
        """,
    )

    hessian_matrix = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 'number_of_atoms', 3, 3],
        description="""
        The matrix with the second derivative of the energy with respect to atom
        displacements.
        """,
    )

    spin_S2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Stores the value of the total spin moment operator $S^2$ for the converged
        wavefunctions calculated with the XC_method. It can be used to calculate the spin
        contamination in spin-unrestricted calculations.
        """,
    )

    time_calculation = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description="""
        Stores the wall-clock time needed to complete the calculation i.e. the real time
        that has elapsed from start to end of calculation.
        """,
        categories=[TimeInfo, AccessoryInfo],
    )

    time_physical = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description="""
        The elapsed real time at the end of the calculation with respect to the start of
        the simulation.
        """,
    )

    energy = SubSection(sub_section=Energy.m_def, categories=[FastAccess])

    forces = SubSection(sub_section=Forces.m_def)

    stress = SubSection(sub_section=Stress.m_def)

    band_gap = SubSection(sub_section=BandGap.m_def, repeats=True)

    dos_electronic = SubSection(sub_section=Dos.m_def, repeats=True)

    dos_phonon = SubSection(sub_section=Dos.m_def, repeats=True)

    eigenvalues = SubSection(sub_section=BandEnergies.m_def, repeats=True)

    band_structure_electronic = SubSection(
        sub_section=BandStructure.m_def, repeats=True
    )

    band_structure_phonon = SubSection(sub_section=BandStructure.m_def, repeats=True)

    thermodynamics = SubSection(sub_section=Thermodynamics.m_def, repeats=True)

    hopping_matrix = SubSection(sub_section=HoppingMatrix.m_def, repeats=True)

    spectra = SubSection(sub_section=Spectra.m_def, repeats=True)

    greens_functions = SubSection(sub_section=GreensFunctions.m_def, repeats=True)

    vibrational_frequencies = SubSection(
        sub_section=VibrationalFrequencies.m_def, repeats=True
    )

    potential = SubSection(sub_section=Potential.m_def, repeats=True)

    multipoles = SubSection(sub_section=Multipoles.m_def, repeats=True)

    charges = SubSection(sub_section=Charges.m_def, repeats=True)

    density_charge = SubSection(sub_section=Density.m_def, repeats=True)

    radius_of_gyration = SubSection(sub_section=RadiusOfGyration.m_def, repeats=True)

    magnetic_shielding = SubSection(sub_section=MagneticShielding.m_def, repeats=True)

    electric_field_gradient = SubSection(
        sub_section=ElectricFieldGradient.m_def, repeats=True
    )

    spin_spin_coupling = SubSection(sub_section=SpinSpinCoupling.m_def, repeats=True)

    magnetic_susceptibility = SubSection(
        sub_section=MagneticSusceptibility.m_def, repeats=True
    )

    volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description="""
        Value of the volume of the system.
        """,
    )

    density = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='kg / m ** 3',
        description="""
        Value of the density of the system.
        """,
    )

    pressure = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Value of the pressure of the system.
        """,
    )

    pressure_tensor = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='pascal',
        description="""
        Value of the pressure in terms of the x, y, z components of the simulation cell.
        Typically calculated as the difference between the kinetic energy and the virial.
        """,
    )

    virial_tensor = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 3],
        unit='joule',
        description="""
        Value of the virial in terms of the x, y, z components of the simulation cell.
        Typically calculated as the cross product between positions and forces.
        """,
    )

    enthalpy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the calculated enthalpy per cell i.e. energy_total + pressure * volume.
        """,
    )

    temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='kelvin',
        description="""
        Value of the temperature of the system.
        """,
    )

    step = Quantity(
        type=int,
        shape=[],
        description="""
        The number of time steps with respect to the start of the simulation.
        """,
    )

    time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description="""
        The elapsed simulated physical time since the start of the simulation.
        """,
    )


class ScfIteration(BaseCalculation):
    """
    Every scf_iteration section represents a self-consistent field (SCF) iteration,
    and gives detailed information on the SCF procedure of the specified quantities.
    """

    m_def = Section(validate=False)


class Calculation(BaseCalculation):
    """
    Every calculation section contains the values computed
    during a *single configuration calculation*, i.e. a calculation performed on a given
    configuration of the system (as defined in section_system) and a given computational
    method (e.g., exchange-correlation method, basis sets, as defined in section_method).

    The link between the current section calculation and the related
    system and method sections is established by the values stored in system_ref and
    method_ref, respectively.

    The reason why information on the system configuration and computational method is
    stored separately is that several *single configuration calculations* can be performed
    on the same system configuration, viz. several system configurations can be evaluated
    with the same computational method. This storage strategy avoids redundancies.
    """

    m_def = Section(validate=False)

    n_scf_iterations = Quantity(
        type=int,
        shape=[],
        description="""
        Gives the number of performed self-consistent field (SCF) iterations.
        """,
        categories=[ScfInfo],
    )

    scf_iteration = SubSection(sub_section=ScfIteration.m_def, repeats=True)
