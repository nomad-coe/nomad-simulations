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

from typing import TYPE_CHECKING

import numpy as np

from nomad.metainfo import Quantity, Section, Context, SubSection, MEnum

if TYPE_CHECKING:
    from nomad.metainfo import Section, Context
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import PhysicalProperty


class FermiLevel(PhysicalProperty):
    """
    Energy required to add or extract a charge from a material at zero temperature. It can be also defined as the chemical potential at zero temperature.
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    iri = 'http://fairmat-nfdi.eu/taxonomy/FermiLevel'

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the Fermi level.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = []
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ChemicalPotential(PhysicalProperty):
    """
    Free energy cost of adding or extracting a particle from a thermodynamic system.
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    iri = 'http://fairmat-nfdi.eu/taxonomy/ChemicalPotential'

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the chemical potential.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = []
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


####################################################
# Abstract energy classes
####################################################


class Energy(PhysicalProperty):
    """
    Abstract physical property section describing some energy of a (sub)system.
    """

    type = Quantity(
        type=MEnum('classical', 'quantum'),
        description="""
        Refers to the method used for calculating the energy.

        Allowed values are:

        | Energy Type            | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"classical"`          |  The energy is determined via a classical mechanics formalism. |

        | `"quantum"`            |  The energy is determined via a quantum mechanics formalism. |
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ClassicalEnergy(Energy):
    """
    Abstract physical property section describing some classical energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        if not self.type:
            self.type == 'classical'
        elif self.type != 'classical':
            logger.error(f'Misidentified type for classical energy.')


class QuantumEnergy(Energy):
    """
    Abstract physical property section describing some quantum energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        if not self.type:
            self.type == 'quantum'
        elif self.type != 'quantum':
            logger.error(f'Misidentified type for quantum energy.')


######################################################
# List of general energy properties/contributions that
# can have both classical and quantum interpretations
######################################################


class TotalEnergy(Energy):
    """
    Physical property section describing the total energy of a (sub)system.
    """

    contributions = SubSection(sub_section=Energy.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class KineticEnergy(Energy):
    """
    Physical property section describing the kinetic energy of a (sub)system.
    """

    type = Quantity(
        type=MEnum('classical', 'quantum'),
        description="""
        Refers to the method used for calculating the kinetic energy.

        Allowed values are:

        | Energy Type            | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"classical"`                   |   The kinetic energy is calculated directly from
        the velocities of particles as KE = /sum_i 1/2 m_i v_i^2, where the sum runs over the number of particles in the system,
        m_i is the mass of particle i, and v_i is the velocity of particle i.          |

        | `"quantum"`           |  ... |
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


####################################################
# List of classical energy contribuions
####################################################


class PotentialEnergy(ClassicalEnergy):
    """
    Physical property section describing the potential energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class IntermolecularEnergy(ClassicalEnergy):
    """
    Physical property section describing all intramolecular contributions to the potential energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class VDWEnergy(ClassicalEnergy):
    """
    Physical property section describing the van der Waals contributions to the potential energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ElectrostaticEnergy(ClassicalEnergy):
    """
    Physical property section describing all electrostatic contributions to the potential energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ElectrostaticShortRangeEnergy(ClassicalEnergy):
    """
    Physical property section describing short-range electrostatic contributions to the potential energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ElectrostaticLongRangeEnergy(ClassicalEnergy):
    """
    Physical property section describing long-range electrostatic contributions to the potential energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class IntramolecularEnergy(ClassicalEnergy):
    """
    Physical property section describing all intramolecular contributions to the potential energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class BondEnergy(ClassicalEnergy):
    """
    Physical property section describing contributions to the potential energy from bond interactions of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class AngleEnergy(ClassicalEnergy):
    """
    Physical property section describing contributions to the potential energy from angle interactions of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class DihedralEnergy(ClassicalEnergy):
    """
    Physical property section describing contributions to the potential energy from dihedral interactions of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ImproperDihedralEnergy(ClassicalEnergy):
    """
    Physical property section describing contributions to the potential energy from improper dihedral interactions of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ExternalEnergy(ClassicalEnergy):
    """
    Physical property section describing contributions to the potential energy from external interactions of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


######################################
# List of quantum energy contributions
######################################


class ElectronicEnergy(QuantumEnergy):
    """
    Physical property section describing the electronic energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


#! allowed_contributions = ['PotKin'] Not sure how to deal with sub-contributions... - I lean towards keeping a flat list but I am not sure of all the usages


class ElectronicKineticEnergy(QuantumEnergy):
    """
    Physical property section describing the electronic kinetic energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class XCEnergy(QuantumEnergy):
    """
    Physical property section describing the exchange-correlation (XC) energy of a (sub)system,
    calculated using the functional stored in XC_functional.
    """

    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class XCPotentialEnergy(QuantumEnergy):
    """
    Physical property section describing the potential energy contribution to the exchange-correlation (XC) energy,
    i.e., the integral of the first order derivative of the functional
        stored in XC_functional (integral of v_xc*electron_density), i.e., the component
        of XC that is in the sum of the eigenvalues. Value associated with the
        configuration, should be the most converged value.          |
    """

    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# ? XCCorrelationEnergy?
class CorrelationEnergy(QuantumEnergy):
    """
    Physical property section describing the correlation energy of a (sub)system,
    calculated using the method described in XC_functional.
    """

    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the correlation energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# ? XCExchangeEnergy?
class ExchangeEnergy(QuantumEnergy):
    """
    Physical property section describing the exchange energy of a (sub)system,
    calculated using the method described in XC_functional.
    """

    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ZeroTemperatureEnergy(QuantumEnergy):
    """
    Physical property section describing the total energy of a (sub)system extrapolated to $T=0$, based on a free-electron gas argument.
    """

    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ZeroPointEnergy(QuantumEnergy):
    """
    Physical property section describing the zero-point vibrational energy of a (sub)system,
    calculated using the method described in zero_point_method.
    """

    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ElectrostaticEnergy(QuantumEnergy):
    """
    Physical property section describing the electrostatic energy (nuclei + electrons) of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class NuclearRepulsionEnergy(QuantumEnergy):
    """
    Physical property section describing the nuclear-nuclear repulsion energy of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


##########################
# Other / General energies
##########################


# madelung = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the Madelung energy.
#     """,
# )

# # TODO I suggest ewald is moved to "long range" under electrostatic->energyentry, unless there is some other usage I am misunderstanding
# ewald = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the Ewald energy.
#     """,
# )

# free = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the free energy (nuclei + electrons)
#     (whose minimum gives the smeared occupation density calculated with
#     smearing_kind).
#     """,
# )

# sum_eigenvalues = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the sum of the eigenvalues of the
#     Hamiltonian matrix.
#     """,
# )

# van_der_waals = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the Van der Waals energy. A multiple
#     occurence is expected when more than one van der Waals methods are defined. The
#     van der Waals kind should be specified in Energy.kind
#     """,
# )

# hartree_fock_x_scaled = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Scaled exact-exchange energy that depends on the mixing parameter of the
#     functional. For example in hybrid functionals, the exchange energy is given as a
#     linear combination of exact-energy and exchange energy of an approximate DFT
#     functional; the exact exchange energy multiplied by the mixing coefficient of the
#     hybrid functional would be stored in this metadata. Defined consistently with
#     XC_method.
#     """,
# )

# ? This is technically NOT redundant with total energy, but I fear that people will use them interchangeably, so we need to be clear about the definitions in any case
# internal = Quantity(
#     type=np.dtype(np.float64),
#     shape=[],
#     unit='joule',
#     description="""
#     Value of the internal energy.
#     """,
# )

# double_counting = SubSection(
#     sub_section=EnergyEntry.m_def,
#     categories=[FastAccess],
#     description="""
#     Double counting correction when performing Hubbard model calculations.
#     """,
# )

# # TODO remove this should be be entropy.correction
# correction_entropy = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Entropy correction to the potential energy to compensate for the change in
#     occupation so that forces at finite T do not need to keep the change of occupation
#     in account. Defined consistently with XC_method.
#     """,
# )

# # TODO remove this should be in electrostatic.correction
# correction_hartree = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Correction to the density-density electrostatic energy in the sum of eigenvalues
#     (that uses the mixed density on one side), and the fully consistent density-
#     density electrostatic energy. Defined consistently with XC_method.
#     """,
# )

# # TODO remove this should be in xc.correction
# correction_xc = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Correction to energy_XC.
#     """,
# )

# # ? Is it ok to store this in outputs and not in workflow? I guess we can ensure in normalization that this is a WorkflowOutput, etc...?
# change = Quantity(
#     type=np.dtype(np.float64),
#     shape=[],
#     unit='joule',
#     description="""
#     Stores the change of total energy with respect to the previous step.
#     """,
#     categories=[ErrorEstimateContribution, EnergyValue],
# )

# fermi = Quantity(
#     type=np.dtype(np.float64),
#     shape=[],
#     unit='joule',
#     description="""
#     Fermi energy (separates occupied from unoccupied single-particle states)
#     """,
#     categories=[EnergyTypeReference, EnergyValue],
# )

# highest_occupied = Quantity(
#     type=np.dtype(np.float64),
#     unit='joule',
#     shape=[],
#     description="""
#     The highest occupied energy.
#     """,
# )

# lowest_unoccupied = Quantity(
#     type=np.dtype(np.float64),
#     unit='joule',
#     shape=[],
#     description="""
#     The lowest unoccupied energy.
#     """,
# )

# # TODO this should be removed and replaced by correction in EnergyEntry
# current = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the energy calculated with
#     calculation_method_current. energy_current is equal to energy_total for
#     non-perturbative methods. For perturbative methods, energy_current is equal to the
#     correction: energy_total minus energy_total of the calculation_to_calculation_ref
#     with calculation_to_calculation_kind = starting_point
#     """,
# )
