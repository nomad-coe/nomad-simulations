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

import numpy as np
from nomad.metainfo import Quantity, SubSection, MEnum
from .physical_property import PhysicalProperty



class EnergyContributions(PhysicalProperty):
    """
    Base section for all contribution groups to the total energy.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

####################################################
# List of classical energy contribuions
####################################################

class PotentialEnergy(PhysicalProperty):
    """
    Section containing the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the potential energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class KineticEnergy(PhysicalProperty):
    """
    Section containing the kinetic energy of a (sub)system.
    """

    type = Quantity(
        type=MEnum('classical', 'quantum'),
        description="""
        Refers to the method used for calculating the kinetic energy.

        Allowed values are:

        | Thermostat Name        | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"classical"`                   |   The kinetic energy is calculated directly from
        the velocities of particles as KE = /sum_i 1/2 m_i v_i^2, where the sum runs over the number of particles in the system,
        m_i is the mass of particle i, and v_i is the velocity of particle i.          |

        | `"quantum"`           |  ... |
        """,
    )
    # ? Just an idea, not sure if this is necessary since we will be referencing the method, I guess maybe it's overkill in general
    # ? If we do this, maybe we should have a general Energy(PP) class which is inherited from in case there are multiple interpretations/calculation methods

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the kinetic energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class VDWEnergy(PhysicalProperty):
    """
    Section containing the van der Waals contributions to the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of the van der Waals contributions to the potential energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ElectrostaticEnergy(PhysicalProperty):
    """
    Section containing all electrostatic contributions to the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of all electrostatic contributions to the potential energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ElectrostaticShortRangeEnergy(PhysicalProperty):
    """
    Section containing short-range electrostatic contributions to the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of the short-range electrostatic contributions to the potential energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ElectrostaticLongRangeEnergy(PhysicalProperty):
    """
    Section containing long-range electrostatic contributions to the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of the long-range electrostatic contributions to the potential energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class BondedEnergy(PhysicalProperty):
    """
    Section containing all bonded (i.e., intramolecular) contributions to the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of all bonded (i.e., intramolecular) contributions to the potential energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class BondEnergy(PhysicalProperty):
    """
    Section containing contributions to the potential energy from bond interactions of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of contributions to the potential energy from bond interactions.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class AngleEnergy(PhysicalProperty):
    """
    Section containing contributions to the potential energy from angle interactions of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of contributions to the potential energy from angle interactions.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class DihedralEnergy(PhysicalProperty):
    """
    Section containing contributions to the potential energy from dihedral interactions of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of contributions to the potential energy from dihedral interactions.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ImproperDihedralEnergy(PhysicalProperty):
    """
    Section containing contributions to the potential energy from improper dihedral interactions of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of contributions to the potential energy from improper dihedral interactions.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ExternalEnergy(PhysicalProperty):
    """
    Section containing contributions to the potential energy from external interactions of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of contributions to the potential energy from external interactions.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class EnergyContributionsClassical(EnergyContributions):
    """
    Section containing contributions to the potential energy from a classical force field.
    """

    potential = SubSection(sub_section=PotentialEnergy.m_def, repeats=False)

    kinetic = SubSection(sub_section=KineticEnergy.m_def, repeats=False)

    vdw = SubSection(sub_section=VDWEnergy.m_def, repeats=False)

    electrostatic = SubSection(sub_section=ElectrostaticEnergy.m_def, repeats=False)

    electrostatic_short_range = SubSection(sub_section=ElectrostaticShortRangeEnergy.m_def, repeats=False)

    electrostatic_long_range = SubSection(sub_section=ElectrostaticLongRangeEnergy.m_def, repeats=False)

    bonded = SubSection(sub_section=BondedEnergy.m_def, repeats=False)

    bond = SubSection(sub_section=BondEnergy.m_def, repeats=False)

    angle = SubSection(sub_section=AngleEnergy.m_def, repeats=False)

    dihedral = SubSection(sub_section=DihedralEnergy.m_def, repeats=False)

    improper_dihedral = SubSection(sub_section=ImproperDihedralEnergy.m_def, repeats=False)

    external = SubSection(sub_section=ExternalEnergy.m_def, repeats=False)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        # Set the name of the section
        self.name = self.m_def.name

######################################
# List of quantum energy contributions
######################################

class ElectronicEnergy(PhysicalProperty):
    """
    Section containing the electronic energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the electronic energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ElectronicKineticEnergy(PhysicalProperty):
    """
    Section containing the electronic kinetic energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the electronic kinetic energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class XCEnergy(PhysicalProperty):
    """
    Section containing the exchange-correlation (XC) energy of a (sub)system,
    calculated using the functional stored in XC_functional.
    """
    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the exchange-correlation energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class XCPotentialEnergy(PhysicalProperty):
    """
    Section containing the potential energy contribution to the exchange-correlation (XC) energy,
    i.e., the integral of the first order derivative of the functional
        stored in XC_functional (integral of v_xc*electron_density), i.e., the component
        of XC that is in the sum of the eigenvalues. Value associated with the
        configuration, should be the most converged value.          |
    """
    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the potential energy contribution of the exchange-correlation energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

# ? XCCorrelationEnergy?
class CorrelationEnergy(PhysicalProperty):
    """
    Section containing the correlation energy of a (sub)system,
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
class ExchangeEnergy(PhysicalProperty):
    """
    Section containing the exchange energy of a (sub)system,
    calculated using the method described in XC_functional.
    """
    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the exchange energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class T0Energy(PhysicalProperty):
    """
    Section containing the total energy of a (sub)system extrapolated to $T=0$.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the total energy extrapolated to
        $T=0$, based on a free-electron gas argument.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ZeroPointEnergy(PhysicalProperty):
    """
    Section containing the zero-point vibrational energy of a (sub)system,
    calculated using the method described in zero_point_method.
    """
    # ! Someone check this description!
    # ? Do we really want to specify the method here? This can't be user-defined?

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the zero-point energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ElectrostaticEnergy(PhysicalProperty):
    """
    Section containing the electrostatic energy (nuclei + electrons) of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the electrostatic energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class NuclearRepulsionEnergy(PhysicalProperty):
    """
    Section containing the nuclear-nuclear repulsion energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the nuclear-nuclear repulsion energy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class EnergyTotalContributionsQuantum(EnergyContributions):
    """
    Section containing contributions to the potential energy from a DFT calculation.
    """
...

    electronic = SubSection(sub_section=ElectronicEnergy.m_def, repeats=False)

    electronic_kinetic = SubSection(sub_section=ElectronicKineticEnergy.m_def, repeats=False)

    xc = SubSection(sub_section=XCEnergy.m_def, repeats=False)

    xc_potential = SubSection(sub_section=XCPotentialEnergy.m_def, repeats=False)

    correlation = SubSection(sub_section=CorrelationEnergy.m_def, repeats=False)

    exchange = SubSection(sub_section=ExchangeEnergy.m_def, repeats=False)

    t0 = SubSection(sub_section=T0Energy.m_def, repeats=False)

    zero_point = SubSection(sub_section=ZeroPointEnergy.m_def, repeats=False)

    electrostatic = SubSection(sub_section=ElectrostaticEnergy.m_def, repeats=False)

    nuclear_repulsion = SubSection(sub_section=NuclearRepulsionEnergy.m_def, repeats=False)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

##########################
# Other / General energies
##########################

class EnergyTotal(PhysicalProperty):
    """
    Section containing the total energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the total energy.
        """,
    )
    # ? Do we need these descriptions under value? It ends up simply duplicating the section info to some extent.

    # ? I think we want to allow some flexible to the contributions here, like so: ??
    contributions = SubSection(sub_section=EnergyContributions.m_def, repeats=True)
    # classical_contributions = SubSection(sub_section=EnergyContributionsClassical.m_def, repeats=False)
    # quantum_contributions = SubSection(sub_section=EnergyTotalContributionsQuantum.m_def, repeats=False)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

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




# # ? Do we want to allow this?
# class EnergyCustom(PhysicalProperty):
#     """
#     Section containing the total energy of a (sub)system.
#     """

#     name = Quantity(
#         type=str,
#         description="""
#         The name of this custom energy.
#         """,
#     )

#     value = Quantity(
#         type=np.float64,
#         unit='joule',
#         description="""
#         The value of this custom energy type.
#         """,
#     )

#     def normalize(self, archive, logger) -> None:
#         super().normalize(archive, logger)

