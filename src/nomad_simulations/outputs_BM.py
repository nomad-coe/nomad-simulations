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
import numpy as np

from typing import Optional
from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity, SubSection, SectionProxy, Reference, MEnum

from .atoms_state import AtomsState, OrbitalsState
from .model_system import ModelSystem
from .numerical_settings import SelfConsistency

from nomad.units import ureg
from nomad.metainfo.metainfo import DirectQuantity, Dimension

from .outputs import Outputs
from .property import BaseProperty


class Foo:
    foo = 'Bernadette does not know what can be inherited from where yet.'


# TODO: Are common constants (e.g. Boltzmann constant) defined somewhere?


class SimulationEntries(Foo):
    # TODO: get temperature, pressure, volume, etc. From where?
    #       Only target/end values, or also values per snapshot?
    value = Quantity(
        type=np.float64,
        unit='joule',  # Do we have a default unit?
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.value_unit = 'joule'

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
        Value of the temperature of the system.
        """,
    )

    volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description="""
        Value of the simulation box volume.
        """,
    )


class StructureEntries(Foo):
    # TODO: Get atom types, charges, positions, groups, bonds, angles, dihedrals, etc.
    # TODO: Where, what format?
    value = Quantity(
        type=np.float64,
        unit='joule',  # Do we have a default unit?
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.value_unit = 'joule'


# TODO: What is there via the parsers, where is it stored, how is it accessed?
class EnergyEntries(Foo):
    value = Quantity(
        type=np.float64,
        unit='joule',
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.value_unit = 'joule'

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

    # Individual contributions to the potential energy, as provided by the engine
    electrostatic_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the electrostatic energy.
        """,
    )

    vdw_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the van der Waals energy.
        """,
    )

    chemical_potential = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the chemical energy.
        """,
    )

    total_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description="""
        Value of the internal energy.
        """,
    )


class AtomGroup(StructureEntries):
    group = Quantity(
        type=list(),  # ?!
        shape=['*'],
        unit=None,
        description="""
        Selected group of atoms.
        """,
    )


# TODO: rename to something that doesn't incurr the wrath of the StatMech gods.
class ThermodynamicProperties(EnergyEntries, SimulationEntries):
    value = Quantity(
        type=np.float64,
        unit='joule',
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.value_unit = 'joule'

    # TODO: For each individual snapshot, and ensemble average
    # if provided (e.g. read from GMX edr file):
    #     retrieve from archive
    # else:
    #    calculate
    # TODO: need total energy, pressure and box Volume from EnergyEntries and
    #       SimulationEntries
    total_enthalpy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        Value of the average enthalpy over the entire trajectory.
        """,
    )

    snapshot_enthalpy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        Values of the enthalpy for each snapshot.
        """,
    )

    # Does every MD engine return entropy, or do we need to optionally calculate it?
    # Configurational entropy, 2PT, solvation entropy... ?
    entropy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule / kelvin',
        description="""
        Configurational entropy of the system.
        """,
    )

    # TODO: $C_v = \frac{1}{k_BT^2}((\langle E^2\rangle)-(\langle E\rangle)^2)$
    #       Nedds total energy from EnergyEntries
    heat_capacity_volume = Quantity(
        type=np.float64,
        shape=[],
        unit='joule / (kg * kelvin)',
        description="""
        Heat capacity at constant volume.
        """,
    )

    heat_capacity_pressure = Quantity(
        type=np.float64,
        shape=[],
        unit='joule / (kg * kelvin)',
        description="""
        Heat capacity at constant pressure.
        """,
    )


class StructuralProperties(AtomGroup, SimulationEntries):
    # Solvation shell
    # Maybe monitor Angles/Dihedrals => deviation from FF defined values (<- where?)
    # For complexes: contact map? How would we decide when to populate?

    radius_of_gyration = Quantity(
        type=np.float64,
        shape=[],
        unit='m',
        description="""
        Value of g(r).
        """,
    )

    # TODO: $N_c = 4\pi\rho \int_0^r_cutoff r^2 g(r) dr$, if we can define the number of
    #       particles per unit volume (rho).
    coordination_number = Quantity(
        type=np.float64,
        shape=[],
        unit=None,
        description="""
        Value of the coordination number.
        """,
    )

    # Can be considered redundant with coordination number, not sure.
    solvation_shell = Quantity(
        # TODO: Shell boundaries: first peak and first minium of g(r).
        #       Count solvent molecules within shell boundaries.
        n_solvent_molecules=Quantity(
            type=np.int32,
            shape=['n_atoms' / 'n_atoms_per_molecule'],  # TODO: Does this make sense?
            unit=None,
            description="""
            Number of solvent molecules in the solvation shell.
            """,
        )
        # Maybe return solvent molecules within shell boundaries to select for further
        # analysis?
    )

    mean_square_displacement = Quantity(
        type=np.float64,
        shape=[],
        unit='m ** 2',
        description="""
        Values of the mean square displacement.
        """,
    )


class MaterialProperties(SimulationEntries):
    # TODO: Needs box volume and temperature throughout simulation entries

    #       $\kappa_T$ = \frac{\langle V^2\rangle - \langle V\rangle^2}{V}$
    isothermal_compressibility = Quantity(
        type=np.float64,
        shape=[],
        unit='1 / pascal',
        description="""
        Value of the isothermal compressibility.
        """,
    )

    # TODO: Only makes sense if we have simulations of the same system at different
    #       temperatures. May go beyond of what we want to do here.
    #       $\alpha \approx \frac{\frac{\Delta V}{V}}{\Delta T}$
    thermal_expansivity = Quantity(
        type=np.float64,
        shape=[],
        unit='1 / kelvin',
        description="""
        Value of the thermal expansivity.
        """,
    )

    # TODO: Green-Kubo method or Einstein relation?
    #       Scalar or tensor?
    viscosity = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal * s',
        description="""
        Value of the viscosity.
        """,
    )

    # TODO: Can be calculated based on the mean square displacement.
    #       $D = \frac{1}{6} \langle\Delta r^2\rangle$
    #       Should be tensor for (semi)isotropic systems.
    diffusion_coefficient = Quantity(
        type=np.float64,
        shape=[],
        unit='m ** 2 / s',
        description="""
        Value of the diffusion coefficient.
        """,
    )
