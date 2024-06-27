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
from nomad_simulations.schema_packages.physical_property import PhysicalProperty
from nomad_simulations.schema_packages.properties import BaseEnergy


class Enthalpy(BaseEnergy):
    """
    Physical property section describing the enthalpy (i.e. energy_total + pressure * volume.) of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Entropy(PhysicalProperty):
    """
    Physical property section describing the entropy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule / kelvin',
        description="""
        The value of the entropy.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ChemicalPotential(BaseEnergy):
    """
    Free energy cost of adding or extracting a particle from a thermodynamic system.
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    iri = 'http://fairmat-nfdi.eu/taxonomy/ChemicalPotential'

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = []
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Pressure(PhysicalProperty):
    """
    Physical property section describing the pressure of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='pascal',
        description="""
        The value of the pressure.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Virial(BaseEnergy):
    """
    Physical property section describing the virial (cross product between positions and forces) of a (sub)system.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Temperature(PhysicalProperty):
    """
    Physical property section describing the temperature of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='kelvin',
        description="""
        The value of the temperature.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Volume(PhysicalProperty):
    """
    Physical property section describing the volume of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='m ** 3',
        description="""
        The value of the volume.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Density(PhysicalProperty):
    """
    Physical property section describing the density of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='kg / m ** 3',
        description="""
        The value of the density.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# ? Does this go here or in energies?
# ? Naming specific to Potential Energy?
class Hessian(PhysicalProperty):
    """
    Physical property section describing the Hessian matrix, i.e., 2nd derivatives with respect to geometric (typically particle) displacements,
    of the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule / m ** 2',
        description="""
        The value of the Hessian.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class HeatCapacityCV(PhysicalProperty):
    """
    Physical property section describing the heat capacity at constant volume for a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule / kelvin',
        description="""
        The value of the heat capacity.
        """,
    )


class HeatCapacityCP(PhysicalProperty):
    """
    Physical property section describing the heat capacity at constant volume for a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule / kelvin',
        description="""
        The value of the heat capacity.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

    # ? Is this ever used?
    # internal_energy = Quantity(
    #     type=np.dtype(np.float64),
    #     shape=[],
    #     unit='joule',
    #     description="""
    #     Value of the internal energy.
    #     """,
    # )

    # vibrational_free_energy_at_constant_volume = Quantity(
    #     type=np.dtype(np.float64),
    #     shape=[],
    #     unit='joule',
    #     description="""
    #     Value of the vibrational free energy per cell unit at constant volume.
    #     """,
    # )
