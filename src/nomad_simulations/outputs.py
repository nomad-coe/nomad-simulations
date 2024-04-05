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

import numpy as np
from typing import Any
from structlog.stdlib import BoundLogger

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity, SubSection, MEnum
from nomad.datamodel.metainfo.annotations import ELNAnnotation

from .model_system import ModelSystem
from .numerical_settings import SelfConsistency
from .physical_property import PhysicalProperty

from .variables import Variables, Temperatures  # ? delete these imports


class ElectronicBandGap(PhysicalProperty):
    """ """

    shape = [3]

    type = Quantity(
        type=MEnum('direct', 'indirect'),
        description="""
        Type categorization of the electronic band gap. The electronic band gap can be `'direct'` or `'indirect'`.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the electronic band gap.
        """,
    )

    # Add more functionalities here

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Outputs(ArchiveSection):
    """
    Output properties of a simulation. This base class can be used for inheritance in any of the output properties
    defined in this schema.

    It contains references to the specific sections used to obtain the output properties, as well as
    information if the output `is_derived` from another output section or directly parsed from the simulation output files.
    """

    # TODO add time quantities

    normalizer_level = 2

    model_system_ref = Quantity(
        type=ModelSystem,
        description="""
        Reference to the `ModelSystem` section to which the output property references to and on
        on which the simulation is performed.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    # # # # # # # # # #
    # List of properties

    electronic_band_gap = SubSection(sub_section=ElectronicBandGap.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve if the output property `is_derived` or not.
        # self.is_derived = self.resolve_is_derived(self.outputs_ref)


class SCFOutputs(Outputs):
    """
    This section contains the self-consistent (SCF) steps performed to converge an output property,
    as well as the information if the output property `is_converged` or not, depending on the
    settings in the `SelfConsistency` base class defined in `numerical_settings.py`.

    For simplicity, we contain the SCF steps of a simulation as part of the minimal workflow defined in NOMAD,
    the `SinglePoint`, i.e., we do not split each SCF step in its own entry. Thus, each `SinglePoint`
    `Simulation` entry in NOMAD contains the final output properties and all the SCF steps.
    """

    scf_steps = SubSection(
        sub_section=Outputs.m_def,
        repeats=True,
        description="""
        Self-consistent (SCF) steps performed for converging a given output property. Note that the SCF steps belong to
        the same minimal `Simulation` workflow entry which is known as `SinglePoint`.
        """,
    )

    is_scf_converged = Quantity(
        type=bool,
        description="""
        Flag indicating whether the output property is converged or not after a SCF process. This quantity is connected
        with `SelfConsistency` defined in the `numerical_settings.py` module.
        """,
    )

    self_consistency_ref = Quantity(
        type=SelfConsistency,
        description="""
        Reference to the `SelfConsistency` section that defines the numerical settings to converge the
        output property.
        """,
    )

    # TODO add more functionality to automatically check convergence from `self_consistency_ref` and the last two `scf_steps`

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# Playing with `PhysicalProperty`
band_gap = ElectronicBandGap(source='simulation', type='direct', label='DFT')
n_bins = 3
temperature = Temperatures(n_bins=n_bins, bins=np.linspace(0, 100, n_bins))
band_gap.variables.append(temperature)
n_bins = 2
custom_bins = Variables(n_bins=n_bins, bins=np.linspace(0, 100, n_bins))
band_gap.variables.append(custom_bins)
# band_gap.value_unit = 'joule'
band_gap.value = [
    [[1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3]],
] * ureg.eV
# band_gap.value = [1, 2, 3] * ureg.eV
print(band_gap)
