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
from typing import Optional
from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity, SubSection, SectionProxy, Reference

from .atoms_state import AtomsState, OrbitalsState
from .model_system import ModelSystem
from .numerical_settings import SelfConsistency


class BaseOutputs(ArchiveSection):
    """
    Base section to define the output properties of a simulation. This is used as a placeholder for both
    final `Outputs` properties and the self-consistent (SCF) steps, see `Outputs` base section definition.
    """

    normalizer_level = 2

    name = Quantity(
        type=str,
        description="""
        Name of the output property. This is used for easier identification of the property and is conneceted
        with the class name of each output property class, e.g., `'ElectronicBandGap'`, `'ElectronicBandStructure'`, etc.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    orbitals_state_ref = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the `OrbitalsState` section on which the simulation is performed and the
        output property is calculated.
        """,
    )

    atoms_state_ref = Quantity(
        type=AtomsState,
        description="""
        Reference to the `AtomsState` section on which the simulation is performed and the
        output property is calculated.
        """,
    )

    model_system_ref = Quantity(
        type=ModelSystem,
        description="""
        Reference to the `ModelSystem` section on which the simulation is performed and the
        output property is calculated.
        """,
    )

    is_derived = Quantity(
        type=bool,
        default=False,
        description="""
        Flag indicating whether the output property is derived from other output properties. We make
        the distinction between directly parsed, derived, and post-processing output properties:
            - Directly parsed: the output property is directly parsed from the simulation output files.
            - Derived: the output property is derived from other output properties. No extra numerical settings
                are required to calculate the output property.
            - Post-processing: the output property is derived from other output properties. Extra numerical settings
                are required to calculate the output property.
        """,
    )

    outputs_ref = Quantity(
        type=Reference(SectionProxy('BaseOutputs')),
        description="""
        Reference to the `BaseOutputs` section from which the output property was derived. This is only
        relevant if `is_derived` is set to True.
        """,
    )

    def check_is_derived(self, is_derived: bool, outputs_ref) -> Optional[bool]:
        if not is_derived:
            if outputs_ref is not None:
                return True
            return False
        elif is_derived and outputs_ref is not None:
            return True
        return None

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        check_derived = self.check_is_derived(self.is_derived, self.outputs_ref)
        if check_derived is not None:
            self.is_derived = check_derived
        else:
            logger.error(
                'A derived output property must have a reference to another `Outputs` section.'
            )
            return


class Outputs(BaseOutputs):
    """
    Output properties of a simulation.

    # ! add more description once we defined the output properties
    """

    n_scf_steps = Quantity(
        type=np.int32,
        description="""
        Number of self-consistent steps to converge the output property.
        """,
    )

    scf_step = SubSection(
        sub_section=BaseOutputs.m_def,
        repeats=True,
        description="""
        Self-consistent (SCF) steps performed for converging a given output property.
        """,
    )

    is_converged = Quantity(
        type=bool,
        description="""
        Flag indicating whether the output property is converged or not. This quantity is connected
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

    # ? Can we add more functionality to automatically check convergence from `self_consistency_ref` and the last `scf_step[-1]`
    def check_is_converged(self, is_converged: bool, logger: BoundLogger) -> bool:
        if not is_converged:
            logger.info('The output property is not converged.')
            return False
        return True

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        self.is_converged = self.check_is_converged(self.is_converged, logger)
