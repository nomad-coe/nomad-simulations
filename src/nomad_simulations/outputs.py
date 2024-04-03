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


class Outputs(ArchiveSection):
    """
    Output properties of a simulation. This base class can be used for inheritance in any of the output properties
    defined in this schema.

    It contains references to the specific sections used to obtain the output properties, as well as
    information if the output `is_derived` from another output section or directly parsed from the simulation output files.
    """

    # TODO add time quantities

    normalizer_level = 2

    name = Quantity(
        type=str,
        description="""
        Name of the output property. This is used for easier identification of the property and is connected
        with the class name of each output property class, e.g., `'ElectronicBandGap'`, `'ElectronicBandStructure'`, etc.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    orbitals_state_ref = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the `OrbitalsState` section to which the output property references to and on
        on which the simulation is performed.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    atoms_state_ref = Quantity(
        type=AtomsState,
        description="""
        Reference to the `AtomsState` section to which the output property references to and on
        on which the simulation is performed.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    model_system_ref = Quantity(
        type=ModelSystem,
        description="""
        Reference to the `ModelSystem` section to which the output property references to and on
        on which the simulation is performed.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    is_derived = Quantity(
        type=bool,
        default=False,
        description="""
        Flag indicating whether the output property is derived from other output properties. We make
        the distinction between directly parsed and derived output properties:
            - Directly parsed: the output property is directly parsed from the simulation output files.
            - Derived: the output property is derived from other output properties. No extra numerical settings
                are required to calculate the output property.
        """,
    )

    outputs_ref = Quantity(
        type=Reference(SectionProxy('Outputs')),
        description="""
        Reference to the `Outputs` section from which the output property was derived. This is only
        relevant if `is_derived` is set to True.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    def resolve_is_derived(self, outputs_ref) -> bool:
        """
        Resolves if the output property is derived or not.

        Args:
            outputs_ref (_type_): The reference to the `Outputs` section from which the output property was derived.

        Returns:
            bool: The flag indicating whether the output property is derived or not.
        """
        if outputs_ref is not None:
            return True
        return False

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve if the output property `is_derived` or not.
        self.is_derived = self.resolve_is_derived(self.outputs_ref)


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
