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

    def check_is_derived(self, is_derived: bool, outputs_ref) -> Optional[bool]:
        """
        Check if the output property is derived or not.

        Args:
            is_derived (bool): The flag indicating whether the output property is derived or not.
            outputs_ref (_type_): The reference to the `BaseOutputs` section from which the output property was derived.

        Returns:
            Optional[bool]: The flag indicating whether the output property is derived or not, or whether there are missing references exists (returns None).
        """
        if not is_derived:
            if outputs_ref is not None:
                return True
            return False
        elif is_derived and outputs_ref is not None:
            return True
        return None

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Check if the output property `is_derived` or not, or if there are missing references.
        check_derived = self.check_is_derived(self.is_derived, self.outputs_ref)
        if check_derived is not None:
            self.is_derived = check_derived
        else:
            logger.error(
                'A derived output property must have a reference to another `Outputs` section.'
            )
            return


class SCFOutputs(Outputs):
    """
    This section contains the self-consistent (SCF) steps performed to converge an output property,
    as well as the information if the output property `is_converged` or not, depending on the
    settings in the `SelfConsistency` base class defined in `numerical_settings.py`.

    For simplicity, we contain the SCF steps of a simulation as part of the minimal workflow defined in NOMAD,
    the `SinglePoint`, i.e., we do not split each SCF step in its own entry. Thus, each `SinglePoint`
    `Simulation` entry in NOMAD contains the final output properties and all the SCF steps.
    """

    n_scf_steps = Quantity(
        type=np.int32,
        description="""
        Number of self-consistent steps to converge the output property. Note that the SCF steps belong to
        the same minimal `Simulation` workflow entry which is known as `SinglePoint`.
        """,
    )

    scf_step = SubSection(
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

    # TODO add more functionality to automatically check convergence from `self_consistency_ref` and the last `scf_step[-1]`
    def check_is_scf_converged(
        self, is_scf_converged: bool, logger: BoundLogger
    ) -> bool:
        """
        Check if the output property is converged or not.

        Args:
            is_converged (bool): The flag indicating whether the output property is converged or not.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (bool): The flag indicating whether the output property is converged or not.
        """
        if not is_scf_converged:
            # ? It will be nice if some of this logger messages can be checked or be used when querying data
            logger.info('The output property is not converged after the SCF process.')
            return False
        return True

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set if the output property `is_converged` or not.
        self.is_scf_converged = self.check_is_scf_converged(
            self.is_scf_converged, logger
        )
