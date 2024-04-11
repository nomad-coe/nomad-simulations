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

from structlog.stdlib import BoundLogger
from typing import Optional

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity, SubSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation

from .model_system import ModelSystem
from .physical_property import PhysicalProperty
from .numerical_settings import SelfConsistency
from .properties import ElectronicBandGap


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

    custom_physical_property = SubSection(
        sub_section=PhysicalProperty.m_def,
        repeats=True,
        description="""
        A custom physical property used to store properties not yet covered by the NOMAD schema.
        """,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # List of properties
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    electronic_band_gap = SubSection(sub_section=ElectronicBandGap.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve if the output property `is_derived` or not.
        # self.is_derived = self.resolve_is_derived(self.outputs_ref)


class SCFOutputs(Outputs):
    """
    This section contains the self-consistent (SCF) steps performed to converge an output property.

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

    def get_last_scf_steps_value(
        self,
        scf_last_steps: list,
        property_name: str,
        i_property: int,
        scf_parameters: SelfConsistency,
        logger: BoundLogger,
    ) -> Optional[list]:
        """
        Get the last two SCF values' magnitudes of a physical property and appends then in a list.

        Args:
            scf_last_steps (list): The list of the last two SCF steps.
            property_name (str): The name of the physical property.
            i_property (int): The index of the physical property.

        Returns:
            (Optional[list]): The list of the last two SCF values' magnitudes of a physical property.
        """
        scf_values = []
        for step in scf_last_steps:
            scf_phys_property = getattr(step, property_name)[i_property]
            try:
                if scf_phys_property.value.u != scf_parameters.threshold_change_unit:
                    logger.error(
                        f'The units of the `scf_step.{property_name}.value` does not coincide with the units of the `self_consistency_ref.threshold_unit`.'
                    )
                    return []
            except Exception:
                return []
            scf_values.append(scf_phys_property.value.magnitude)
        return scf_values

    def resolve_is_scf_converged(
        self,
        property_name: str,
        i_property: int,
        phys_property: PhysicalProperty,
        logger: BoundLogger,
    ) -> Optional[bool]:
        """
        Resolves if the physical property is converged or not after a SCF process. This is only ran
        when there are at least two `scf_steps` elements.

        Returns:
            (bool): The flag indicating whether the physical property is converged or not after a SCF process.
        """
        # If there are not at least 2 `scf_steps`, return None
        if len(self.scf_steps) < 2:
            logger.warning('The SCF normalization needs at least two SCF steps.')
            return None
        scf_last_steps = self.scf_steps[-2:]

        # If there is not `self_consistency_ref` section, return None
        scf_parameters = phys_property.self_consistency_ref
        if scf_parameters is None:
            return None

        # Extract the value.magnitude of the phys_property to be checked if converged or not
        scf_values = self.get_last_scf_steps_value(
            scf_last_steps, property_name, i_property, scf_parameters, logger
        )
        if scf_values is None or len(scf_values) != 2:
            logger.warning(
                f'The SCF normalization could not resolve the SCF values for the property `{property_name}`.'
            )
            return None

        # Compare with the `threshold_change`
        scf_diff = abs(scf_values[0] - scf_values[1])
        threshold_change = scf_parameters.threshold_change
        if scf_diff <= threshold_change:
            return True
        else:
            logger.info(
                f'The SCF process for the property `{property_name}` did not converge.'
            )
            return False

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve the `is_scf_converged` flag for all SCF obtained properties
        for property_name in self.m_def.all_sub_sections.keys():
            # Skip the `scf_steps` and `custom_physical_property` sub-sections
            if (
                property_name == 'scf_steps'
                or property_name == 'custom_physical_property'
            ):
                continue

            # Check if the physical property with that property name is populated
            phys_properties = getattr(self, property_name)
            if phys_properties is None:
                continue
            if not isinstance(phys_properties, list):
                phys_properties = [phys_properties]

            # Loop over the physical property of the same m_def type and set `is_scf_converged`
            for i_property, phys_property in enumerate(phys_properties):
                phys_property.is_scf_converged = self.resolve_is_scf_converged(
                    property_name, i_property, phys_property, logger
                )
