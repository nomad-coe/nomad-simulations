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

from typing import TYPE_CHECKING, Optional

import numpy as np
from nomad.metainfo import MEnum, Quantity

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import PhysicalProperty
from nomad_simulations.schema_packages.properties.spectral_profile import (
    AbsorptionSpectrum,
)
from nomad_simulations.schema_packages.utils import get_variables
from nomad_simulations.schema_packages.variables import Frequency, KMesh

# TODO add `DielectricStrength` when we have examples and understand how to extract it from the `Permittivity` tensor.


class Permittivity(PhysicalProperty):
    """
    Response of the material to polarize in the presence of an electric field.

    Alternative names: `DielectricFunction`.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/Permittivity'

    type = Quantity(
        type=MEnum('static', 'dynamic'),
        description="""
        Type of permittivity which allows to identify if the permittivity depends on the frequency or not.
        """,
    )

    value = Quantity(
        type=np.complex128,
        # unit='joule',  # TODO check units (they have to match `SpectralProfile.value`)
        description="""
        Value of the permittivity tensor. If the value does not depend on the scattering vector `q`, then we
        can extract the optical absorption spectrum from the imaginary part of the permittivity tensor (this is also called
        macroscopic dielectric function).
        """,
    )

    # ? We need use cases to understand if we need to define contributions to the permittivity tensor.
    # ? `ionic` and `electronic` contributions are common in the literature.

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]
        self.name = self.m_def.name
        self._axes_map = ['xx', 'yy', 'zz']

    def resolve_type(self) -> str:
        frequencies = get_variables(self.variables, Frequency)
        if len(frequencies) > 0:
            return 'dynamic'
        return 'static'

    def extract_absorption_spectra(
        self, logger: 'BoundLogger'
    ) -> Optional[list[AbsorptionSpectrum]]:
        """
        Extract the absorption spectrum from the imaginary part of the permittivity tensor.
        """
        # If the `pemittivity` depends on the scattering vector `q`, then we cannot extract the absorption spectrum
        q_mesh = get_variables(self.variables, KMesh)
        if len(q_mesh) > 0:
            logger.warning(
                'The `permittivity` depends on the scattering vector `q`, so that we cannot extract the absorption spectrum.'
            )
            return None
        # Extract the `Frequency` variable to extract the absorption spectrum
        frequencies = get_variables(self.variables, Frequency)
        if len(frequencies) == 0:
            logger.warning(
                'The `permittivity` does not have a `Frequency` variable to extract the absorption spectrum.'
            )
            return None
        # Define the `absorption_spectra` for each principal direction along the diagonal of the `Permittivity.value` as the imaginary part
        spectra = []
        for i in range(3):
            val = self.value[:, i, i].imag
            absorption_spectrum = AbsorptionSpectrum(
                axis=self._axes_map[i], variables=frequencies
            )
            absorption_spectrum.value = val
            absorption_spectrum.physical_property_ref = self
            spectra.append(absorption_spectrum)
        return spectra

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve the `type` of permittivity
        self.type = self.resolve_type()

        # `AbsorptionSpectrum` extraction
        absorption_spectra = self.extract_absorption_spectra(logger)
        if absorption_spectra is not None:
            self.m_parent.absorption_spectrum = absorption_spectra
