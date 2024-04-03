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

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    JSON,
)

from .outputs import Outputs, PhysicalProperty


class SpectralProfile(Outputs, PhysicalProperty):
    """
    A base section used to define the spectral profile.
    """

    intensities = Quantity(
        type=np.float64,
        shape=['*'],
        description="""
        Intensities in arbitrary units.
        """,
    )

    intensities_units = Quantity(
        type=str,
        description="""
        Unit using the pint UnitRegistry() notation for the `intensities`. Example, if the spectra `is_derived`
        from the imaginary part of the dielectric function, `intensities_units` are `'F/m'`.
        """,
    )

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super(Outputs).__init__(m_def, m_context, **kwargs)
        super(PhysicalProperty).__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super(Outputs).normalize(archive, logger)


class ElectronicSpectralProfile(SpectralProfile):
    """
    A base section used to define the electronic spectral profile variables. These are defined as the excitation energies.
    """

    n_energies = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of energies.
        """,
    )

    energies = Quantity(
        type=np.float64,
        shape=['n_energies'],
        unit='joule',
        description="""
        Energies.
        """,
    )

    energies_origin = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Origin of reference for the energies. This quantity is used to set the `energies` to zero at the origin.
        """,
    )


class ElectronicDensityOfStates(ElectronicSpectralProfile):
    """
    Electronic Density of States (DOS).
    """

    energy_fermi = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Fermi energy.
        """,
    )

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding DOS. It can take values of 0 or 1.
        """,
    )

    normalization_factor = Quantity(
        type=np.float64,
        description="""
        Normalization factor for DOS values to get a cell-independent intensive DOS,
        defined as the DOS integral from the lowest energy state to the Fermi level for a neutrally charged system.
        """,
    )

    intensities_integrated = Quantity(
        type=np.float64,
        shape=['n_energies'],
        description="""
        A cumulative DOS starting from the mimunum energy available up to the energy level specified in `energies`.
        """,
    )
