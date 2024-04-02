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

# import typing
# from structlog.stdlib import BoundLogger
import numpy as np

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity


# TODO check this once outputs.py is defined
class HoppingMatrix(ArchiveSection):
    """
    Section containing the hopping/overlap matrix elements between N projected orbitals. This
    is also the output of a TB calculation.
    """

    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of projected orbitals.
        """,
    )

    n_wigner_seitz_points = Quantity(
        type=np.int32,
        description="""
        Number of Wigner-Seitz real points.
        """,
    )

    # TODO check with SlaterKoster and OrbitalsState.degeneracy
    degeneracy_factors = Quantity(
        type=np.int32,
        shape=['n_wigner_seitz_points'],
        description="""
        Degeneracy of each Wigner-Seitz grid point.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=['n_wigner_seitz_points', 'n_orbitals * n_orbitals', 7],
        unit='joule',
        description="""
        Real space hopping matrix for each Wigner-Seitz grid point. The elements are
        defined as follows:

            n_x   n_y   n_z   orb_1   orb_2   real_part + j * imag_part

        where (n_x, n_y, n_z) define the Wigner-Seitz cell vector in fractional coordinates,
        (orb_1, orb_2) indicates the hopping amplitude between orb_1 and orb_2, and the
        real and imaginary parts of the hopping.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
