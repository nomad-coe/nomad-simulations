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

import ase
from ase import neighborlist as ase_nl
import pytest
import numpy as np

from . import logger

from nomad.units import ureg
from nomad_simulations.model_system import (
    AtomicCell,
    AtomsState,
    Distribution,
    DistributionHistogram,
    DistributionFactory,
)


def setup_geometry_analysis():  # ? move to conftest.py
    """Produce a highly symmetric ethane molecule in the staggered conformation."""
    return ase.Atoms(
        symbols=(['C'] * 2 + ['H'] * 6),
        positions=[
            [0, 0, 0.765],
            [0, 0, -0.765],
            [0, 1.01692, 1.1574],
            [-0.88068, -0.050846, 1.1574],
            [0.88068, -0.050846, 1.1574],
            [0, -1.01692, -1.1574],
            [-0.88068, 0.050846, -1.1574],
            [0.88068, 0.050846, -1.1574],
        ],  # in angstrom
        cell=[
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
        ],  # in angstrom
    )


@pytest.mark.parametrize(
    'atomic_cell, analysis_type, elements, references',
    [
        [setup_geometry_analysis(), 'distances', ['C', 'C'], [1.53]],
        [setup_geometry_analysis(), 'distances', ['C', 'H'], [0.97] * 4 + [1.09] * 2],
        [
            setup_geometry_analysis(),
            'angles',
            ['C', 'C', 'H'],
            [111.1] * 2 + [113.98] * 4,
        ],
        [setup_geometry_analysis(), 'angles', ['C', 'H', 'H'], [120]],
    ],
)  # references should be in ascending order
def test_distribution(
    atomic_cell: AtomicCell,
    analysis_type: str,
    elements: tuple[str],
    references: list[float],
):
    """
    Check the actual interatomic distances against the expected values.
    """
    neighbor_list = ase_nl.build_neighbor_list(
        atomic_cell,
        ase_nl.natural_cutoffs(atomic_cell, mult=1.0),  # ? test element X
        self_interaction=False,
    )
    values = np.sort(
        Distribution(
            elements, analysis_type, atomic_cell, neighbor_list
        ).values.magnitude
    )

    if analysis_type == 'distances':
        # only positive reference values allowed
        assert (0 <= values).all()  # ? should this be restricted to positive definite
    elif analysis_type == 'angles':
        # only angles between 0 and pi allowed
        assert (0 <= values).all() and (values <= 180).all()
    assert np.allclose(values, references, atol=0.01)  # check the actual values
