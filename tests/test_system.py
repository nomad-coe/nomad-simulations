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


# Produce a highly symmetric ethane molecule in the staggered conformation.
ethane = ase.Atoms(
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


def setup_neighbor_list(atomic_cell: ase.Atoms):
    """Build a neighbor list for the ethane molecule."""
    return ase_nl.build_neighbor_list(
        atomic_cell,
        ase_nl.natural_cutoffs(atomic_cell, mult=1.0),
        self_interaction=False,
    )


@pytest.mark.parametrize(
    'atomic_cell, pair_references, triple_references',
    [
        [
            ethane,
            [
                ('C', 'C'),
                ('C', 'H'),
                ('H', 'H'),
            ],
            [
                ('C', 'C', 'C'),
                ('C', 'C', 'H'),
                ('C', 'H', 'H'),
                ('H', 'C', 'C'),
                ('H', 'C', 'H'),
                ('H', 'H', 'H'),
            ],
        ]
    ],
)
def test_distribution_factory(
    atomic_cell: ase.Atoms,
    pair_references: list[tuple[str, str]],
    triple_references: list[tuple[str, str, str]],
):
    """
    Check the correct generation of elemental paris and triples.
    Important factors include:
    - no duplicates: commutation laws apply, i.e. the whole pair and the last 2 elements in a triple
    - alphabetical ordering of the elements
    """
    df = DistributionFactory(atomic_cell, setup_neighbor_list(atomic_cell))
    assert df.get_elemental_pairs == pair_references
    assert df.get_elemental_triples_centered == triple_references


@pytest.mark.parametrize(
    'atomic_cell, analysis_type, elements, references',
    [
        [ethane, 'distances', ['C', 'C'], [1.53]],
        [ethane, 'distances', ['C', 'H'], [0.97] * 4 + [1.09] * 2],
        [
            ethane,
            'angles',
            ['C', 'C', 'H'],
            [111.1] * 2 + [113.98] * 4,
        ],
        [ethane, 'angles', ['C', 'H', 'H'], [120]],
    ],
)  # references should be in ascending order
def test_distribution(
    atomic_cell: AtomicCell,
    analysis_type: str,
    elements: list[str],
    references: list[float],
):
    """
    Check the actual interatomic distances against the expected values.
    """
    values = np.sort(
        Distribution(
            elements, analysis_type, atomic_cell, setup_neighbor_list(atomic_cell)
        ).values.magnitude
    )

    if analysis_type == 'distances':
        # only positive reference values allowed
        assert (0 <= values).all()  # ? should this be restricted to positive definite
    elif analysis_type == 'angles':
        # only angles between 0 and pi allowed
        assert (0 <= values).all() and (values <= 180).all()
    assert np.allclose(values, references, atol=0.01)  # check the actual values
