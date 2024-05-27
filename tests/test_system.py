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
                ['C', 'C'],
                ['C', 'H'],
                ['H', 'H'],
            ],
            [
                ['C', 'C', 'C'],
                ['C', 'C', 'H'],
                ['H', 'C', 'H'],
                ['C', 'H', 'C'],
                ['C', 'H', 'H'],
                ['H', 'H', 'H'],
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
    Important factors (enforced by the references) include:
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
    Extra checks are performed for:
    - distances: positive values only
    - angles: values between 0 and 180
    """
    values = np.sort(
        Distribution(
            elements, atomic_cell, setup_neighbor_list(atomic_cell)
        ).values.magnitude
    )

    if analysis_type == 'distances':
        assert (0 <= values).all()  # ? should this be restricted to positive definite
    elif analysis_type == 'angles':
        assert (0 <= values).all() and (values <= 180).all()
    assert np.allclose(values, references, atol=0.01)  # check the actual values


@pytest.mark.parametrize(
    'atomic_cell, elements, bins, count_reference_map',
    [
        [ethane, ['C', 'C'], np.array([]), {}],  # empty bins
        [ethane, ['C', 'C'], np.arange(0, 2, 0.01), {1.53: 1}],
        [ethane, ['C', 'H'], np.arange(0, 2, 0.01), {0.96: 2, 1.09: 1}],
        [ethane, ['C', 'C', 'H'], np.arange(0, 180, 1), {111: 1, 113: 2}],
        [ethane, ['H', 'C', 'H'], np.arange(0, 180, 1), {131: 1}],  # why 131?
    ],  # note that the exact bin is hard to pin down: may deviate by 1 index
)
def test_distribution_histogram(
    atomic_cell: AtomicCell,
    elements: list[str],
    bins: np.ndarray,
    count_reference_map: dict[float, int],
):
    """
    Check the conversion of a distribution into a histogram, such as:
    Tests focusing on the binning include:
    1. test empty bins generating empty histogram
    2. test bin units
    Lastly, the histogram frequency count is checked, with attention for:
    3. normalization of the count, so the proper lower limit is 1
    4. the actual count
    """
    to_type = 'distances' if len(elements) == 2 else 'angles'
    dist = Distribution(elements, atomic_cell, setup_neighbor_list(atomic_cell))
    dh = DistributionHistogram(elements, to_type, dist.values, bins)

    if len(bins) == 0:  # 1.
        assert len(dh.frequency) == 0
        return
    assert dh.bins.u == ureg.angstrom if to_type == 'distances' else ureg.degrees  # 2.
    assert np.min(dh.frequency[dh.frequency > 0]) == 1  # 3.
    for bin, count in count_reference_map.items():  # 4.
        assert dh.frequency[np.where(dh.bins.magnitude == bin)] == count


@pytest.mark.parametrize(
    'elements',
    [['C', 'C'], ['C', 'H'], ['C', 'C', 'H'], ['C', 'H', 'H']],
)
def test_nomad_distribution(elements):
    """
    Check the instantiation of `GeometryDistribution`. via `DistributionHistogram`.
    Specifically, check sub-typing, value storage, and reporting of the structure.
    """
    # define variables
    gd = 'GeometryDistribution'
    if len(elements) == 2:
        subtype = 'Distance'
        bins = np.arange(0, 2, 0.01)
        units = ureg.angstrom
    else:
        subtype = 'Angle'
        bins = np.arange(0, 180, 0.001)
        units = ureg.degrees

    # instantiate objects
    dh = DistributionHistogram(
        elements,
        subtype.lower() + 's',
        np.random.rand(len(bins)) * units,
        bins,
    )
    nomad_dh = dh.produce_nomad_distribution().normalize(None, None)

    # test
    assert nomad_dh.__class__.__name__ == subtype + gd
    assert np.all(dh.frequency == nomad_dh.frequency)
    assert np.allclose(dh.bins, nomad_dh.bins.to(units), atol=0.01)

    assert nomad_dh.extremity_atom_labels == [elements[0], elements[-1]]
    if subtype == 'Angle':
        assert nomad_dh.central_atom_labels == [elements[1]]
