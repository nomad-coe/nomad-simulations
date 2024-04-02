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

import pytest

from nomad.utils import get_logger
from nomad_simulations.model_method import CoupledCluster


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


LOGGER = get_logger(__name__)


@pytest.mark.parametrize(
    'name, reference',
    [
        (
            'cc',
            {
                'exc_order': None,
                'ptb_order': None,
                'slv': 'variational',
                'corr': '',
            },
        ),
        (
            'ccd',
            {
                'exc_order': [2],
                'ptb_order': None,
                'slv': 'variational',
                'corr': '',
            },
        ),
        (
            'ccsd',
            {
                'exc_order': [1, 2],
                'ptb_order': None,
                'slv': 'variational',
                'corr': '',
            },
        ),
        (
            'ccsdt',
            {
                'exc_order': [1, 2, 3],
                'ptb_order': None,
                'slv': 'variational',
                'corr': '',
            },
        ),
        (
            'ccsdtq',
            {
                'exc_order': [1, 2, 3, 4],
                'ptb_order': None,
                'slv': 'variational',
                'corr': '',
            },
        ),
        (
            'ccsd(t)',
            {
                'exc_order': [1, 2],
                'ptb_order': [3],
                'slv': 'variational',
                'corr': '',
            },
        ),
        (
            'ccsdt(q)',
            {
                'exc_order': [1, 2, 3],
                'ptb_order': [4],
                'slv': 'variational',
                'corr': '',
            },
        ),
        (
            'qvccd',
            {
                'exc_order': [2],
                'ptb_order': None,
                'slv': 'quasi-variational',
                'corr': '',
            },
        ),
        (
            'bccd',
            {
                'exc_order': [2],
                'ptb_order': None,
                'slv': 'Brueckner',
                'corr': '',
            },
        ),
        (
            'ccd-f12',
            {
                'exc_order': [2],
                'ptb_order': None,
                'slv': 'variational',
                'corr': 'F12',
            },
        ),
        (
            'ccd-r12',
            {
                'exc_order': [2],
                'ptb_order': None,
                'slv': 'variational',
                'corr': 'R12',
            },
        ),
    ],
)
def test_cc(name, reference):
    method = CoupledCluster(type=name.upper())
    method.normalize(None, LOGGER)
    for ref, res in {'exc': 'excitation', 'ptb': 'perturbative'}.items():
        res_order = getattr(method, f'{res}_order')
        res_order = list(res_order) if res_order is not None else None
        assert res_order == reference[f'{ref}_order']
    assert method.solver == reference['slv']
    assert method.explicit_correlation == reference['corr']
