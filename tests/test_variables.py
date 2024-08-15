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
from nomad.datamodel import EntryArchive

from nomad_simulations.schema_packages.variables import Variables

from . import logger


class TestVariables:
    """
    Test the `Variables` class defined in `variables.py`.
    """

    @pytest.mark.parametrize(
        'n_points, points, result',
        [
            (3, [-1, 0, 1], 3),
            (5, [-1, 0, 1], 3),
            (None, [-1, 0, 1], 3),
            (4, None, 4),
            (4, [], 4),
        ],
    )
    def test_normalize(self, n_points: int, points: list, result: int):
        """
        Test the `normalize` and `get_n_points` methods.
        """
        variable = Variables(
            name='variable_1',
            n_points=n_points,
            points=points,
        )
        assert variable.get_n_points(logger) == result
        variable.normalize(EntryArchive(), logger)
        assert variable.n_points == result
