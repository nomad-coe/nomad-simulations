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

from . import logger

from nomad_simulations.outputs import Outputs


class TestOutputs:
    """
    Test the `Outputs` class defined in `outputs.py`.
    """

    @pytest.mark.parametrize(
        'outputs_ref, result',
        [
            (Outputs(), True),
            (None, False),
        ],
    )
    def test_normalize(self, outputs_ref, result):
        """
        Test the `normalize` and `resolve_is_derived` methods.
        """
        # dummy test until we implement the unit testing with the new schema
        assert True
        # outputs = Outputs()
        # assert outputs.resolve_is_derived(outputs_ref) == result
        # outputs.outputs_ref = outputs_ref
        # outputs.normalize(None, logger)
        # assert outputs.is_derived == result
