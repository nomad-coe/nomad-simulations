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

from nomad_simulations.outputs import BaseOutputs


class TestBaseOutputs:
    """
    Test the `BaseOutputs` class defined in `outputs.py`.
    """

    @pytest.mark.parametrize(
        'is_derived, outputs_ref, result',
        [
            (False, BaseOutputs(), True),
            (False, None, False),
            (True, BaseOutputs(), True),
            (True, None, None),
        ],
    )
    def test_normalize(self, is_derived, outputs_ref, result):
        """
        Test the `normalize` and `check_is_derived` methods.
        """
        outputs = BaseOutputs()
        assert outputs.check_is_derived(is_derived, outputs_ref) == result
        outputs.is_derived = is_derived
        outputs.outputs_ref = outputs_ref
        outputs.normalize(None, logger)
        if result is not None:
            assert outputs.is_derived == result
