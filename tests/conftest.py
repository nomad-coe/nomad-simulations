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

import os
import pytest

from nomad.units import ureg

from nomad_simulations.outputs import Outputs, SCFOutputs
from nomad_simulations.numerical_settings import SelfConsistency
from nomad_simulations.properties import ElectronicBandGap

if os.getenv('_PYTEST_RAISE', '0') != '0':

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


def get_scf_electronic_band_gap_template(threshold_change: float = 1e-3) -> SCFOutputs:
    scf_outputs = SCFOutputs()
    # Define a list of scf_steps with values of the total energy like [1, 1.1, 1.11, 1.111, etc],
    # such that the difference between one step and the next one decreases a factor of 10.
    n_scf_steps = 5
    for i in range(1, n_scf_steps):
        value = 1 + sum([1 / (10**j) for j in range(1, i + 1)])
        scf_step = Outputs(
            electronic_band_gap=[ElectronicBandGap(value=value * ureg.joule)]
        )
        scf_outputs.scf_steps.append(scf_step)
    # Add a SCF calculated PhysicalProperty
    scf_outputs.electronic_band_gap.append(ElectronicBandGap(value=value * ureg.joule))
    # and a `SelfConsistency` ref section
    scf_params = SelfConsistency(
        threshold_change=threshold_change, threshold_change_unit='joule'
    )
    scf_outputs.electronic_band_gap[0].self_consistency_ref = scf_params
    return scf_outputs


@pytest.fixture(scope='session')
def scf_electronic_band_gap() -> SCFOutputs:
    return get_scf_electronic_band_gap_template()
