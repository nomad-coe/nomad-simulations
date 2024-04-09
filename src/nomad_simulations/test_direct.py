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

from nomad_simulations.outputs import ElectronicBandGap

from nomad_simulations.variables import Variables, Temperature  # ? delete these imports

# Playing with `PhysicalProperty`
band_gap = ElectronicBandGap(source='simulation', type='direct', label='DFT')
n_grid_points = 3
temperature = Temperature(n_grid_points=n_grid_points, grid_points=np.linspace(0, 100, n_grid_points))
band_gap.variables.append(temperature)
n_grid_points = 2
custom_bins = Variables(n_grid_points=n_grid_points, grid_points=np.linspace(0, 100, n_grid_points))
band_gap.variables.append(custom_bins)
# band_gap.value_unit = 'joule'
# band_gap.shape = [3]
# band_gap.value = [
#     [[1, 2, 3], [1, 2, 3]],
#     [[1, 2, 3], [1, 2, 3]],
#     [[1, 2, 3], [1, 2, 3]],
# ] * ureg.eV
band_gap.value = [
    [1, 1],
    [1, 1],
    [1, 1],
] * ureg.eV
# band_gap.value = [1, 2, 3] * ureg.eV
print(band_gap)
