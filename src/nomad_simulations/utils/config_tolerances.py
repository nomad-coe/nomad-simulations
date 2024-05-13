#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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

from pydantic import BaseModel, Field


class ConfigTolerances(BaseModel):
    dos_value_threshold: float = Field(
        1e-8,
        description="""Threshold value at which the density of states (DOS) values are considered to be zero.""",
    )

    dos_energy_tolerance: float = Field(
        8.01088e-21,
        description="""Tolerance for the energies in the density of states (DOS) to compare with: $ (value - tolerance) <= value <= (value + tolerance) $""",
    )

    band_structure_energy_tolerance: float = Field(
        8.01088e-21,
        description="""Tolerance for the energies in the band structure to compare with: $ (value - tolerance) <= value <= (value + tolerance) $""",
    )

    k_space_precision: float = Field(
        150000000.0,
        description="""Precision be used when comparing the k-space values.""",
    )

    cluster_threshold: float = Field(
        2.5, description="""Threshold used when clustering."""
    )

    system_classification_with_clusters_threshold: int = Field(
        64,
        description="""Threshold used when classifying the system with clusters.""",
    )

    symmetry_tolerance: float = Field(
        0.1, description="""Tolerance used when comparing the symmetry."""
    )
