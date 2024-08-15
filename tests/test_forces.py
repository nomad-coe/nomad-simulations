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

from nomad_simulations.schema_packages.properties import TotalForce


class TestTotalForce:
    """
    Test the `TotalForce` class defined in `properties/forces.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `TotalForce` class.
        """
        total_force = TotalForce()
        # assert total_force.iri == 'http://fairmat-nfdi.eu/taxonomy/TotalForce'
        assert total_force.name == 'TotalForce'
        assert total_force.rank == []
