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

from nomad_simulations.schema_packages.properties import (
    FermiLevel,
    KineticEnergy,
    PotentialEnergy,
    TotalEnergy,
)


class TestFermiLevel:
    """
    Test the `FermiLevel` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `FermiLevel` class.
        """
        fermi_level = FermiLevel()
        assert fermi_level.iri == 'http://fairmat-nfdi.eu/taxonomy/FermiLevel'
        assert fermi_level.name == 'FermiLevel'
        assert fermi_level.rank == []


class TestTotalEnergy:
    """
    Test the `TotalEnergy` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `TotalEnergy` class.
        """
        total_energy = TotalEnergy()
        # assert total_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/TotalEnergy'
        assert total_energy.name == 'TotalEnergy'
        assert total_energy.rank == []


class TestKineticEnergy:
    """
    Test the `KineticEnergy` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `KineticEnergy` class.
        """
        kinetic_energy = KineticEnergy()
        # assert kinetic_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/KineticEnergy'
        assert kinetic_energy.name == 'KineticEnergy'
        assert kinetic_energy.rank == []


class TestPotentialEnergy:
    """
    Test the `PotentialEnergy` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `PotentialEnergy` class.
        """
        potential_energy = PotentialEnergy()
        # assert potential_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/PotentialEnergy'
        assert potential_energy.name == 'PotentialEnergy'
        assert potential_energy.rank == []
