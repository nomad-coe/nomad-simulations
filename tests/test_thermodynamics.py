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
    Pressure,
    Volume,
    Temperature,
    HeatAdded,
    WorkDone,
    InternalEnergy,
    Enthalpy,
    Entropy,
    GibbsFreeEnergy,
    HelmholtzFreeEnergy,
    ChemicalPotential,
    HeatCapacityCV,
    HeatCapacityCP,
    Virial,
    Density,
    Hessian,
)
class TestPressure:
    """
    Test the `Pressure` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Pressure` class.
        """
        pressure = Pressure()
        assert pressure.iri == 'http://fairmat-nfdi.eu/taxonomy/Pressure'
        assert pressure.name == 'Pressure'
        assert pressure.rank == []

class TestVolume:
    """
    Test the `Volume` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Volume` class.
        """
        volume = Volume()
        assert volume.iri == 'http://fairmat-nfdi.eu/taxonomy/Volume'
        assert volume.name == 'Volume'
        assert volume.rank == []

class TestTemperature:
    """
    Test the `Temperature` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Temperature` class.
        """
        temperature = Temperature()
        assert temperature.iri == 'http://fairmat-nfdi.eu/taxonomy/Temperature'
        assert temperature.name == 'Temperature'
        assert temperature.rank == []

class TestHeatAdded:
    """
    Test the `HeatAdded` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `HeatAdded` class.
        """
        heat_added = HeatAdded()
        assert heat_added.iri == 'http://fairmat-nfdi.eu/taxonomy/HeatAdded'
        assert heat_added.name == 'HeatAdded'
        assert heat_added.rank == []

class TestWorkDone:
    """
    Test the `WorkDone` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `WorkDone` class.
        """
        work_done = WorkDone()
        assert work_done.iri == 'http://fairmat-nfdi.eu/taxonomy/WorkDone'
        assert work_done.name == 'WorkDone'
        assert work_done.rank == []

class TestInternalEnergy:
    """
    Test the `InternalEnergy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `InternalEnergy` class.
        """
        internal_energy = InternalEnergy()
        assert internal_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/InternalEnergy'
        assert internal_energy.name == 'InternalEnergy'
        assert internal_energy.rank == []

class TestEnthalpy:
    """
    Test the `Enthalpy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Enthalpy` class.
        """
        enthalpy = Enthalpy()
        assert enthalpy.iri == 'http://fairmat-nfdi.eu/taxonomy/Enthalpy'
        assert enthalpy.name == 'Enthalpy'
        assert enthalpy.rank == []


class TestEntropy:
    """
    Test the `Entropy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Entropy` class.
        """
        entropy = Entropy()
        assert entropy.iri == 'http://fairmat-nfdi.eu/taxonomy/Entropy'
        assert entropy.name == 'Entropy'
        assert entropy.rank == []

class TestGibbsFreeEnergy:
    """
    Test the `GibbsFreeEnergy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `GibbsFreeEnergy` class.
        """
        gibbs_free_energy = GibbsFreeEnergy()
        assert gibbs_free_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/GibbsFreeEnergy'
        assert gibbs_free_energy.name == 'GibbsFreeEnergy'
        assert gibbs_free_energy.rank == []

class TestHelmholtzFreeEnergy:
    """
    Test the `HelmholtzFreeEnergy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `HelmholtzFreeEnergy` class.
        """
        helmholtz_free_energy = HelmholtzFreeEnergy()
        assert helmholtz_free_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/HelmholtzFreeEnergy'
        assert helmholtz_free_energy.name == 'HelmholtzFreeEnergy'
        assert helmholtz_free_energy.rank == []

class TestChemicalPotential:
    """
    Test the `ChemicalPotential` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `ChemicalPotential` class.
        """
        chemical_potential = ChemicalPotential()
        assert (
            chemical_potential.iri
            == 'http://fairmat-nfdi.eu/taxonomy/ChemicalPotential'
        )
        assert chemical_potential.name == 'ChemicalPotential'
        assert chemical_potential.rank == []


class TestHeatCapacityCV:
    """
    Test the `HeatCapacityCV` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `HeatCapacityCV` class.
        """
        heat_capacity_cv = HeatCapacityCV()
        assert heat_capacity_cv.iri == 'http://fairmat-nfdi.eu/taxonomy/HeatCapacityCV'
        assert heat_capacity_cv.name == 'HeatCapacityCV'
        assert heat_capacity_cv.rank == []


class TestHeatCapacityCP:
    """
    Test the `HeatCapacityCP` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `HeatCapacityCP` class.
        """
        heat_capacity_cp = HeatCapacityCP()
        assert heat_capacity_cp.iri == 'http://fairmat-nfdi.eu/taxonomy/HeatCapacityCP'
        assert heat_capacity_cp.name == 'HeatCapacityCP'
        assert heat_capacity_cp.rank == []

class TestVirial:
    """
    Test the `Virial` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Virial` class.
        """
        virial = Virial()
        assert virial.iri == 'http://fairmat-nfdi.eu/taxonomy/Virial'
        assert virial.name == 'Virial'
        assert virial.rank == []


class TestDensity:
    """
    Test the `Density` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Density` class.
        """
        density = Density()
        assert density.iri == 'http://fairmat-nfdi.eu/taxonomy/Density'
        assert density.name == 'Density'
        assert density.rank == []


class TestHessian:
    """
    Test the `Hessian` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Hessian` class.
        """
        hessian = Hessian()
        assert hessian.iri == 'http://fairmat-nfdi.eu/taxonomy/Hessian'
        assert hessian.name == 'Hessian'
        assert hessian.rank == []


