import pytest
import numpy as np
from typing import List, Tuple, Dict, Any


from nomad_simulations.schema_packages.general import Simulation

# from nomad_simulations.schema_packages.method import ModelMethod
from nomad_simulations.schema_packages.force_field import (
    ForceField,
    ParameterEntry,
    Potential,
    BondPotential,
    HarmonicBond,
    CubicBond,
    MorseBond,
    FeneBond,
    TabulatedBond,
)
from nomad_simulations.schema_packages.numerical_settings import ForceCalculations

from nomad.datamodel import EntryArchive
from structlog.stdlib import BoundLogger
from nomad.units import ureg


MOL = 6.022140857e23


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


def assert_dict_equal(d1, d2):
    """
    Recursively assert that two dictionaries are equal.

    Args:
        d1 (dict): First dictionary to compare.
        d2 (dict): Second dictionary to compare.
    """

    assert isinstance(d1, dict), f'Expected dict, got {type(d1)}'
    assert isinstance(d2, dict), f'Expected dict, got {type(d2)}'
    assert d1.keys() == d2.keys(), f'Keys mismatch: {d1.keys()} != {d2.keys()}'

    for key in d1:
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            assert_dict_equal(d1[key], d2[key])
        else:
            if isinstance(d1[key], (str, bool)):
                assert (
                    d1[key] == d2[key]
                ), f"Value mismatch for key '{key}': {d1[key]} != {d2[key]}"
            elif isinstance(d1[key], np.ndarray):
                assert np.isclose(
                    d1[key], d2[key]
                ).all(), f"Value mismatch for key '{key}': {d1[key]} != {d2[key]}"
            elif abs(d1[key]) == float('inf'):
                assert 'inf' == d2[key] if d1[key] > 0 else '-inf' == d2[key]
            else:
                assert d1[key] == approx(
                    d2[key]
                ), f"Value mismatch for key '{key}': {d1[key]} != {d2[key]}"


def get_simulation_template():
    data = Simulation()
    sec_FF = ForceField()
    data.model_method.append(sec_FF)
    sec_force_calculations = ForceCalculations()
    data.model_method[0].numerical_settings.append(sec_force_calculations)

    return data


def populate_potential(
    class_potential,
    n_interactions=None,
    n_particles=None,
    particle_labels=None,
    particle_indices=None,
):
    sec_potential = class_potential()
    sec_potential.n_interactions = n_interactions
    sec_potential.n_particles = n_particles
    sec_potential.particle_indices = particle_indices
    sec_potential.particle_labels = particle_labels

    return sec_potential


# Test Data

# BOND POTENTIALS

# System: 3 x OH molecules
#   particle number       particle label
#   0                     O
#   1                     H
#   2                     O
#   3                     H
#   4                     O
#   5                     H


# harmonic
results_harmonic_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [3, 4]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'equilibrium_value': 9.6e-11,
    'force_constant': 0.8302695202135819,
    'name': 'HarmonicBond',
    'type': 'bond',
    'functional_form': 'harmonic',
}
data_harmonic_bond = (
    HarmonicBond,
    3,
    2,
    [('O', 'H'), ('O', 'H'), ('O', 'H')],
    [(0, 1), (2, 3), (3, 4)],
    {
        'equilibrium_value': 0.96 * ureg.angstrom,
        'force_constant': 3500 * ureg.kJ / MOL / ureg.nanometer**2,
    },
    results_harmonic_bond,
)


@pytest.mark.parametrize(
    'potential_class, n_interactions, n_particles, particle_labels, particle_indices, parameters, results',
    [
        data_harmonic_bond,
    ],
)
def test_potential(
    potential_class: Potential,
    n_interactions: int,
    n_particles: int,
    particle_labels: List[Tuple[str, str]],
    particle_indices: List[Tuple[int, int]],
    parameters: Dict[str, Any],
    results: Dict[str, Any],
):
    """_summary_

    Args:
        input (str): _description_
        result (Dict[Any]): _description_
    """

    data = get_simulation_template()
    sec_FF = data.model_method[0]
    sec_potential = populate_potential(
        potential_class,
        n_interactions=n_interactions,
        n_particles=n_particles,
        particle_labels=particle_labels,
        particle_indices=particle_indices,
    )
    for key, value in parameters.items():
        setattr(sec_potential, key, value)

    sec_FF.contributions.append(sec_potential)
    sec_FF.contributions[-1].normalize(EntryArchive, BoundLogger)

    potential_dict = sec_FF.contributions[-1].m_to_dict()
    assert_dict_equal(potential_dict, results)


# System: 3 x OH molecules
#   particle number       particle label
#   0                     O
#   1                     H
#   2                     O
#   3                     H
#   4                     O
#   5                     H
data = get_simulation_template()
sec_FF = data.model_method[0]
ctr_potential = 0
# Harmonic bond potential
sec_harmonic_bond = populate_potential(
    HarmonicBond,
    n_interactions=3,
    n_particles=2,
    particle_labels=[('O', 'H'), ('O', 'H'), ('O', 'H')],
    particle_indices=[(0, 1), (2, 3), (3, 4)],
)
sec_harmonic_bond.equilibrium_value = 0.96 * ureg.angstrom
sec_harmonic_bond.force_constant = 3500 * ureg.kJ / MOL / ureg.nanometer**2

sec_FF.contributions.append(sec_harmonic_bond)
sec_FF.contributions[ctr_potential].normalize(EntryArchive, BoundLogger)
ctr_potential += 1
# Result
# {'n_interactions': 3, 'n_particles': 2, 'particle_indices': array([[0, 1],
#        [2, 3],
#        [3, 4]]), 'particle_labels': array([['O', 'H'],
#        ['O', 'H'],
#        ['O', 'H']], dtype='<U1'), 'equilibrium_value': 9.6e-11, 'force_constant': 0.8302695202135819, 'name': 'HarmonicBond', 'type': 'bond', 'functional_form': 'harmonic'}

# data = populate_simulation()
# # data.model_method[0].normalize(EntryArchive, BoundLogger)
# sec_potential = data.model_method[0].contributions
# print(sec_potential)


# Cubic bond potential
sec_cubic_bond = populate_potential(
    CubicBond,
    n_interactions=3,
    n_particles=2,
    particle_labels=[('O', 'H'), ('O', 'H'), ('O', 'H')],
    particle_indices=[(0, 1), (2, 3), (3, 4)],
)
sec_cubic_bond.equilibrium_value = 0.96 * ureg.angstrom
sec_cubic_bond.force_constant = 5000 * ureg.kJ / MOL / ureg.nanometer**2
sec_cubic_bond.force_constant_cubic = 200 * ureg.kJ / MOL / ureg.nanometer**3

sec_FF.contributions.append(sec_cubic_bond)
sec_FF.contributions[ctr_potential].normalize(EntryArchive, BoundLogger)
ctr_potential += 1
# Result
# {'n_interactions': 3, 'n_particles': 2, 'particle_indices': array([[0, 1],
#        [2, 3],
#        [3, 4]]), 'particle_labels': array([['O', 'H'],
#        ['O', 'H'],
#        ['O', 'H']], dtype='<U1'), 'equilibrium_value': 9.6e-11, 'force_constant': 0.8302695202135819, 'force_constant_cubic': 332107808.0854328, 'name': 'CubicBond', 'type': 'bond', 'functional_form': 'cubic'}

# Morse bond potential
sec_morse_bond = populate_potential(
    MorseBond,
    n_interactions=3,
    n_particles=2,
    particle_labels=[('O', 'H'), ('O', 'H'), ('O', 'H')],
    particle_indices=[(0, 1), (2, 3), (3, 4)],
)
sec_morse_bond.equilibrium_value = 0.96 * ureg.angstrom
sec_morse_bond.well_depth = 4500 * ureg.kJ / MOL
sec_morse_bond.well_steepness = 25 * (1 / ureg.nanometer)

sec_FF.contributions.append(sec_morse_bond)
sec_FF.contributions[ctr_potential].normalize(EntryArchive, BoundLogger)
ctr_potential += 1
# Result
#  'n_interactions': 3, 'n_particles': 2, 'particle_indices': array([[0, 1],
#        [2, 3],
#        [3, 4]]), 'particle_labels': array([['O', 'H'],
#        ['O', 'H'],
#        ['O', 'H']], dtype='<U1'), 'equilibrium_value': 9.6e-11, 'well_depth': 5.811886641495074e-19, 'well_steepness': 24999999999.999996, 'name': 'MorseBond', 'type': 'bond', 'functional_form': 'morse', 'force_constant': 726.4858301868842}

# FENE bond potential
sec_fene_bond = populate_potential(
    FeneBond,
    n_interactions=3,
    n_particles=2,
    particle_labels=[('O', 'H'), ('O', 'H'), ('O', 'H')],
    particle_indices=[(0, 1), (2, 3), (3, 4)],
)
sec_fene_bond.equilibrium_value = 0.96 * ureg.angstrom
sec_fene_bond.maximum_extension = 0.5 * ureg.angstrom
sec_fene_bond.force_constand = 3750 * ureg.kJ / MOL / ureg.nanometer**2
sec_FF.contributions.append(sec_fene_bond)
sec_FF.contributions[ctr_potential].normalize(EntryArchive, BoundLogger)
ctr_potential += 1
# Result
# {'n_interactions': 3, 'n_particles': 2, 'particle_indices': array([[0, 1],
#        [2, 3],
#        [3, 4]]), 'particle_labels': array([['O', 'H'],
#        ['O', 'H'],
#        ['O', 'H']], dtype='<U1'), 'equilibrium_value': 9.6e-11, 'maximum_extension': 5e-11, 'force_constand': <Quantity(6.2270214e-21, 'kilojoule / nanometer ** 2')>, 'name': 'FeneBond', 'type': 'bond', 'functional_form': 'fene'}


# Tabulated bond potential
sec_tab_bond = populate_potential(
    TabulatedBond,
    n_interactions=3,
    n_particles=2,
    particle_labels=[('O', 'H'), ('O', 'H'), ('O', 'H')],
    particle_indices=[(0, 1), (2, 3), (3, 4)],
)
bins = [
    0.076,
    0.0797,
    0.0834,
    0.0871,
    0.0907,
    0.0944,
    0.0981,
    0.1018,
    0.1055,
    0.1092,
    0.1128,
    0.1165,
    0.1202,
    0.1239,
    0.1276,
    0.1313,
    0.1349,
    0.1386,
    0.1423,
    0.146,
]
energies = [
    0.7968,
    0.5307,
    0.3183,
    0.1598,
    0.0553,
    0.005,
    0.0089,
    0.0671,
    0.1798,
    0.3472,
    0.5692,
    0.8461,
    1.178,
    1.5649,
    2.0071,
    2.5045,
    3.0574,
    3.6659,
    4.33,
    5.05,
]
# sec_tab_bond.bins = np.array(bins) * ureg.nanometer
# sec_tab_bond.energies = np.array(energies) * ureg.kJ / MOL
sec_tab_bond.bins = bins * ureg.nanometer
sec_tab_bond.energies = energies * ureg.kJ / MOL
sec_FF.contributions.append(sec_tab_bond)
sec_FF.contributions[ctr_potential].normalize(EntryArchive, BoundLogger)
ctr_potential += 1

# {'n_interactions': 3, 'n_particles': 2, 'particle_indices': array([[0, 1],
#        [2, 3],
#        [3, 4]]), 'particle_labels': array([['O', 'H'],
#        ['O', 'H'],
#        ['O', 'H']], dtype='<U1'), 'bins': array([7.600e-11, 7.970e-11, 8.340e-11, 8.710e-11, 9.070e-11, 9.440e-11,
#        9.810e-11, 1.018e-10, 1.055e-10, 1.092e-10, 1.128e-10, 1.165e-10,
#        1.202e-10, 1.239e-10, 1.276e-10, 1.313e-10, 1.349e-10, 1.386e-10,
#        1.423e-10, 1.460e-10]), 'energies': array([1.32311751e-21, 8.81248069e-22, 5.28549577e-22, 2.65354139e-22,
#        9.18278089e-23, 8.30269520e-24, 1.47787975e-23, 1.11422170e-22,
#        2.98564919e-22, 5.76539155e-22, 9.45178822e-22, 1.40498208e-21,
#        1.95611499e-21, 2.59857754e-21, 3.33286791e-21, 4.15882003e-21,
#        5.07693206e-21, 6.08737007e-21, 7.19013405e-21, 8.38572215e-21]), 'name': 'TabulatedBond', 'type': 'bond', 'functional_form': 'tabulated'}

# Custom Potential -- LJ potential for a bond as an example
sec_custom_bond = populate_potential(
    BondPotential,
    n_interactions=3,
    n_particles=2,
    particle_labels=[('O', 'H'), ('O', 'H'), ('O', 'H')],
    particle_indices=[(0, 1), (2, 3), (3, 4)],
)
sec_custom_bond.name = 'LennardJonesBond'
epsilon = ParameterEntry(name='epsilon', value=0.155 / MOL, unit=str(ureg.kJ))
sec_custom_bond.parameters.append(epsilon)
sigma = ParameterEntry(name='sigma', value=0.96, unit=str(ureg.angstrom))
sec_custom_bond.parameters.append(sigma)
sec_FF.contributions.append(sec_custom_bond)

# {'n_interactions': 3, 'n_particles': 2, 'particle_indices': array([[0, 1],
#        [2, 3],
#        [3, 4]]), 'particle_labels': array([['O', 'H'],
#        ['O', 'H'],
#        ['O', 'H']], dtype='<U1'), 'name': 'LennardJonesBond', 'parameters': [epsilon:ParameterEntry(name, value, unit), sigma:ParameterEntry(name, value, unit)]}
# {'name': 'epsilon', 'value': '2.5738355126621044e-25', 'unit': 'kilojoule'}
# {'name': 'sigma', 'value': '0.96', 'unit': 'angstrom'}
