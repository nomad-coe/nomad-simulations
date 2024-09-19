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

    def compare_unknown(key, value1, value2):
        assert value1 == None, f"Unknown types'{key}': {type(value1)} != {type(value2)}"

    def compare_string(key, str1, str2):
        assert str1 == str2, f"Value mismatch for key '{key}': {str1} != {str2}"

    def compare_float(key, float1, float2):
        if abs(float1) == float('inf'):
            assert 'inf' == float2 if float1 > 0 else '-inf' == float2
        else:
            assert float1 == approx(
                float2
            ), f"Value mismatch for key '{key}': {float1} != {float2}"

    def compare_arrays(key, arr1, arr2):
        assert np.isclose(
            arr1, arr2
        ).all(), f"Value mismatch for key '{key}': {arr1} != {arr2}"

    def compare_lists(key, l1, l2):
        assert len(l1) == len(
            l2
        ), f"Length mismatch for key '{key}': {len(l1)} != {len(l2)}"

        for i, l1_item in enumerate(l1):
            if isinstance(l1_item, dict) and isinstance(l2[i], dict):
                assert_dict_equal(l1_item, l2[i])
            elif isinstance(l1_item, (str, bool)) and isinstance(l2[i], (str, bool)):
                compare_string(f'{key}-{i}', l1_item, l2[i])
            elif isinstance(l1_item, list) and isinstance(l2[i], list):
                compare_lists(f'{key}-{i}', l1_item, l2[i])
            elif isinstance(l1_item, np.ndarray) and isinstance(l2[i], np.ndarray):
                compare_arrays(f'{key}-{i}', l1_item, l2[i])
            elif isinstance(l1_item, (float, int)) and isinstance(l2[i], (float, int)):
                compare_float(f'{key}-{i}', l1_item, l2[i])
            else:
                compare_unknown(f'{key}-{i}', l1_item, l2[i])

    for key in d1:
        print(f'key: {key}', d1[key], d2[key])
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            assert_dict_equal(d1[key], d2[key])
        elif isinstance(d1[key], (str, bool)) and isinstance(d2[key], (str, bool)):
            compare_string(key, d1[key], d2[key])
        elif isinstance(d1[key], list) and isinstance(d2[key], list):
            compare_lists(key, d1[key], d2[key])
        elif isinstance(d1[key], np.ndarray) and isinstance(d2[key], np.ndarray):
            compare_arrays(key, d1[key], d2[key])
        elif isinstance(d1[key], (float, int)) and isinstance(d2[key], (float, int)):
            compare_float(key, d1[key], d2[key])
        else:
            compare_unknown(key, d1[key], d2[key])


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


def populate_parameters(sec_potential, parameters):
    for key, value in parameters.items():
        if key == 'parameter_entries':
            for entry in value:
                sec_parameter = ParameterEntry()
                sec_parameter.name = entry['name']
                sec_parameter.value = entry['value']
                sec_parameter.unit = entry['unit']
                sec_potential.parameters.append(sec_parameter)
        else:
            setattr(sec_potential, key, value)


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
n_interactions = 3
n_particles = 2
particle_labels = [('O', 'H'), ('O', 'H'), ('O', 'H')]
particle_indices = [(0, 1), (2, 3), (4, 5)]

# harmonic
results_harmonic_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [4, 5]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'equilibrium_value': 9.6e-11,
    'force_constant': 5.811886641495074,
    'name': 'HarmonicBond',
    'type': 'bond',
    'functional_form': 'harmonic',
}
data_harmonic_bond = (
    HarmonicBond,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 0.96 * ureg.angstrom,
        'force_constant': 3500 * ureg.kJ / MOL / ureg.nanometer**2,
    },
    results_harmonic_bond,
)


# cubic
results_cubic_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [4, 5]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'equilibrium_value': 9.6e-11,
    'force_constant': 8.302695202135819,
    'force_constant_cubic': 332107808.0854328,
    'name': 'CubicBond',
    'type': 'bond',
    'functional_form': 'cubic',
}
data_cubic_bond = (
    CubicBond,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 0.96 * ureg.angstrom,
        'force_constant': 5000 * ureg.kJ / MOL / ureg.nanometer**2,
        'force_constant_cubic': 200 * ureg.kJ / MOL / ureg.nanometer**3,
    },
    results_cubic_bond,
)

# morse
results_morse_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [4, 5]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'equilibrium_value': 9.6e-11,
    'well_depth': 7.472425681922239e-18,
    'well_steepness': 24999999999.999996,
    'name': 'MorseBond',
    'type': 'bond',
    'functional_form': 'morse',
    'force_constant': 9340.532102402796,
}
data_morse_bond = (
    MorseBond,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 0.96 * ureg.angstrom,
        'well_depth': 4500 * ureg.kJ / MOL,
        'well_steepness': 25 * (1 / ureg.nanometer),
    },
    results_morse_bond,
)

# fene
results_fene_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [4, 5]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'equilibrium_value': 9.6e-11,
    'maximum_extension': 5e-11,
    'force_constant': 6.227021401601864,
    'name': 'FeneBond',
    'type': 'bond',
    'functional_form': 'fene',
}
data_fene_bond = (
    FeneBond,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 0.96 * ureg.angstrom,
        'maximum_extension': 0.5 * ureg.angstrom,
        'force_constant': 3750 * ureg.kJ / MOL / ureg.nanometer**2,
    },
    results_fene_bond,
)

# tabulated
results_tabulated_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [3, 4]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'bins': [
        7.600e-11,
        7.970e-11,
        8.340e-11,
        8.710e-11,
        9.070e-11,
        9.440e-11,
        9.810e-11,
        1.018e-10,
        1.055e-10,
        1.092e-10,
        1.128e-10,
        1.165e-10,
        1.202e-10,
        1.239e-10,
        1.276e-10,
        1.313e-10,
        1.349e-10,
        1.386e-10,
        1.423e-10,
        1.460e-10,
    ],
    'energies': [
        1.32311751e-21,
        8.81248069e-22,
        5.28549577e-22,
        2.65354139e-22,
        9.18278089e-23,
        8.30269520e-24,
        1.47787975e-23,
        1.11422170e-22,
        2.98564919e-22,
        5.76539155e-22,
        9.45178822e-22,
        1.40498208e-21,
        1.95611499e-21,
        2.59857754e-21,
        3.33286791e-21,
        4.15882003e-21,
        5.07693206e-21,
        6.08737007e-21,
        7.19013405e-21,
        8.38572215e-21,
    ],
    'name': 'TabulatedBond',
    'type': 'bond',
    'functional_form': 'tabulated',
}
data_tabulated_bond = (
    TabulatedBond,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'bins': [
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
        * ureg.nanometer,
        'energies': [
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
        * ureg.kJ
        / MOL,
    },
    results_tabulated_bond,
)

# custom - LJ
results_custom_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [4, 5]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'name': 'BondPotential',
    'parameters': [
        {'name': 'epsilon', 'value': '2.5738355126621044e-25', 'unit': 'kilojoule'},
        {'name': 'sigma', 'value': '0.96', 'unit': 'angstrom'},
    ],
}
data_custom_bond = (
    BondPotential,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'parameter_entries': [
            {'name': 'epsilon', 'value': 0.155 / MOL, 'unit': str(ureg.kJ)},
            {'name': 'sigma', 'value': 0.96, 'unit': str(ureg.angstrom)},
        ],
    },
    results_custom_bond,
)


@pytest.mark.parametrize(
    'potential_class, n_interactions, n_particles, particle_labels, particle_indices, parameters, results',
    [
        data_harmonic_bond,
        data_cubic_bond,
        data_morse_bond,
        data_fene_bond,
        data_custom_bond,
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
    populate_parameters(sec_potential, parameters)

    sec_FF.contributions.append(sec_potential)
    sec_FF.contributions[-1].normalize(EntryArchive, BoundLogger)

    potential_dict = sec_FF.contributions[-1].m_to_dict()
    potential_dict = {
        key: value for key, value in potential_dict.items() if key in results
    }
    print(potential_dict)
    print(results)
    assert_dict_equal(potential_dict, results)
