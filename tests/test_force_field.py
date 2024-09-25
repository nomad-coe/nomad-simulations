import pytest
import numpy as np
from typing import List, Tuple, Dict, Any


from nomad_simulations.schema_packages.general import Simulation

# from nomad_simulations.schema_packages.method import ModelMethod
from nomad_simulations.schema_packages.force_field import (
    ForceField,
    ParameterEntry,
    Potential,
    TabulatedPotential,
    BondPotential,
    HarmonicBond,
    CubicBond,
    PolynomialForceConstant,
    PolynomialBond,
    MorseBond,
    FeneBond,
    TabulatedBond,
    AnglePotential,
    HarmonicAngle,
    CosineAngle,
    RestrictedCosineAngle,
    FourierSeriesAngle,
    UreyBradleyAngle,
    PolynomialAngle,
    TabulatedAngle,
)
from nomad_simulations.schema_packages.numerical_settings import ForceCalculations

from nomad.datamodel import EntryArchive

# from structlog.stdlib import BoundLogger
from . import logger
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

# polynomial
results_polynomial_bond = {
    'n_interactions': 3,
    'n_particles': 2,
    'particle_indices': [[0, 1], [2, 3], [4, 5]],
    'particle_labels': [['O', 'H'], ['O', 'H'], ['O', 'H']],
    'equilibrium_value': 9.6e-11,
    'force_constants': [
        {
            'name': 'k_2',
            'value': 8.302695202135821e-21,
            'unit': 'kilojoule / nanometer ** 2',
            'exponent': 2,
        },
        {
            'name': 'k_4',
            'value': 3.4871319848970445e-21,
            'unit': 'kilojoule / nanometer ** 4',
            'exponent': 4,
        },
    ],
    'name': 'PolynomialBond',
    'type': 'bond',
    'functional_form': 'polynomial',
}
data_polynomial_bond = (
    PolynomialBond,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 0.96 * ureg.angstrom,
        'force_constants': [
            PolynomialForceConstant(
                name='k_2',
                exponent=2,
                value=5000.0 / MOL,
                unit=str(ureg.kJ / ureg.nanometer**2),
            ),
            PolynomialForceConstant(
                name='k_4',
                exponent=4,
                value=2100.0 / MOL,
                unit=str(ureg.kJ / ureg.nanometer**4),
            ),
        ],
    },
    results_polynomial_bond,
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
    'particle_indices': [[0, 1], [2, 3], [4, 5]],
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
    'forces': [
        1.3216958010639183e-10,
        1.0784837919852664e-10,
        8.347183948070384e-11,
        5.903996095292366e-11,
        3.521528791034704e-11,
        1.0674227406164319e-11,
        -1.3922171907975959e-11,
        -3.857391003207359e-11,
        -6.32809869661286e-11,
        -8.804340271014125e-11,
        -1.1218967953067358e-10,
        -1.3706127725108878e-10,
        -1.6198821378146133e-10,
        -1.8697048912179138e-10,
        -2.1200810327207885e-10,
        -2.3710105623232373e-10,
        -2.6156893683081195e-10,
        -2.867710717674596e-10,
        -3.1202854551406456e-10,
        -3.373413580706269e-10,
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
        'energies': [  # ! pass in energies and test the auto-generation of forces
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

# Angle POTENTIALS

# System: 2 x H20 molecules
#   particle number       particle label
#   0                     O
#   1                     H
#   2                     H
#   3                     O
#   4                     H
#   5                     H
n_interactions = 2
n_particles = 3
particle_labels = [('O', 'H', 'H'), ('O', 'H', 'H')]
particle_indices = [(0, 1, 2), (3, 4, 5)]

# harmonic
results_harmonic_angle = {
    'n_interactions': 2,
    'n_particles': 3,
    'particle_indices': [[0, 1, 2], [3, 4, 5]],
    'particle_labels': [['O', 'H', 'H'], ['O', 'H', 'H']],
    'equilibrium_value': 104.45020605234907,
    'force_constant': 3.7937183846251475e-23,
    'name': 'HarmonicAngle',
    'type': 'angle',
    'functional_form': 'harmonic',
}
data_harmonic_angle = (
    HarmonicAngle,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 1.823 * ureg.radian,
        'force_constant': 75 * ureg.kJ / MOL / ureg.radian**2,
    },
    results_harmonic_angle,
)

# cosine
results_cosine_angle = {
    'n_interactions': 2,
    'n_particles': 3,
    'particle_indices': [[0, 1, 2], [3, 4, 5]],
    'particle_labels': [['O', 'H', 'H'], ['O', 'H', 'H']],
    'equilibrium_value': 104.45020605234907,
    'force_constant': 3.7937183846251475e-23,
    'name': 'CosineAngle',
    'type': 'angle',
    'functional_form': 'cosine',
}
data_cosine_angle = (
    CosineAngle,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 1.823 * ureg.radian,
        'force_constant': 75 * ureg.kJ / MOL / ureg.radian**2,
    },
    results_cosine_angle,
)

# restricted cosine
results_restrictedcosine_angle = {
    'n_interactions': 2,
    'n_particles': 3,
    'particle_indices': [[0, 1, 2], [3, 4, 5]],
    'particle_labels': [['O', 'H', 'H'], ['O', 'H', 'H']],
    'equilibrium_value': 104.45020605234907,
    'force_constant': 8.70026082874034e-23,
    'name': 'RestrictedCosineAngle',
    'type': 'angle',
    'functional_form': 'restricted_cosine',
}
data_restrictedcosine_angle = (
    RestrictedCosineAngle,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 1.823 * ureg.radian,
        'force_constant': 172 * ureg.kJ / MOL / ureg.radian**2,
    },
    results_restrictedcosine_angle,
)

# fourier_series
results_fourier_angle = {
    'n_interactions': 2,
    'n_particles': 3,
    'particle_indices': [[0, 1, 2], [3, 4, 5]],
    'particle_labels': [['O', 'H', 'H'], ['O', 'H', 'H']],
    'equilibrium_value': 104.45020605234907,
    'fourier_force_constants': [5.0582911795001975e-23, 0.0, 1.2645727948750494e-23],
    'name': 'FourierSeriesAngle',
    'type': 'angle',
    'functional_form': 'fourier_series',
}
data_fourier_angle = (
    FourierSeriesAngle,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 1.823 * ureg.radian,
        'fourier_force_constants': [100, 0, 25] * ureg.kJ / MOL / ureg.radian**2,
    },
    results_fourier_angle,
)

# urey_bradley
results_ureybradley_angle = {
    'n_interactions': 2,
    'n_particles': 3,
    'particle_indices': [[0, 1, 2], [3, 4, 5]],
    'particle_labels': [['O', 'H', 'H'], ['O', 'H', 'H']],
    'equilibrium_value': 104.45020605234907,
    'force_constant': 3.7937183846251475e-23,
    'equilibrium_value_UB': 1.5140000000000001e-10,
    'force_constant_UB': 4.9816171212814925e-19,
    'name': 'UreyBradleyAngle',
    'type': 'angle',
    'functional_form': 'urey_bradley',
}
data_ureybradley_angle = (
    UreyBradleyAngle,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 1.823 * ureg.radian,
        'force_constant': 75 * ureg.kJ / MOL / ureg.radian**2,
        'equilibrium_value_UB': 1.514 * ureg.angstrom,
        'force_constant_UB': 300 * ureg.kJ / MOL / ureg.m**2,
    },
    results_ureybradley_angle,
)

# polynomial
results_polynomial_angle = {
    'n_interactions': 2,
    'n_particles': 3,
    'particle_indices': [[0, 1, 2], [3, 4, 5]],
    'particle_labels': [['O', 'H', 'H'], ['O', 'H', 'H']],
    'equilibrium_value': 104.45020605234907,
    'force_constants': [
        {
            'name': 'k_2',
            'value': 1.245404280320373e-22,
            'unit': 'kilojoule / radian ** 2',
            'exponent': 2,
        },
        {
            'name': 'k_3',
            'value': -3.3210780808543285e-24,
            'unit': 'kilojoule / radian ** 4',
            'exponent': 3,
        },
        {
            'name': 'k_4',
            'value': 7.472425681922239e-24,
            'unit': 'kilojoule / radian ** 4',
            'exponent': 4,
        },
    ],
    'name': 'PolynomialAngle',
    'type': 'angle',
    'functional_form': 'polynomial',
}
data_polynomial_angle = (
    PolynomialAngle,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'equilibrium_value': 1.823 * ureg.radian,
        'force_constants': [
            PolynomialForceConstant(
                name='k_2',
                exponent=2,
                value=75 / MOL,
                unit=str(ureg.kJ / ureg.radian**2),
            ),
            PolynomialForceConstant(
                name='k_3',
                exponent=3,
                value=-2.0 / MOL,
                unit=str(ureg.kJ / ureg.radian**4),
            ),
            PolynomialForceConstant(
                name='k_4',
                exponent=4,
                value=4.5 / MOL,
                unit=str(ureg.kJ / ureg.radian**4),
            ),
        ],
    },
    results_polynomial_angle,
)

# tabulated
results_tabulated_angle = {
    'n_interactions': 2,
    'n_particles': 3,
    'particle_indices': [[0, 1, 2], [3, 4, 5]],
    'particle_labels': [['O', 'H', 'H'], ['O', 'H', 'H']],
    'bins': [
        92.99105014973262,
        94.19727699638311,
        95.40350384303362,
        96.60973068968411,
        97.81595810929241,
        99.02218495594292,
        100.22841180259343,
        101.43463864924392,
        102.64086549589443,
        103.84709234254493,
        105.05331976215322,
        106.25954660880373,
        107.46577345545423,
        108.67200030210473,
        109.87822714875524,
        111.08445399540572,
        112.29068141501403,
        113.49690826166454,
        114.70313510831502,
        115.90936195496555,
    ],
    'energies': [
        2.483908793147618e-21,
        1.9871270690593166e-21,
        1.5455433097527105e-21,
        1.1591575152954348e-21,
        8.279695415419479e-22,
        5.51979703171419e-22,
        3.311878298023737e-22,
        1.6559392146862525e-22,
        5.519797819553163e-23,
        0.0,
        0.0,
        5.519797819553239e-23,
        1.6559392146862525e-22,
        3.311878298023748e-22,
        5.519797031714216e-22,
        8.2796954154194335e-22,
        1.1591575152954338e-21,
        1.545543309752715e-21,
        1.98712706905931e-21,
        2.483908793147623e-21,
    ],
    'forces': [
        4.347281042004185e-22,
        3.8896725108092945e-22,
        3.432063979614403e-22,
        2.974455448419512e-22,
        2.516846920122808e-22,
        2.0592383889279167e-22,
        1.6016298577330257e-22,
        1.1440213265381345e-22,
        6.864127953432432e-23,
        2.288042641483519e-23,
        -2.288042641483519e-23,
        -6.864127953432432e-23,
        -1.1440213265381345e-22,
        -1.6016298577330257e-22,
        -2.0592383889279167e-22,
        -2.516846920122808e-22,
        -2.974455448419512e-22,
        -3.432063979614403e-22,
        -3.8896725108092945e-22,
        -4.347281042004185e-22,
    ],
    'name': 'TabulatedAngle',
    'type': 'angle',
    'functional_form': 'tabulated',
}
data_tabulated_angle = (
    TabulatedAngle,
    n_interactions,
    n_particles,
    particle_labels,
    particle_indices,
    {
        'bins': [
            1.623,
            1.64405263,
            1.66510526,
            1.68615789,
            1.70721053,
            1.72826316,
            1.74931579,
            1.77036842,
            1.79142105,
            1.81247368,
            1.83352632,
            1.85457895,
            1.87563158,
            1.89668421,
            1.91773684,
            1.93878947,
            1.95984211,
            1.98089474,
            2.00194737,
            2.023,
        ]
        * ureg.radian,
        'forces': [  # ! This time input the forces and test the auto-generation of energies
            15.0,
            13.42105263,
            11.84210526,
            10.26315789,
            8.68421053,
            7.10526316,
            5.52631579,
            3.94736842,
            2.36842105,
            0.78947368,
            -0.78947368,
            -2.36842105,
            -3.94736842,
            -5.52631579,
            -7.10526316,
            -8.68421053,
            -10.26315789,
            -11.84210526,
            -13.42105263,
            -15.0,
        ]
        * ureg.kJ
        / MOL
        / ureg.radian,
    },
    results_tabulated_angle,
)


@pytest.mark.parametrize(
    'potential_class, n_interactions, n_particles, particle_labels, particle_indices, parameters, results',
    [
        data_harmonic_bond,
        data_cubic_bond,
        data_polynomial_bond,
        data_morse_bond,
        data_fene_bond,
        data_tabulated_bond,
        data_custom_bond,
        data_harmonic_angle,
        data_cosine_angle,
        data_restrictedcosine_angle,
        data_fourier_angle,
        data_ureybradley_angle,
        data_polynomial_angle,
        data_tabulated_angle,
    ],
)
def test_potentials(
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
    sec_FF.contributions[-1].normalize(EntryArchive, logger)  # BoundLogger)

    potential_dict = sec_FF.contributions[-1].m_to_dict()
    potential_dict = {  # ! The dev is required to add new results to the dictionary, this will not be caught by the test!
        key: value for key, value in potential_dict.items() if key in results
    }
    print(potential_dict)
    print(results)
    # assert 1 == 2
    assert_dict_equal(potential_dict, results)
