from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pint

import numpy as np
import pytest
from nomad.datamodel.datamodel import EntryArchive
from nomad.units import ureg

from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.basis_set import (
    APWBaseOrbital,
    APWLocalOrbital,
    APWOrbital,
    APWPlaneWaveBasisSet,
    AtomCenteredBasisSet,
    BasisSetContainer,
    MuffinTinRegion,
    PlaneWaveBasisSet,
    generate_apw,
)
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_method import BaseModelMethod, ModelMethod
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem
from tests.conftest import refs_apw

from . import logger


@pytest.mark.parametrize(
    'ref_cutoff_radius, cutoff_energy',
    [
        (None, None),
        (1.823 / ureg.angstrom, 500 * ureg.eV),  # reference computed by ChatGPT 4o
    ],
)
def test_cutoff(
    ref_cutoff_radius: 'pint.Quantity', cutoff_energy: 'pint.Quantity'
) -> None:
    """Test the quantitative results when computing certain plane-wave cutoffs."""
    pw = APWPlaneWaveBasisSet(cutoff_energy=cutoff_energy)
    cutoff_radius = pw.compute_cutoff_radius(cutoff_energy)

    if cutoff_radius is None:
        assert cutoff_radius is ref_cutoff_radius
    else:
        assert np.isclose(
            cutoff_radius.to(ref_cutoff_radius.units).magnitude,
            ref_cutoff_radius.magnitude,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    'mts, ref_mt_r_min',
    [
        ([], None),
        ([None], None),
        ([MuffinTinRegion(radius=1.0 * ureg.angstrom)], 1.0),
        ([MuffinTinRegion(radius=r * ureg.angstrom) for r in (1.0, 2.0, 3.0)], 1.0),
    ],
)
def test_mt_r_min(mts: list[Optional[MuffinTinRegion]], ref_mt_r_min: float) -> None:
    """
    Test the computation of the minimum muffin-tin radius.
    """
    bs = BasisSetContainer(basis_set_components=mts)
    mt_r_min = bs._find_mt_r_min()

    try:
        assert mt_r_min.to('angstrom').magnitude == ref_mt_r_min
    except AttributeError:
        assert mt_r_min is ref_mt_r_min

    bs.basis_set_components.append(APWPlaneWaveBasisSet(cutoff_energy=500 * ureg('eV')))
    bs.normalize(None, logger)

    try:
        assert (
            bs.basis_set_components[-2].mt_r_min.to('angstrom').magnitude
            == ref_mt_r_min
        )
    except (IndexError, AttributeError):
        assert ref_mt_r_min is None


@pytest.mark.parametrize(
    'ref_cutoff_fractional, cutoff_energy, mt_radius',
    [
        (None, None, None),
        (None, 500.0 * ureg.eV, None),
        (None, None, 1.0),
        (1.823, 500.0 * ureg.eV, 1.0 * ureg.angstrom),
    ],
)
def test_cutoff_failure(
    ref_cutoff_fractional: float,
    cutoff_energy: 'pint.Quantity',
    mt_radius: 'pint.Quantity',
) -> None:
    """Test modes where `cutoff_fractional` is not computed."""
    pw = APWPlaneWaveBasisSet(cutoff_energy=cutoff_energy if cutoff_energy else None)
    if mt_radius is not None:
        pw.cutoff_fractional = pw.compute_cutoff_fractional(
            pw.compute_cutoff_radius(cutoff_energy), mt_radius
        )

    if ref_cutoff_fractional is None:
        assert pw.cutoff_fractional is None
    else:
        assert np.isclose(pw.cutoff_fractional, ref_cutoff_fractional, atol=1e-3)


@pytest.mark.parametrize(
    'ref_index, species_def, cutoff',
    [
        (0, {}, None),
        (1, {}, 500.0),
        (
            2,
            {
                '/data/model_system/0/cell/0/atoms_state/0': {
                    'r': 1,
                    'l_max': 2,
                    'orb_d_o': [[0]],
                    'orb_param': [[0.0]],
                }
            },
            500.0,
        ),
    ],
)
def test_full_apw(
    ref_index: int, species_def: dict[str, dict[str, Any]], cutoff: Optional[float]
) -> None:
    """Test the composite structure of APW basis sets."""
    entry = EntryArchive(
        data=Simulation(
            model_system=[
                ModelSystem(
                    cell=[AtomicCell(atoms_state=[AtomsState(chemical_symbol='H')])]
                )
            ],
            model_method=[ModelMethod(numerical_settings=[])],
        )
    )

    numerical_settings = entry.data.model_method[0].numerical_settings
    numerical_settings.append(generate_apw(species_def, cutoff=cutoff))

    # test structure
    assert numerical_settings[0].m_to_dict() == refs_apw[ref_index]


@pytest.mark.parametrize(
    'ref_n_terms, e, d_o',
    [
        (None, None, None),  # unset
        (0, [], []),  # empty
        (None, [0.0], []),  # logically inconsistent
        (1, [0.0], [0]),  # apw
        (2, 2 * [0.0], [0, 1]),  # lapw
    ],
)
def test_apw_base_orbital(ref_n_terms: Optional[int], e: list[float], d_o: list[int]):
    orb = APWBaseOrbital(energy_parameter=e, differential_order=d_o)
    assert orb.get_n_terms() == ref_n_terms


@pytest.mark.parametrize('n_terms, ref_n_terms', [(None, 1), (1, 1), (2, None)])
def test_apw_base_orbital_normalize(
    n_terms: Optional[int], ref_n_terms: Optional[int]
) -> None:
    orb = APWBaseOrbital(
        n_terms=n_terms,
        energy_parameter=[0],
        differential_order=[1],
    )
    orb.normalize(None, logger)
    assert orb.n_terms == ref_n_terms


@pytest.mark.parametrize(
    'ref_type, do',
    [
        (None, None),
        (None, []),
        (None, [0, 0, 1]),
        ('apw', [0]),
        ('lapw', [0, 1]),
        ('slapw', [0, 2]),
    ],
)
def test_apw_orbital(ref_type: Optional[str], do: Optional[int]) -> None:
    orb = APWOrbital(differential_order=do)
    assert orb.do_to_type(orb.differential_order) == ref_type


# ? necessary
@pytest.mark.parametrize(
    'ref_n_terms, e, d_o',
    [
        (None, [0.0], []),
        (1, [0.0], [0]),
        (2, 2 * [0.0], [0, 1]),
        (3, 3 * [0.0], [0, 1, 0]),
    ],
)
def test_apw_local_orbital(
    ref_n_terms: Optional[int],
    e: list[float],
    d_o: list[int],
) -> None:
    orb = APWLocalOrbital(
        energy_parameter=e,
        differential_order=d_o,
    )
    assert orb.get_n_terms() == ref_n_terms


@pytest.mark.parametrize(
    'ref_type, ref_mt_counts, ref_l_counts, species_def, cutoff',
    [
        (
            None,
            [[0, 0, 0, 0, 0]],
            [[[0, 0, 0, 0, 0]]],
            {
                'H': {
                    'r': 1.0,
                    'l_max': 0,
                    'orb_d_o': [],
                    'orb_param': [],
                    'lo_d_o': [],
                    'lo_param': [],
                }
            },
            None,
        ),
        (
            None,
            [[1, 0, 0, 0, 0]],
            [[[1, 0, 0, 0, 0]]],
            {
                'H': {
                    'r': 1.0,
                    'l_max': 0,
                    'orb_d_o': [[0]],
                    'orb_param': [[0.0]],
                    'lo_d_o': [],
                    'lo_param': [],
                }
            },
            None,
        ),
        (
            'APW-like',
            [[0, 0, 0, 0, 1]],
            [[[0, 0, 0, 0, 1]]],
            {
                'H': {
                    'r': 1.0,
                    'l_max': 0,
                    'orb_d_o': [[]],
                    'orb_param': [[]],
                    'lo_d_o': [],
                    'lo_param': [],
                }
            },
            500.0,
        ),
        (
            'APW',
            [[1, 0, 0, 0, 0]],
            [[[1, 0, 0, 0, 0]]],
            {
                'H': {
                    'r': 1.0,
                    'l_max': 1,
                    'orb_d_o': [[0]],
                    'orb_param': [[0.0]],
                    'lo_d_o': [],
                    'lo_param': [],
                }
            },
            500.0,
        ),
        (
            'LAPW',
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            [[[1, 0, 0, 0, 0]], [[0, 1, 0, 0, 0]]],
            {
                'H': {
                    'r': 1.0,
                    'l_max': 0,
                    'orb_d_o': [[0]],
                    'orb_param': [[0.0]],
                    'lo_d_o': [],
                    'lo_param': [],
                },
                'O': {
                    'r': 2.0,
                    'l_max': 0,
                    'orb_d_o': [[0, 1]],
                    'orb_param': [2 * [0.0]],
                    'lo_d_o': [],
                    'lo_param': [],
                },
            },
            500.0,
        ),
        (
            'SLAPW',
            [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]],
            [[[1, 0, 0, 0, 0]], [[0, 1, 1, 0, 0]]],
            {
                'H': {
                    'r': 1.0,
                    'l_max': 0,
                    'orb_d_o': [[0]],
                    'orb_param': [[0.0]],
                    'lo_d_o': [],
                    'lo_param': [],
                },
                'O': {
                    'r': 2.0,
                    'l_max': 2,
                    'orb_d_o': [[0, 1], [0, 2]],
                    'orb_param': 2 * [2 * [0.0]],
                    'lo_d_o': [],
                    'lo_param': [],
                },
            },
            500.0,
        ),
    ],
)
def test_determine_apw(
    ref_type: str,
    ref_mt_counts: list[list[int]],
    ref_l_counts: list[list[list[int]]],
    species_def: dict[str, dict[str, Any]],
    cutoff: Optional[float],
) -> None:
    """Test the L-channel APW structure."""
    ref_keys = ('apw', 'lapw', 'slapw', 'lo', 'other')
    bs = generate_apw(species_def, cutoff=cutoff)

    # test from the bottom up
    for bsc in bs.basis_set_components:
        if isinstance(bsc, MuffinTinRegion):
            l_counts = ref_l_counts.pop(0)
            for l_channel in bsc.l_channels:
                try:
                    assert l_channel._determine_apw() == dict(
                        zip(ref_keys, l_counts.pop(0))
                    )
                except IndexError:
                    pass
            try:
                assert bsc._determine_apw() == dict(zip(ref_keys, ref_mt_counts.pop(0)))
            except IndexError:
                pass
    assert bs._determine_apw() == ref_type


def test_quick_step() -> None:
    """Test the feasibility of describing a QuickStep basis set."""
    entry = EntryArchive(
        data=Simulation(
            model_method=[
                ModelMethod(
                    contributions=[
                        BaseModelMethod(name='kinetic'),
                        BaseModelMethod(name='electron-ion'),
                        BaseModelMethod(name='hartree'),
                    ],
                    numerical_settings=[],
                )
            ],
        )
    )
    numerical_settings = entry.data.model_method[0].numerical_settings
    numerical_settings.append(
        BasisSetContainer(
            # scope='density',
            basis_set_components=[
                AtomCenteredBasisSet(
                    hamiltonian_scope=[
                        entry.data.model_method[0].contributions[0],
                        entry.data.model_method[0].contributions[1],
                    ],
                ),
                PlaneWaveBasisSet(
                    cutoff_energy=500 * ureg.eV,
                    hamiltonian_scope=[entry.data.model_method[0].contributions[2]],
                ),
            ],
        )
    )

    assert numerical_settings[0].m_to_dict() == {
        'm_def': 'nomad_simulations.schema_packages.basis_set.BasisSetContainer',
        'basis_set_components': [
            {
                'm_def': 'nomad_simulations.schema_packages.basis_set.AtomCenteredBasisSet',
                'hamiltonian_scope': [
                    '/data/model_method/0/contributions/0',
                    '/data/model_method/0/contributions/1',
                ],
            },
            {
                'm_def': 'nomad_simulations.schema_packages.basis_set.PlaneWaveBasisSet',
                'hamiltonian_scope': ['/data/model_method/0/contributions/2'],
                'cutoff_energy': (500.0 * ureg.eV).to('joule').magnitude,
            },
        ],
    }
    # TODO: generate a QuickStep generator in the CP2K plugin
