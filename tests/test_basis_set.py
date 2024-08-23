from typing import Any, Optional

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
    MuffinTinRegion,
    generate_apw,
)
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem
from tests.conftest import refs_apw

from . import logger


def test_cutoff():
    """Test the quantitative results when computing certain plane-wave cutoffs."""
    p_unit = '1 / angstrom'
    ref_cutoff_radius = 1.823 * ureg(p_unit)
    pw = APWPlaneWaveBasisSet(cutoff_energy=500 * ureg('eV'))
    assert np.isclose(
        pw.cutoff_radius.to(p_unit).magnitude, ref_cutoff_radius.magnitude, atol=1e-3
    )  # reference computed by ChatGPT 4o

    pw.set_cutoff_fractional(1 / ref_cutoff_radius, logger)
    assert np.isclose(pw.cutoff_fractional, 1, rtol=1e-2)


@pytest.mark.parametrize(
    'ref_cutoff_fractional, cutoff_energy, mt_radius',
    [
        (None, None, None),
        (None, 500.0, None),
        (None, None, 1.0),
        (1.823, 500.0, 1.0),
    ],
)
def test_cutoff_failure(ref_cutoff_fractional, cutoff_energy, mt_radius):
    """Test modes where `cutoff_fractional` is not computed."""
    pw = APWPlaneWaveBasisSet(cutoff_energy=cutoff_energy * ureg('eV') if cutoff_energy else None)
    if mt_radius is not None:
        pw.set_cutoff_fractional(mt_radius * ureg.angstrom, logger)

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
):
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
def test_apw_base_orbital_normalize(n_terms: Optional[int], ref_n_terms: Optional[int]):
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
def test_apw_orbital(ref_type: Optional[str], do: Optional[int]):
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
):
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
):
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
