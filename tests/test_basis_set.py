from typing import Any, Optional

from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem
import pytest

from nomad.datamodel.datamodel import EntryArchive
from . import logger
from nomad.units import ureg
import numpy as np

from tests.conftest import refs_apw

from nomad_simulations.schema_packages.basis_set import (
    APWBaseOrbital,
    APWOrbital,
    APWLocalOrbital,
    APWLChannel,
    APWPlaneWaveBasisSet,
    BasisSet,
    BasisSetContainer,
    MuffinTinRegion,
    generate_apw,
)


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


def test_cutoff_failure():
    """Test modes where `cutoff_fractional` is not computed."""
    # missing cutoff_energy
    pw = APWPlaneWaveBasisSet()
    pw.set_cutoff_fractional(ureg.angstrom, logger)
    assert pw.cutoff_fractional is None

    # missing mt_radius
    pw = APWPlaneWaveBasisSet(cutoff_energy=500 * ureg('eV'))
    pw.set_cutoff_fractional(None, logger)
    assert pw.cutoff_fractional is None

    # cutoff_fractional already set
    pw = APWPlaneWaveBasisSet(cutoff_energy=500 * ureg('eV'), cutoff_fractional=1)
    pw.set_cutoff_fractional(ureg.angstrom, logger)
    assert pw.cutoff_fractional == 1


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
                    'orb_type': ['apw'],
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
    assert (
        numerical_settings[0].m_to_dict() == refs_apw[ref_index]
    )  # TODO: add normalization?


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
    orb = APWBaseOrbital(
        energy_parameter=e,
        differential_order=d_o,
    )
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
    'ref_type, n_terms',
    [(None, None), (None, 0), ('apw', 1), ('lapw', 2), ('slapw', 3)],
)
def test_apw_orbital(ref_type: Optional[str], n_terms: Optional[int]):
    orb = APWOrbital(n_terms=n_terms)
    assert orb.n_terms_to_type(orb.n_terms) == ref_type


@pytest.mark.parametrize(
    'ref_n_terms, ref_type, e, d_o, b_o',
    [
        (None, None, [0.0], [], []),  # logically inconsistent
        (1, 'custom', [0.0], [0], [0]),  # custom
        (2, 'lo', 2 * [0.0], [0, 1], [0, 1]),  # lo
        (3, 'LO', 3 * [0.0], [0, 1, 0], [0, 1, 0]),  # LO
    ],
)
def test_apw_local_orbital(
    ref_n_terms: Optional[int],
    ref_type: str,
    e: list[float],
    d_o: list[int],
    b_o: list[int],
):
    orb = APWLocalOrbital(
        energy_parameter=e,
        differential_order=d_o,
        boundary_order=d_o,
    )
    assert orb.get_n_terms() == ref_n_terms
    assert orb.bo_terms_to_type(orb.boundary_order) == ref_type
