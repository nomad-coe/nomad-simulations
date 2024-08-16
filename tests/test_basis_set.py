import itertools
from . import logger
from nomad.units import ureg
import numpy as np
import pytest
from typing import Optional, Any

from tests.conftest import apw

from nomad_simulations.schema_packages.basis_set import (
    APWBaseOrbital,
    APWOrbital,
    APWLocalOrbital,
    APWLChannel,
    APWPlaneWaveBasisSet,
    BasisSet,
    BasisSetContainer,
    MuffinTinRegion,
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


@pytest.mark.skip(reason='This function is not meant to be tested directly')
def generate_apw(
    species: dict[str, dict[str, Any]], cutoff: Optional[float] = None
) -> BasisSetContainer:
    """
    Generate a mock APW basis set with the following structure:
    .
    ├── 1 x plane-wave basis set
    └── n x muffin-tin regions
        └── l_max x l-channels
            ├── orbitals
            └── local orbitals
    """
    basis_set_components: list[BasisSet] = []
    if cutoff is not None:
        pw = APWPlaneWaveBasisSet(cutoff_energy=cutoff)
        basis_set_components.append(pw)

    for sp_name, sp in species.items():
        l_max = sp['l_max']
        mt = MuffinTinRegion(
            radius=sp['r'],
            l_max=l_max,
            l_channels=[
                APWLChannel(
                    name=l,
                    orbitals=list(
                        itertools.chain(
                            (APWOrbital(type=orb) for orb in sp['orb_type']),
                            (APWLocalOrbital(type=lo) for lo in sp['lo_type']),
                        )
                    ),
                )
                for l in range(l_max + 1)
            ],
        )
        basis_set_components.append(mt)

    return BasisSetContainer(basis_set_components=basis_set_components)


def test_full_apw():
    ref_apw = generate_apw(
        {
            'A': {
                'r': 1.823,
                'l_max': 2,
                'orb_type': ['apw', 'lapw'],
                'lo_type': ['lo'],
            }
        },
        cutoff=500,
    )
    assert ref_apw.m_to_dict() == apw
