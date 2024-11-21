"""
Tests user emissivity input to pamtra profile.
"""

from pathlib import Path
import sys

import numpy as np

src_path = Path(__file__).resolve().parent.parent / "python"
sys.path.insert(0, str(src_path))

import pyPamtra


def test_emissivity():
    """
    Consistency checks with user emissivity input. Mainly, this test compares
    if the input emissivity is equal to the output emissivity and if the
    simulated brightness temperature is consistent with the input emissivity.
    For this, all simulations are performed under specular reflection only,
    which allow simple emissivity calculations from surface temperature and
    brightness temperature. Lambertian simulations always use the emissivity
    of the first polarization and angle index for all angles and polarizations,
    which makes consistency checks not meaningful.

    The emissivity is recomputed for each simulation for verification using the
    simulation at the surface:

    tu = e * ts + (1 - e) * td

    e = (tu - td) / (ts - td)

    Overview of tests:

    - No emissivity input (default value of 0.6)
    - Scalar emissivity
    - 1D emissivity vector (angle) with one angle unique
    - 1D emissivity vector (angle) with all angles unique
    - 2D emissivity vector (frequency, angle)
    - 3D emissivity vector (polarization, frequency, angle)

    Emissivity input description:

    When only providing a scalar or vector of length 1 as emissivity, it is
    used for all positions, polarizations, frequencies, and angles. When
    providing a vector with one dimension, it must have the length of 16, i.e.,
    the angles. When providing a 2D array, it must have the shape of
    (frequency, angles=16) and so on. If the shape is not compatible with the
    broadcast rules, an error is raised.

    Dimensions:
    tb: ['gridx', 'gridy', 'outlevels', 'angles', 'frequency', 'passive_npol']
    e: ['ngridx', 'ngridy', 'passive_npol', 'frequency', 'angles_half']

    Order of angles:

    - tb (pamtra): starting at zenith and increasing to nadir
    - e (pamtra): starting near horizon and increasing to nadir
    - e (input): like tb (pamtra) starting zenith and increasing near horizon
    """

    n_layers = 10
    ts = 280
    freq = [20, 50, 90]
    rtol = 1e-3
    atol = 1e-5

    # compability with nmlSet definition of current version
    pam, profile = pamtra_setup(n_layers, ts)
    pam.nmlSet["emissivity"] = 0.9
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=0.9, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=0.9, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=0.9, ts=ts, rtol=rtol, atol=atol)

    # default emissivity
    pam, profile = pamtra_setup(n_layers, ts)
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=0.6, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=0.6, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=0.6, ts=ts, rtol=rtol, atol=atol)

    # set emissivity to 0.92 (other than default)
    pam, profile = pamtra_setup(n_layers, ts)
    emissivity = 0.92
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)

    # use nmlSet when both nmlSet and profile definition are present (1)
    pam, profile = pamtra_setup(n_layers, ts)
    pam.nmlSet["emissivity"] = 0.3
    emissivity = 0.92
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=0.3, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=0.3, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=0.3, ts=ts, rtol=rtol, atol=atol)

    # use nmlSet when both nmlSet and profile definition are present (1)
    pam, profile = pamtra_setup(n_layers, ts)
    emissivity = 0.92
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.nmlSet["emissivity"] = 0.3
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=0.3, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=0.3, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=0.3, ts=ts, rtol=rtol, atol=atol)

    # set constant emissivity for all angles except for second (16 angles)
    pam, profile = pamtra_setup(n_layers, ts)
    emissivity = np.linspace(0, 0, 16)
    emissivity[1] = 1  # modify emissivity near nadir
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)

    # set angular dependence of emissivity (16 angles)
    pam, profile = pamtra_setup(n_layers, ts)
    emissivity = np.linspace(0.5, 0.9, 16)
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)

    # set angular and frequency dependence of emissivity
    pam, profile = pamtra_setup(n_layers, ts)
    e_f0 = np.linspace(0.7, 0.9, 16)
    e_f1 = np.linspace(0.6, 0.8, 16)
    e_f2 = np.linspace(0.5, 0.7, 16)
    emissivity = np.array([e_f0, e_f1, e_f2])
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)

    # set angular, frequency and polarization dependence of emissivity
    pam, profile = pamtra_setup(n_layers, ts)
    e_f0_v = np.linspace(0.7, 0.9, 16)
    e_f1_v = np.linspace(0.6, 0.8, 16)
    e_f2_v = np.linspace(0.5, 0.7, 16)
    e_f0_h = np.linspace(0.7, 0.5, 16)
    e_f1_h = np.linspace(0.6, 0.4, 16)
    e_f2_h = np.linspace(0.5, 0.3, 16)
    emissivity = np.array([[e_f0_v, e_f1_v, e_f2_v], [e_f0_h, e_f1_h, e_f2_h]])
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)

    # list input
    pam, profile = pamtra_setup(n_layers, ts)
    emissivity = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    profile["sfc_emissivity"] = emissivity
    pam.createProfile(**profile)
    pam.runPamtra(freq)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=1)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)
    pam.runParallelPamtra(freq, pp_deltaF=2)
    emis_check(pam=pam, emissivity=emissivity, ts=ts, rtol=rtol, atol=atol)


def pamtra_setup(n_layers, ts):
    """
    PAMTRA setting for all scenarios.
    """

    pam = pyPamtra.pyPamtra()
    pam.nmlSet["active"] = False
    pam.nmlSet["passive"] = True
    pam.df.addHydrometeor(
        (
            "cwc_q",
            -99.0,
            1,
            -99.0,
            -99.0,
            -99.0,
            -99.0,
            -99.0,
            3,
            1,
            "mono",
            -99.0,
            -99.0,
            2.0,
            1.0,
            2.0e-6,
            8.0e-5,
            "disabled",
            "khvorostyanov01_drops",
            -99.0,
        )
    )
    profile = {
        "groundtemp": np.array([[ts]]),
        "press": np.linspace(1000, 100, n_layers)[np.newaxis, np.newaxis, :],
        "hgt": np.linspace(1000, 10000, n_layers)[np.newaxis, np.newaxis, :],
        "relhum": np.linspace(70, 10, n_layers)[np.newaxis, np.newaxis, :],
        "temp": np.linspace(300, 260, n_layers)[np.newaxis, np.newaxis, :],
        "obs_height": np.array([0])[np.newaxis, np.newaxis, :],
        "sfc_refl": "S",
        "sfc_type": -1,
    }

    return pam, profile


def emis_check(pam, emissivity, ts, rtol, atol):
    """
    Basic check of PAMTRA output and input for the emissivity scenarios.
    Note that emissivity result has higher incidence angles first.
    """

    td = pam.r["tb"][0, 0, 0, -1:-17:-1, :, :]  # zenith first
    tu = pam.r["tb"][0, 0, 0, :16, :, :]  # nadir first
    e = (tu - td) / (ts - td)  # 0 incidence angle first
    e = e.transpose(2, 1, 0)  # same order as emissivity input and result
    assert np.isclose(
        pam.r["emissivity"][..., ::-1], emissivity, rtol=rtol, atol=atol
    ).all()
    if not isinstance(emissivity, np.ndarray):
        emissivity = np.array([emissivity])
    assert np.isclose(e, emissivity, rtol=rtol, atol=atol).all()
