"""Wiring tests for qsm_forward.degrade (tiny arrays; the calibration itself is documented, not tested)."""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from qsm_forward import chisep, degrade
from qsm_forward.main import parse_overrides

SHAPE = (12, 12, 12)
TES = [0.004, 0.008, 0.012, 0.016]
SE_TES = [0.01, 0.03, 0.05]


def _phantom(seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(SHAPE, bool)
    mask[2:10, 2:10, 2:10] = True
    chi = np.where(mask, rng.normal(0.0, 0.05, SHAPE), 0.0)
    gt = {"chimap": chi, "localfield": 0.1 * chi, "mask": mask, "r2prime": np.where(mask, 7.0, 0.0)}
    m0 = np.where(mask, 1.0, 0.2)
    acq = {"magnitudes": [m0 * np.exp(-30 * t) for t in TES], "tes": TES, "b0": 3.0, "voxel": [1.0, 1.0, 1.0],
           "peak_snr": 50.0, "se_magnitudes": [m0 * np.exp(-20 * t) for t in SE_TES], "se_tes": SE_TES}
    return gt, acq


def test_levels_and_validation():
    assert degrade.LEVELS == ("L0", "L1", "L2", "L3")
    assert not degrade.level_params("L0").noise
    assert degrade.level_params("L3", chi_offset_ppm=0.01).chi_offset_ppm == 0.01
    with pytest.raises(KeyError):
        degrade.level_params("L9")
    with pytest.raises(ValueError):
        degrade.DegradationParams(reference="csf")
    with pytest.raises(ValueError):
        degrade.DegradationParams(noise=True, noise_cone_threshold=0.0)
    with pytest.raises(ValueError):
        chisep.ChisepConfig(degrade_levels=("L5",))


def test_noise_is_seeded_and_shared_across_levels():
    gt, acq = _phantom()
    p1 = degrade.DegradationParams()
    a = degrade.degrade(gt, acq, p1, seed=3)
    b = degrade.degrade(gt, acq, p1, seed=3)
    c = degrade.degrade(gt, acq, p1, seed=4)
    np.testing.assert_array_equal(a["localfield"], b["localfield"])
    assert not np.allclose(a["localfield"], c["localfield"])
    # an extra additive term leaves the noise realisation untouched
    d = degrade.degrade(gt, acq, degrade.DegradationParams(chi_offset_ppm=0.02), seed=3)
    np.testing.assert_allclose(d["chimap"][gt["mask"]] - a["chimap"][gt["mask"]], 0.02, atol=1e-9)
    np.testing.assert_array_equal(d["localfield"], a["localfield"])


def test_field_noise_matches_analytic_sd():
    gt, acq = _phantom()
    big = np.ones((40, 40, 40), bool)
    mags = [np.full(big.shape, np.exp(-30 * t)) for t in TES]
    sig = degrade.echo_sigmas(mags, 50.0)
    n_f, n_e = degrade.field_noise(mags, TES, 3.0, sig, big, np.random.default_rng(0))
    a2 = [max(m.max() ** 2 - 2 * s * s, s * s) for m, s in zip(mags, sig)]
    sd = 1 / np.sqrt(sum(a / (s * s) * t * t for a, s, t in zip(a2, sig, TES))) / (2 * np.pi * degrade.GAMMA_BAR * 3.0)
    assert abs(n_f.std() / sd - 1) < 0.05
    assert n_e.shape == big.shape + (len(TES),)
    assert n_e[..., 0].std() > n_e[..., -1].std()   # short echoes are the noisiest in field units


def test_noiseless_dataset_has_no_noise():
    gt, acq = _phantom()
    acq["peak_snr"] = float("inf")
    r = degrade.degrade(gt, acq, degrade.DegradationParams(), seed=0)
    np.testing.assert_allclose(r["localfield"], gt["localfield"])


def test_transfer_and_additive_terms():
    gt, acq = _phantom()
    p = degrade.DegradationParams(noise=False, chi_gain=0.5)
    r = degrade.degrade(gt, acq, p, seed=0)
    np.testing.assert_allclose(r["chimap"][gt["mask"]], 0.5 * gt["chimap"][gt["mask"]], atol=1e-9)
    assert "r2prime" in r and np.allclose(r["r2prime"][gt["mask"]], 7.0)   # oracle kept without noise
    p3 = degrade.DegradationParams(noise=False, reference="mean", mask_erosion_mm=1.0, background_poly_order=2,
                                   background_field_ppm=0.001, background_chi_ppm=0.002,
                                   edge_error_ppm=0.01, edge_error_width_mm=2.0)
    r3 = degrade.degrade(gt, acq, p3, seed=0)
    m = r3["mask"] > 0
    assert m.sum() < gt["mask"].sum()
    assert np.all(r3["chimap"][~m] == 0)
    assert abs(r3["chimap"][m].mean()) < 1e-9   # mean reference, applied after the additive terms


def test_echoes_keep_the_ground_truth_echo_dependence():
    gt, acq = _phantom()
    shift = np.linspace(-0.01, 0.01, len(TES))
    gt["chimap_echoes"] = gt["chimap"][..., None] + shift * gt["mask"][..., None]
    r = degrade.degrade(gt, acq, degrade.DegradationParams(noise=False, chi_offset_ppm=0.003), seed=0)
    d = r["chimap_echoes"] - r["chimap"][..., None]
    np.testing.assert_allclose(d[gt["mask"]], np.broadcast_to(shift, d[gt["mask"]].shape), atol=1e-6)
    # with a transfer, each echo passes the same one: the echo mean equals the transferred mean
    r = degrade.degrade(gt, acq, degrade.DegradationParams(noise=False, cone_lambda=0.01), seed=0)
    np.testing.assert_allclose(r["chimap_echoes"].mean(-1), r["chimap"], atol=1e-6)


def test_r2prime_fit_is_exact_for_mono_exponential():
    gt, acq = _phantom()
    r2p = degrade.fit_r2prime(acq["magnitudes"], TES, acq["se_magnitudes"], SE_TES, gt["mask"], "arlo")
    np.testing.assert_allclose(r2p[gt["mask"]], 10.0, rtol=2e-2)   # ARLO's Simpson rule: ~1% at these spacings
    r2p = degrade.fit_r2prime(acq["magnitudes"], TES, acq["se_magnitudes"], SE_TES, gt["mask"], "loglinear")
    np.testing.assert_allclose(r2p[gt["mask"]], 10.0, rtol=1e-9)


def test_measure_and_fit_transfer_recover_a_gain():
    rng = np.random.default_rng(1)
    m = np.ones((24, 24, 24), bool)
    x = rng.normal(size=m.shape)
    meas = degrade.measure_transfer(0.6 * x, x, m, [1, 1, 1])
    assert np.nanmax(np.abs(meas["H_radial"] - 0.6)) < 1e-6
    fit, r2 = degrade.fit_chi_transfer(meas, fix={"cone_lambda": 1e-5, "chi_lowk_loss": 0.0, "chi_highk_cutoff": 50.0,
                                                  "chi_lowk_width": 0.02})
    assert abs(fit["chi_gain"] - 0.6) < 0.01


def _tiny_bids(root):
    gt, acq = _phantom()
    raw = os.path.join(root, "sub-1", "anat")
    der = os.path.join(root, "derivatives", "qsm-forward", "sub-1", "anat")
    os.makedirs(raw)
    os.makedirs(der)
    aff = np.eye(4)
    save = lambda a, p: nib.save(nib.Nifti1Image(np.asarray(a, np.float32), aff), p)
    for i, (m, t) in enumerate(zip(acq["magnitudes"], TES), 1):
        save(m, os.path.join(raw, f"sub-1_echo-{i}_part-mag_MEGRE.nii"))
        json.dump({"EchoTime": t, "MagneticFieldStrength": 3.0, "B0_dir": [0, 0, 1], "PeakSNR": 50.0},
                  open(os.path.join(raw, f"sub-1_echo-{i}_part-mag_MEGRE.json"), "w"))
    for i, (m, t) in enumerate(zip(acq["se_magnitudes"], SE_TES), 1):
        save(m, os.path.join(raw, f"sub-1_echo-{i}_part-mag_MESE.nii"))
        json.dump({"EchoTime": t, "MagneticFieldStrength": 3.0, "PeakSNR": 50.0},
                  open(os.path.join(raw, f"sub-1_echo-{i}_part-mag_MESE.json"), "w"))
    save(gt["chimap"], os.path.join(der, "sub-1_Chimap.nii"))
    save(gt["localfield"], os.path.join(der, "sub-1_fieldmap-local.nii"))
    save(gt["mask"], os.path.join(der, "sub-1_mask.nii"))
    save(gt["r2prime"], os.path.join(der, "sub-1_R2prime.nii"))
    save(0.001 * gt["mask"], os.path.join(der, "sub-1_desc-subvoxel_fieldmap.nii"))
    return der


def test_degrade_bids_writes_named_derivatives(tmp_path):
    der = _tiny_bids(str(tmp_path))
    written = degrade.degrade_bids(str(tmp_path), ["L0", "L1", "L3"], seed=5,
                                   overrides={"mask_erosion_mm": 1.0})
    assert written["L0"] == {}
    for lv in ("L1", "L3"):
        for name in ("fieldmap-local", "Chimap", "Chimap-echoes", "R2prime", "mask"):
            assert os.path.exists(os.path.join(der, f"sub-1_desc-degraded{lv}_{name}.nii"))
        side = json.load(open(os.path.join(der, f"sub-1_desc-degraded{lv}_degradation.json")))
        assert side["Seed"] == 5 and side["DegradationLevel"] == lv
        assert side["Parameters"]["mask_erosion_mm"] == 1.0
        assert ("noise_scale_chi" in side["CalibrationSpread"]) and \
            (("background_chi_ppm" in side["CalibrationSpread"]) == (lv == "L3"))
    assert nib.load(os.path.join(der, "sub-1_desc-degradedL1_Chimap-echoes.nii")).shape == SHAPE + (len(TES),)


def test_cli_overrides():
    assert parse_overrides(["chi_lowk_loss=0.3", "reference=mean", "chi_highk_cutoff=none",
                            "background_poly_order=2", "noise=false"]) == {
        "chi_lowk_loss": 0.3, "reference": "mean", "chi_highk_cutoff": None, "background_poly_order": 2,
        "noise": False}
    with pytest.raises(SystemExit):
        parse_overrides(["nonsense=1"])


def test_subvoxel_field_is_the_weighted_phase_slope():
    class Fake:
        mask = np.zeros((4, 4, 4), bool)
        mask[1:3, 1:3, 1:3] = True
        def gre(self, te, R2):
            w = 2 * np.pi * degrade.GAMMA_BAR * 3.0 * 0.01   # 0.01 ppm at 3 T, rad/s
            return self.mask, np.full(int(self.mask.sum()), np.exp(1j * w * te - 20 * te))
    f = chisep.subvoxel_field(Fake(), TES, 3.0, None)
    np.testing.assert_allclose(f[Fake.mask], 0.01, rtol=1e-6)
    assert np.all(f[~Fake.mask] == 0)
