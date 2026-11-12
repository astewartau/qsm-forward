"""Wiring tests for qsm_forward.phantoms (small synthetic heads; fast)."""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from qsm_forward import ReconParams, phantoms

SHAPE = (40, 40, 32)
VOX = (1.5, 1.5, 1.5)
TES = [0.004, 0.012, 0.020, 0.028]


def _synthetic_head(seed=0):
    """Ellipsoidal head: brain, a CSF/scalp layer, a dark skull shell, a frontal air cavity."""
    rng = np.random.default_rng(seed)
    g = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) * v for n, v in zip(SHAPE, VOX)], indexing="ij")
    r = np.sqrt((g[0] / 26) ** 2 + (g[1] / 28) ** 2 + (g[2] / 22) ** 2)
    brain = r < 0.62
    skull = (r >= 0.72) & (r < 0.84)
    head = r < 1.0
    sinus = (np.sqrt((g[0] / 8) ** 2 + ((g[1] - 21) / 6) ** 2 + ((g[2] + 4) / 6) ** 2) < 1) & head & ~brain
    s0 = np.where(head, 1000.0, 0.0)
    s0[skull] = 100.0  # cortical bone + marrow: dark, above the noise
    s0[sinus] = 0.0
    r2 = np.where(brain, 25.0, 30.0)
    mag = np.stack([s0 * np.exp(-r2 * t) for t in TES], -1) + rng.normal(0, 2, SHAPE + (len(TES),))
    mag = np.abs(mag).astype(np.float32)
    chi = np.where(brain, 0.02 * np.sin(g[0] / 3.0) + rng.normal(0, 0.01, SHAPE), 0.0)
    # V-SHARP-like eroded support
    support = brain & (r < 0.58)
    chi = np.where(support, chi, 0.0)
    return chi, mag, brain, support, head, skull, sinus


@pytest.fixture(scope="module")
def built():
    chi, mag, brain, support, head, skull, sinus = _synthetic_head()
    ph = phantoms.build_invivo_phantom(chi, mag, TES, VOX, brain, chi_support=support, TR=0.035,
                                       flip_angle=15, neck_mm=6.0, name="synthetic")
    return ph, dict(chi=chi, mag=mag, brain=brain, support=support, head=head, skull=skull, sinus=sinus)


def test_fit_r2star_recovers_rate():
    s0, r2 = 500.0, 40.0
    mag = np.full((3, 3, 3, len(TES)), 0.0)
    for i, t in enumerate(TES):
        mag[..., i] = s0 * np.exp(-r2 * t)
    S0, R2 = phantoms.fit_r2star(mag, TES)
    assert np.allclose(R2, r2, rtol=1e-6)
    assert np.allclose(S0, s0, rtol=1e-6)


def test_head_model_classes(built):
    ph, src = built
    n = ph.meta["neck_slices"]
    seg = ph.seg[:, :, n:]  # drop the neck extension (index 0 is inferior for the identity affine)
    assert np.all(seg[src["brain"]] == phantoms.LABEL_BRAIN)
    assert np.mean(seg[src["skull"]] == phantoms.LABEL_BONE) > 0.8
    assert np.mean(seg[src["sinus"]] == phantoms.LABEL_AIR) > 0.5
    assert np.all(seg[~src["head"]] == phantoms.LABEL_AIR)
    soft = src["head"] & ~src["brain"] & ~src["skull"] & ~src["sinus"]
    assert np.mean(seg[soft] == phantoms.LABEL_SOFT_TISSUE) > 0.7


def test_truth_values_and_extension(built):
    ph, src = built
    n = ph.meta["neck_slices"]
    assert n == 4 and ph.chi.shape == SHAPE[:2] + (SHAPE[2] + n,)
    assert ph.fov[2] == [n, SHAPE[2] + n]
    assert np.isclose(ph.affine[2, 3], -n * VOX[2])
    chi = ph.chi[:, :, n:]
    assert np.allclose(chi[ph.seg[:, :, n:] == phantoms.LABEL_AIR], phantoms.CHI_AIR_PPM)
    assert np.allclose(chi[ph.seg[:, :, n:] == phantoms.LABEL_BONE], phantoms.CHI_BONE_PPM)
    # truth defined on the whole brain mask, gap filled and flagged
    assert np.all(ph.mask[:, :, n:] == src["brain"])
    assert np.all(ph.filled[:, :, n:] == (src["brain"] & ~src["support"]))
    assert abs(np.median(chi[src["brain"]])) < 1e-6
    # extension carries no brain and no source χ
    assert not ph.mask[:, :, :n].any()
    assert not np.any(ph.seg[:, :, :n] == phantoms.LABEL_BRAIN)
    # air has no signal
    assert np.all(ph.M0[ph.seg == phantoms.LABEL_AIR] == 0)


def test_clean_chi_removes_cone_excess_and_spikes():
    rng = np.random.default_rng(1)
    shape, vox = (48, 48, 48), (1.0, 1.0, 1.0)
    mask = np.zeros(shape, bool)
    mask[6:-6, 6:-6, 6:-6] = True
    chi = rng.normal(0, 0.02, shape)
    # add streaks: energy concentrated on the magic cone
    kr, dev = phantoms._cone_deviation(shape, vox, (0, 0, 1))
    streak = np.real(np.fft.ifftn(np.where(dev < 2.0, 1.0, 0.0) * np.fft.fftn(rng.normal(0, 0.2, shape))))
    chi = np.where(mask, chi + streak, 0.0)
    chi[24, 24, 24] += 3.0
    out, info = phantoms.clean_chi(chi, mask, mask, vox, denoise=None)
    assert info["cone_excess_db_median"] > 3
    assert info["spike_fraction"] > 0
    assert abs(out[24, 24, 24]) < 0.5
    _, info2 = phantoms.clean_chi(out, mask, mask, vox, denoise=None, despike_sd=None)
    assert info["cone_excess_db_median_after"] < 1.5
    assert info2["cone_excess_db_median"] < 1.5
    # a cone deficit is left alone by "clamp" and filled by "equalise"
    hole = np.real(np.fft.ifftn(np.where(dev < 6.0, 0.2, 1.0) * np.fft.fftn(np.where(mask, rng.normal(0, 0.02, shape), 0.0))))
    hole = np.where(mask, hole, 0.0)
    _, i_c = phantoms.clean_chi(hole, mask, mask, vox, denoise=None, despike_sd=None)
    _, i_e = phantoms.clean_chi(hole, mask, mask, vox, denoise=None, despike_sd=None, cone="equalise")
    assert i_c["cone_excess_db_median"] < -3 and i_c["cone_excess_db_median_after"] < -3
    assert i_e["cone_excess_db_median_after"] > i_c["cone_excess_db_median_after"] + 2
    with pytest.raises(ValueError):
        phantoms.clean_chi(chi, mask, mask, vox, cone="fill")
    _, info3 = phantoms.clean_chi(chi, mask, mask, vox, denoise="wiener")
    assert "spectral_exponent" in info3
    out_tv, _ = phantoms.clean_chi(chi, mask, mask, vox, denoise="tv", tv_weight=0.01)
    assert np.std(out_tv[mask]) < np.std(out[mask])
    with pytest.raises(ValueError):
        phantoms.clean_chi(chi, mask, mask, vox, denoise="bm4d")


def test_insert_sources_records_and_labels(built):
    ph, _ = built
    specs = [phantoms.SourceSpec("calcification", count=2, radius_mm=2.0, chi_ppm=-2.0),
             phantoms.SourceSpec("microbleed", count=2, radius_mm=1.5, chi_ppm=1.5),
             phantoms.SourceSpec("vein", count=1, radius_mm=1.2, length_mm=10.0, gap_mm=1.0),
             phantoms.SourceSpec("ms_lesion", count=1, radius_mm=2.5, rim_probability=1.0,
                                 central_vein_probability=1.0, region="brain")]
    out = phantoms.insert_sources(ph, specs, seed=3)
    assert len(out.sources) == 6
    ids = sorted(np.unique(out.source_labels[out.source_labels > 0]).tolist())
    assert ids == [1, 2, 3, 4, 5, 6]
    calc = out.sources[0]
    core = out.source_labels == calc["id"]
    assert np.median(out.chi[core]) < -1.5
    assert np.all(out.seg[core] == phantoms.LABEL_CALCIFICATION)
    assert np.all(out.R2star[core] > ph.R2star[core])
    assert np.all(out.M0[core] < ph.M0[core])
    assert calc["moment_ppm_ml"] < 0 and out.sources[2]["moment_ppm_ml"] > 0
    assert out.sources[5].get("rim_delta_chi_ppm", 0) > 0 and out.sources[5]["central_vein"]
    # sources stay inside the brain and the original is untouched
    assert not np.any(out.source_labels[~ph.mask])
    assert ph.source_labels is None and len(ph.sources) == 0
    kinds = phantoms.source_kind_map(out)
    assert set(np.unique(kinds)) <= set(range(len(phantoms.KINDS) + 1))
    # same seed, same phantom
    again = phantoms.insert_sources(ph, specs, seed=3)
    assert np.array_equal(again.chi, out.chi)
    with pytest.raises(KeyError):
        phantoms.insert_sources(ph, "no-such-preset")


def test_streak_heavy_preset_exists():
    kinds = [s.kind for s in phantoms.PRESETS["streak-heavy"]]
    assert {"calcification", "haemorrhage", "microbleed", "vein"} <= set(kinds)


def test_cone_restricted_error():
    rng = np.random.default_rng(2)
    shape, vox = (32, 32, 32), (1.0, 1.0, 1.0)
    mask = np.ones(shape, bool)
    truth = rng.normal(0, 0.05, shape)
    kr, dev = phantoms._cone_deviation(shape, vox, (0, 0, 1))
    streak = np.real(np.fft.ifftn(np.where(dev < 1.5, 1.0, 0.0) * np.fft.fftn(rng.normal(0, 0.1, shape))))
    s, frac = phantoms.cone_restricted_error(truth + streak, truth, mask, vox)
    assert frac > 0.9
    assert np.corrcoef(s.ravel(), streak.ravel())[0, 1] > 0.95
    s0, frac0 = phantoms.cone_restricted_error(truth, truth, mask, vox)
    assert frac0 == 0.0 and not s0.any()


def test_save_load_simulate_roundtrip(tmp_path, built):
    ph, _ = built
    ph = phantoms.insert_sources(ph, [phantoms.SourceSpec("calcification", count=1, radius_mm=2.0)], seed=0)
    pdir = str(tmp_path / "phantom")
    ph.save(pdir)
    back = phantoms.Phantom.load(pdir)
    assert np.array_equal(back.source_labels, ph.source_labels)
    assert back.fov == ph.fov and back.sources[0]["kind"] == "calcification"
    rp = ReconParams(subject="1", TEs=np.array(TES), B0=3, voxel_size=np.array(VOX), peak_snr=100,
                     random_seed=1, TR=0.035, flip_angle=15)
    out = phantoms.simulate(back, rp, str(tmp_path / "bids"))
    assert out["cropped"]
    for key in ("chi", "mask", "totalfield", "localfield"):
        img = nib.load(out[key])
        assert img.shape == SHAPE, key
    mag = nib.load(os.path.join(out["anat"], "sub-1_echo-1_part-mag_MEGRE.nii"))
    assert mag.shape == SHAPE
    lab = nib.load(os.path.join(out["derivatives"], "sub-1_desc-sources_dseg.nii")).get_fdata()
    assert lab.max() == 1
    # the cropped truth is the phantom's χ inside its source FOV
    chi = nib.load(out["chi"]).get_fdata()
    n = ph.meta["neck_slices"]
    assert np.allclose(chi, ph.chi[:, :, n:], atol=1e-4)
    meta = json.load(open(os.path.join(out["derivatives"], "sub-1_phantom.json")))
    assert meta["sources"][0]["kind"] == "calcification"


def test_cut_faces_are_extended():
    chi, mag, brain, support, *_ = _synthetic_head()
    cut = (slice(None), slice(None), slice(0, SHAPE[2] - 3))  # the FOV cuts off the top of the head
    ph = phantoms.build_invivo_phantom(chi[cut], mag[cut], TES, VOX, brain[cut], chi_support=support[cut],
                                       neck_mm=6.0, cut_extend_mm=6.0)
    assert ph.meta["extended_faces"] == {"2-": 4, "2+": 4}
    assert ph.chi.shape[2] == SHAPE[2] - 3 + 8
    assert ph.fov[2] == [4, SHAPE[2] - 3 + 4]
    assert not ph.mask[:, :, -4:].any()
