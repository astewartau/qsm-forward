"""Tests for qsm_forward.artefacts (tiny synthetic volumes; every effect, its labels and severity ladder)."""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from qsm_forward import artefacts as A

N = 24
VOX = (1.0, 1.0, 1.0)
TES = np.array([0.004, 0.008, 0.012, 0.016])


def _sphere(r=7.0, n=N, c=None):
    c = (n - 1) / 2 if c is None else c
    x, y, z = np.indices((n, n, n))
    return (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= r * r


def _chi(seed=0):
    m = _sphere()
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    return np.where(m, ndimage.gaussian_filter(rng.standard_normal(m.shape), 1.5) * 0.05, 0.0), m


def _signal(seed=0):
    """Complex 4-echo signal: smooth magnitude inside a sphere plus a smooth field (with phase wraps)."""
    m = _sphere(9.0)
    x, y, z = np.indices(m.shape)
    mag = np.where(m, 1.0 + 0.2 * np.sin(x / 3.0), 0.05)
    field_hz = 40.0 * np.sin(y / 5.0) + 15.0 * (z - N / 2) / N
    phase = 2 * np.pi * field_hz[..., None] * TES
    return mag[..., None] * np.exp(-TES / 0.04) * np.exp(1j * phase), m, field_hz


LADDER_N = 32
LADDER_VOX = (2.0, 2.0, 2.0)  # a 64 mm field of view with a 48 mm 'brain': room for the brain-scale ladders


def _ladder_data(domain):
    from scipy import ndimage
    m = _sphere(12.0, LADDER_N)
    rng = np.random.default_rng(0)
    if domain == "chi":
        return np.where(m, ndimage.gaussian_filter(rng.standard_normal(m.shape), 1.5) * 0.05, 0.0), m
    if domain == "mask":
        return m, m
    x, y, z = np.indices(m.shape)
    fhz = 40.0 * np.sin(y / 5.0) + 15.0 * (z - LADDER_N / 2) / LADDER_N
    if domain == "phase":
        return 2 * np.pi * fhz[..., None] * TES, m
    mag = np.where(m, 1.0 + 0.2 * np.sin(x / 3.0), 0.05)
    return mag[..., None] * np.exp(1j * 2 * np.pi * fhz[..., None] * TES), m


# ---------------------------------------------------------------- registry and labels

@pytest.mark.parametrize("name", sorted(A.EFFECTS))
def test_every_effect_ladder_runs_labels_and_grows(name):
    eff = A.EFFECTS[name]
    assert eff.subscore in A.SUBSCORES
    data, mask = _ladder_data(eff.domain)
    sev = []
    for kw in eff.ladder:
        r = A.apply_effect(name, data, voxel=LADDER_VOX, mask=mask, seed=3, **kw)
        lab = r.label
        for key in ("effect", "taxonomy", "subscore", "severity", "params", "seed"):
            assert key in lab
        assert lab["subscore"] in A.SUBSCORES
        json.dumps(lab)  # serialisable
        out = r.data
        assert np.shape(out)[:3] == np.shape(data)[:3] or name == "gross_axes"
        if r.affected is not None:
            assert r.affected.shape == np.shape(data)[:3]
        sev.append(abs(lab["severity"]["value"]))
    assert all(b >= a - 1e-12 for a, b in zip(sev, sev[1:])), f"{name} severities not increasing: {sev}"
    # same seed, same result
    r1 = A.apply_effect(name, data, voxel=LADDER_VOX, mask=mask, seed=5, **eff.ladder[-1])
    r2 = A.apply_effect(name, data, voxel=LADDER_VOX, mask=mask, seed=5, **eff.ladder[-1])
    np.testing.assert_array_equal(np.asarray(r1.data), np.asarray(r2.data))


def test_write_labels(tmp_path):
    chi, m = _chi()
    labs = [A.gross_sign(chi).label, A.gross_scale(chi, cause="hz_as_ppm", b0=3).label]
    p = A.write_labels(str(tmp_path / "l.json"), labs, {"Note": "x"})
    d = json.load(open(p))
    assert [e["effect"] for e in d["Effects"]] == ["gross_sign", "gross_scale"]
    assert d["Effects"][1]["params"]["factor"] == pytest.approx(42.58 * 3)
    assert d["Note"] == "x"


# ---------------------------------------------------------------- 1. gross

def test_gross_transforms():
    chi, m = _chi()
    np.testing.assert_array_equal(A.gross_sign(chi).data, -chi)
    r = A.gross_scale(chi, factor=1.04)
    assert r.label["subscore"] == "CALIB"
    np.testing.assert_allclose(r.data, chi * 1.04)
    assert A.gross_scale(chi, cause="ppb_as_ppm").label["subscore"] == "GROSS"
    with pytest.raises(ValueError):
        A.gross_scale(chi)
    r = A.gross_axes(chi, flip=(0,), permute=(2, 1, 0))
    np.testing.assert_array_equal(r.data, np.flip(np.transpose(chi, (2, 1, 0)), 0))
    q = A.gross_quantise(chi, step_ppm=0.01, clip_ppm=0.03, mask=m)
    assert np.allclose(np.round(q.data / 0.01), q.data / 0.01)
    assert np.abs(q.data).max() <= 0.03 + 1e-12
    assert 0 <= q.label["clipped_fraction"] <= 1
    nf = A.gross_nonfinite(chi, m, VOX, n_patches=2, radius_mm=3, value="mixed", seed=1)
    assert (~np.isfinite(nf.data)).sum() == nf.label["severity"]["value"] == nf.affected.sum()
    assert not (nf.affected & ~m).any()
    o = A.chi_offset(chi, 0.02, mask=m)
    np.testing.assert_allclose(o.data[m] - chi[m], 0.02)
    assert (o.data[~m] == 0).all()


def test_gross_wrong_b0_zero_when_direction_right():
    chi, m = _chi()
    same = A.gross_wrong_b0(chi, VOX, (0, 0, 1), (0, 0, 1), mask=m)
    assert same.label["severity"]["value"] == pytest.approx(0, abs=1e-6)
    assert same.label["error_rms_ppb"] == pytest.approx(0, abs=1e-9)
    wrong = A.gross_wrong_b0(chi, VOX, (0, 0, 1), (1, 0, 0), mask=m)
    assert wrong.label["severity"]["value"] == pytest.approx(90)
    assert wrong.label["subscore"] == "GROSS"
    assert wrong.label["error_rms_ppb"] > 1.0


# ---------------------------------------------------------------- 2. mask

def test_mask_dilate_erode_distance():
    m = _sphere(7.0)
    d = A.mask_dilate(m, VOX, mm=2.0)
    assert (d.data >= m).all() and d.label["excess_ml"] > 0
    assert d.label["severity"]["value"] == pytest.approx(2.0, abs=1.0)
    e = A.mask_erode(m, VOX, mm=2.0)
    assert (e.data <= m).all()
    assert e.label["severity"]["value"] == pytest.approx(-2.0, abs=1.0)
    lim = A.mask_dilate(m, VOX, mm=4.0, within=_sphere(8.0))
    assert not (lim.data & ~_sphere(8.0)).any()


def test_mask_cut_holes_islands_misregister():
    m = _sphere(7.0)
    c = A.mask_plane_cut(m, VOX, depth_mm=3.0, normal=(0, 0, -1))
    z = np.indices(m.shape)[2]
    assert z[c.affected].max() < z[m].min() + 3 + 1e-9
    assert not (c.data & c.affected).any()
    h = A.mask_holes(m, VOX, n_holes=2, radius_mm=2.0, seed=1)
    assert h.label["severity"]["value"] == pytest.approx(h.affected.sum() / 1000)
    i = A.mask_islands(m, VOX, n_islands=2, radius_mm=1.5, gap_mm=(2.0, 4.0), seed=1)
    from scipy import ndimage
    assert ndimage.label(i.data)[1] > 1  # disconnected parts
    s = A.mask_misregister(m, VOX, shift_mm=(2.0, 0, 0))
    np.testing.assert_array_equal(s.data, np.roll(m, 2, axis=0))
    assert s.label["severity"]["value"] == pytest.approx(2.0)
    assert s.label["dice"] < 1
    f = A.mask_misregister(m, VOX, flip_axis=0)
    assert f.label["severity"]["value"] >= 0


def test_chi_under_mask():
    chi, m = _chi()
    big = A.mask_dilate(m, VOX, mm=2.0).data
    r = A.chi_under_mask(chi, m, big, VOX, seed=0)
    np.testing.assert_array_equal(r.data[m], chi[m])
    assert np.abs(r.data[r.affected]).max() > 0
    assert (r.data[~big] == 0).all()


# ---------------------------------------------------------------- 3. unwrapping

def test_unwrap_patch_phase_and_field():
    _, m, fhz = _signal()
    ph = 2 * np.pi * fhz[..., None] * TES
    r = A.unwrap_patch(ph, m, VOX, k=1, radius_mm=4, echoes=[1, 2], tes=TES, seed=2)
    d = r.data - ph
    np.testing.assert_allclose(d / (2 * np.pi), np.round(d / (2 * np.pi)), atol=1e-9)
    assert (d[..., 0] == 0).all() and (d[..., 3] == 0).all()
    np.testing.assert_allclose(d[..., 1], 2 * np.pi * r.affected)
    assert not (r.affected[~m] != 0).any()
    assert r.label["field_error_hz_per_cycle"] == pytest.approx(
        np.sum(TES[1:3]) / np.sum(TES ** 2))
    f = A.unwrap_patch(fhz, m, VOX, k=[2], radius_mm=4, domain="field", te=0.01, seed=2)
    np.testing.assert_allclose((f.data - fhz)[f.affected != 0], 200.0 * np.sign(f.affected[f.affected != 0]))
    p = A.unwrap_patch(fhz / (42.58 * 3), m, VOX, k=[1], radius_mm=4, domain="field", te=0.01, b0=3,
                       field_unit="ppm", seed=2)
    assert p.label["jump_per_cycle"] == pytest.approx(100 / (42.58 * 3))
    with pytest.raises(ValueError):
        A.unwrap_patch(fhz, m, VOX, domain="field")
    ring = A.unwrap_patch(ph, m, VOX, radius_mm=6, ring_width_mm=2, roughness=0.0, seed=0)
    c = np.array(ring.label["centres_vox"][0])
    rr = np.sqrt(((np.argwhere(ring.affected != 0) - c) ** 2).sum(1))
    assert rr.min() >= 4.9 and rr.max() <= 7.1


def test_unwrap_slicewise_and_echo():
    _, m, fhz = _signal()
    ph = 2 * np.pi * fhz[..., None] * TES
    s = A.unwrap_slicewise(ph, m, VOX, axis=2, fraction=0.25, seed=1)
    d = (s.data - ph)[..., 0]
    for z in range(N):
        vals = np.unique(np.round(d[..., z][m[..., z]] / (2 * np.pi)))
        assert len(vals) <= 1
    assert len(s.label["slices"]) == round(0.25 * m.any(axis=(0, 1)).sum())
    g = A.unwrap_echo_error(ph, m, VOX, echo=3, k=1, radius_mm=None, tes=TES)
    d = g.data - ph
    assert (d[..., :3] == 0).all()
    np.testing.assert_allclose(d[..., 3][m], 2 * np.pi)
    assert g.label["field_error_hz"] == pytest.approx(TES[3] / np.sum(TES ** 2))


# ---------------------------------------------------------------- 4. motion

def test_resample_complex_translation_keeps_wrapped_phase():
    sig, m, _ = _signal()
    T = A.rigid_matrix(translation_mm=(2.0, 0, 0))
    out = A.resample_complex(sig, T, VOX)
    np.testing.assert_allclose(out[4:-4, 4:-4, 4:-4], np.roll(sig, 2, axis=0)[4:-4, 4:-4, 4:-4], atol=1e-12)
    # a point moves where T sends it
    v = np.zeros((N, N, N))
    v[10, 12, 12] = 1
    o = A._resample_real(v, A.rigid_matrix((0, 0, 1), 90.0, centre_mm=(12, 12, 12)), VOX)
    assert np.unravel_index(np.argmax(o), o.shape) == (12, 10, 12)


def test_interecho_motion():
    sig, m, _ = _signal()
    r = A.interecho_motion(sig, VOX, m, rot_deg=2.0, trans_mm=1.0, seed=0)
    np.testing.assert_array_equal(r.data[..., 0], sig[..., 0])
    assert r.label["echoes"][0]["angle_deg"] == 0
    assert r.label["echoes"][3]["angle_deg"] == pytest.approx(6.0)
    d = [e["rms_displacement_mm"] for e in r.label["echoes"]]
    assert d[0] == 0 and d[1] < d[2] < d[3]
    assert r.affected.max() == pytest.approx(max(e["max_displacement_mm"] for e in r.label["echoes"]))
    z = A.interecho_motion(sig, VOX, m, rot_deg=0.0, trans_mm=0.0)
    np.testing.assert_array_equal(z.data, sig)
    rnd = A.interecho_motion(sig, VOX, m, mode="random", seed=1)
    assert rnd.label["echoes"][0]["rms_displacement_mm"] == 0


def test_slicewise_and_within_echo_motion():
    sig, m, _ = _signal()
    s = A.slicewise_motion(sig, VOX, m, axis=2, max_shift_mm=1.0, fraction=0.3, seed=0)
    moved = [e["slice"] for e in s.label["slices"]]
    still = [z for z in range(N) if z not in moved]
    np.testing.assert_array_equal(s.data[:, :, still], sig[:, :, still])
    w0 = A.within_echo_motion(sig, VOX, m, onset=1.01, seed=0)
    np.testing.assert_allclose(w0.data, sig, atol=1e-12)
    assert w0.label["severity"]["value"] == pytest.approx(0, abs=1e-20)
    w = A.within_echo_motion(sig, VOX, m, onset=0.6, rot_deg=3, trans_mm=1, seed=0)
    assert w.label["severity"]["value"] > 0
    assert 0.35 < w.label["moved_line_fraction"] < 0.45


# ---------------------------------------------------------------- 5. bipolar

def test_bipolar_readout():
    sig, m, _ = _signal()
    r = A.bipolar_readout(sig, VOX, m, readout_axis=0, gradient_rad_per_mm=0.05, offset_rad=0.3)
    np.testing.assert_array_equal(r.data[..., 0], sig[..., 0])
    np.testing.assert_array_equal(r.data[..., 2], sig[..., 2])
    dphi = np.angle(r.data[..., 1] * np.conj(sig[..., 1]))
    x = np.arange(N) - np.argwhere(m).mean(0)[0]
    want = np.angle(np.exp(1j * (0.05 * x + 0.3)))
    np.testing.assert_allclose(dphi[:, 12, 12], want, atol=1e-9)
    s = A.bipolar_readout(sig, VOX, m, gradient_rad_per_mm=0.0, shift_voxels=1.0)
    np.testing.assert_allclose(s.data[..., 1], np.roll(sig[..., 1], 1, axis=0), atol=1e-9)
    assert s.label["shift_mm"] == 1.0


# ---------------------------------------------------------------- 6. k-space

def test_kspace_effects():
    sig, m, _ = _signal()
    sp = A.kspace_spike(sig, VOX, amplitude=0.05, echoes=[2], seed=0)
    np.testing.assert_array_equal(sp.data[..., 0], sig[..., 0])
    diff = sp.data[..., 2] - sig[..., 2]
    assert np.allclose(np.abs(diff), np.abs(diff).mean())  # a pure stripe (complex exponential)
    assert sp.label["stripe_wavelength_mm"] > 0
    pf1 = A.partial_fourier(sig, axis=1, fraction=1.0)
    np.testing.assert_allclose(pf1.data, sig, atol=1e-12)
    pf = A.partial_fourier(sig, axis=1, fraction=0.75)
    K = np.fft.fft(pf.data[..., 0], axis=1)
    k = np.fft.fftfreq(N) * N
    assert np.abs(K[:, k < -(18 - N // 2)]).max() < 1e-9
    hd = A.partial_fourier(sig, axis=1, fraction=0.625, method="homodyne")
    assert np.isfinite(hd.data).all()
    # a slowly varying phase is removed by the high-pass, the magnitude kept
    x = np.indices((N, N, N))[0]
    smooth = np.where(m, 1.0, 0.0) * np.exp(1j * 0.1 * x)
    hp = A.phase_highpass(smooth, VOX, sigma_mm=3.0)
    inner = _sphere(5.0)
    assert np.abs(np.angle(hp.data[inner])).max() < 0.1 < np.abs(np.angle(smooth[inner])).max()
    np.testing.assert_allclose(np.abs(hp.data), np.abs(smooth))


# ---------------------------------------------------------------- BIDS injection and CLI

def _tiny_bids(root):
    sig, m, fhz = _signal()
    anat = os.path.join(root, "sub-1", "anat")
    deriv = os.path.join(root, "derivatives", "qsm-forward", "sub-1", "anat")
    os.makedirs(anat)
    os.makedirs(deriv)
    aff = np.eye(4)
    for e, te in enumerate(TES):
        for part, arr in (("mag", np.abs(sig[..., e])), ("phase", np.angle(sig[..., e]))):
            stem = os.path.join(anat, f"sub-1_echo-{e + 1}_part-{part}_MEGRE")
            nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), stem + ".nii")
            json.dump({"EchoTime": float(te), "MagneticFieldStrength": 3, "B0_dir": [0, 0, 1]}, open(stem + ".json", "w"))
    nib.save(nib.Nifti1Image(m.astype(np.uint8), aff), os.path.join(deriv, "sub-1_mask.nii"))
    nib.save(nib.Nifti1Image((fhz / (42.58 * 3)).astype(np.float32), aff), os.path.join(deriv, "sub-1_fieldmap.nii"))
    nib.save(nib.Nifti1Image(np.zeros(m.shape, np.float32), aff), os.path.join(deriv, "sub-1_Chimap.nii"))
    return sig, m


def test_inject_bids_and_cli(tmp_path):
    from qsm_forward.main import main
    import sys
    src = str(tmp_path / "src")
    sig, m = _tiny_bids(src)
    out = str(tmp_path / "out")
    effects = [("bipolar_readout", {"gradient_rad_per_mm": 0.05}), ("mask_erode", {"mm": 2.0}),
               ("unwrap_patch", {"domain": "field", "te": 0.01, "radius_mm": 4}),
               ("sidecar", {"te_factor": 1e-3})]
    p = A.inject_bids(src, out, effects, seed=1)
    d = json.load(open(p))
    assert [e["effect"] for e in d["Effects"]] == ["bipolar_readout", "mask_erode", "unwrap_patch", "sidecar"]
    anat = os.path.join(out, "sub-1", "anat")
    ph2 = nib.load(os.path.join(anat, "sub-1_echo-2_part-phase_MEGRE.nii")).get_fdata()
    assert not np.allclose(ph2, np.angle(sig[..., 1]), atol=1e-4)
    ph1 = nib.load(os.path.join(anat, "sub-1_echo-1_part-phase_MEGRE.nii")).get_fdata()
    np.testing.assert_allclose(ph1, np.angle(sig[..., 0]), atol=1e-5)
    assert json.load(open(os.path.join(anat, "sub-1_echo-1_part-mag_MEGRE.json")))["EchoTime"] == pytest.approx(4e-6)
    deriv = os.path.join(out, "derivatives", "qsm-forward", "sub-1", "anat")
    fm = nib.load(os.path.join(deriv, "sub-1_desc-fault_mask.nii")).get_fdata() > 0
    assert fm.sum() < m.sum()
    assert os.path.exists(os.path.join(deriv, "sub-1_desc-fault_fieldmap.nii"))
    with pytest.raises(FileExistsError):
        A.inject_bids(src, out, effects)
    # CLI
    out2 = str(tmp_path / "out2")
    argv = sys.argv
    sys.argv = ["qsm-forward", "artefacts", src, out2, "--effect", "interecho_motion:rot_deg=1.5,trans_mm=0.5",
                "--effect", "mask_misregister:shift_mm=[2,0,0]"]
    try:
        main()
    finally:
        sys.argv = argv
    d2 = json.load(open(os.path.join(out2, "artefacts.json")))
    assert d2["Effects"][0]["params"]["rot_deg"] == 1.5
    assert d2["Effects"][1]["params"]["shift_mm"] == [2, 0, 0]
    with pytest.raises(SystemExit):
        A._parse_effect("nonsense")


def test_swi_homodyne():
    sig, m, _ = _signal()
    x = np.indices((N, N, N))[0]
    smooth = np.where(m, 1.0, 0.0)[..., None] * np.exp(1j * 0.1 * x)[..., None] * np.ones(4)
    r = A.swi_homodyne(smooth, VOX, window=8)
    inner = _sphere(5.0)
    assert np.abs(np.angle(r.data[..., 0][inner])).max() < 0.1
    np.testing.assert_allclose(np.abs(r.data), np.abs(smooth))
    assert r.label["severity"]["value"] == pytest.approx(4 / N)
    # slice-wise (2D): a phase constant within each slice is removed whatever it does across slices
    z = np.indices((N, N, N))[2]
    sl = np.exp(1j * 0.3 * z)
    np.testing.assert_allclose(A.swi_homodyne(sl, VOX, window=8).data, np.ones(sl.shape), atol=1e-9)
