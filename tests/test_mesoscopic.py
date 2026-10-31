"""Wiring tests of the LUT-driven chi-sep signal models (qsm_forward.mesoscopic) on small
synthetic look-up tables whose signals are known in closed form. The physics of the real
tables is checked where they are built, not here."""
import json
import os
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import qsm_forward
from qsm_forward import chisep, mesoscopic as meso

# synthetic mesoscopic table: rate (1/ms) = (K_POS·χ+ + K_NEG·|χ-|·sin²θ)·B0/3, SE rate K_SE·χ+
K_POS, K_NEG, K_SE, PHASE = 0.140, 0.100, 0.010, 0.5


def _meso_lut(path):
    cp = np.array([0.0, 0.1, 0.2])
    cn = np.array([0.0, -0.04, -0.08])
    th = np.array([0.0, 30.0, 60.0, 90.0])
    b0 = np.array([3.0, 7.0])
    te = np.arange(0, 80.0001, 0.5)
    tse = np.arange(0, 80.0001, 1.0)
    P, N, T, B = np.meshgrid(cp, np.abs(cn), np.deg2rad(th), b0, indexing="ij")
    P, N, T, B = (np.moveaxis(a, 0, 1) for a in (P, N, T, B))          # (neg, pos, th, B0)
    rate = (K_POS * P + K_NEG * N * np.sin(T) ** 2) * B / 3
    gre = np.exp((-rate + 2j * np.pi * PHASE * N * np.cos(T) ** 2)[..., None, None] * te)
    se = np.exp((-K_SE * P)[..., None, None] * tse)
    gre = np.broadcast_to(gre, gre.shape[:4] + (3, len(te))).astype(np.complex64)
    se = np.broadcast_to(se, se.shape[:4] + (3, len(tse))).astype(np.complex64)
    frac = np.array([[1, 0, 0], [0.8, 0.1, 0.1], [0.6, 0.2, 0.2]], float)
    np.savez(path, gre=gre, se=se, frac=frac, chi_pos=cp, chi_neg=cn, theta=th, B0=b0, TE=te,
             TE_SE=tse, mode="diffusion", params="{}")
    return path


def _sphere_lut(path, extra=0.002):
    """S = exp(i·c·(ζ+χs+ - ζ-χs- + extra)·TE) with c = (2/3)γB0 (decompose) and a pure decay
    for lorentz."""
    z = np.round(np.arange(0.0, 0.6001, 0.05), 2)
    s = np.array([0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75])
    te = np.arange(0, 50.0001, 0.5)
    c = (2 / 3) * meso.GAMMA * 3.0 * 1e-6
    ZP, ZM, SP, SM = np.meshgrid(z, z, s, s, indexing="ij")
    bulk = ZP * SP - ZM * SM
    S_dec = np.exp(1j * c * (bulk + extra)[..., None] * te)
    S_lor = np.exp(-(ZP * SP + ZM * SM)[..., None] * te * 0.3)
    np.savez(path, zeta=z, chi_s=s, te=te, b0=3.0, S_decompose=S_dec.astype(np.complex64),
             S_lorentz=S_lor.astype(np.complex64))
    return path


class FakeTools:
    """STAR stand-in: χ = phase / (2πγ̄B0·TE) (the field); unwrapping is the identity."""
    calls = 0

    def unwrap(self, phases, mask, voxel):
        return [np.asarray(p, float) for p in phases]

    def star(self, phase, mask, te_ms, b0, voxel):
        FakeTools.calls += 1
        return np.where(mask, phase / (2 * np.pi * 42.58 * b0 * te_ms * 1e-3), 0.0)


@pytest.fixture
def luts(tmp_path, monkeypatch):
    m = _meso_lut(str(tmp_path / meso.LUT_FILES["diffusion"]))
    s = _sphere_lut(str(tmp_path / meso.LUT_FILES["spheres"]))
    monkeypatch.setenv(meso.LUT_ENV, str(tmp_path))
    return m, s


class TestMesoLUT:
    def test_lut_path(self, luts, monkeypatch):
        assert meso.lut_path("diffusion") == luts[0]
        monkeypatch.setenv(meso.LUT_ENV, "/nonexistent")
        with pytest.raises(FileNotFoundError):
            meso.lut_path("diffusion")
        assert meso.lut_path("spheres", luts[1]) == luts[1]

    def test_lut_download_cache_and_checksum(self, luts, monkeypatch, tmp_path):
        """No env and no path: fetched from the pinned URL into the cache, checked, then reused."""
        monkeypatch.delenv(meso.LUT_ENV)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        name = meso.LUT_FILES["diffusion"]
        monkeypatch.setitem(meso.LUT_SHA256, name, meso._sha256(luts[0]))
        urls = []

        def fake_fetch(url, dest):
            urls.append(url)
            shutil.copyfile(luts[0], dest)

        monkeypatch.setattr(meso, "_fetch", fake_fetch)
        p = meso.lut_path("diffusion")
        assert p.startswith(str(tmp_path / "cache")) and meso._sha256(p) == meso.LUT_SHA256[name]
        assert urls == [f"https://huggingface.co/datasets/{meso.LUT_REPO}/resolve/"
                        f"{meso.LUT_REVISION}/{name}"]
        assert meso.lut_path("diffusion") == p and len(urls) == 1      # cached: no second fetch

        monkeypatch.setitem(meso.LUT_SHA256, name, "0" * 64)            # a corrupt download
        with pytest.raises(OSError, match="SHA-256"):
            meso.lut_path("diffusion")
        assert not any(f.endswith(".part") for f in os.listdir(os.path.dirname(p)))

    def test_nodes_extrapolation_and_compartments(self, luts):
        L = meso.MesoLUT(luts[0])
        kw = dict(T2_ms=(1e12,) * 3, rho=(1, 0, 0))
        cp, cn, th = np.array([0.1, 0.3]), np.array([-0.04, 0.0]), np.deg2rad([60.0, 0.0])
        F = L.decay(cp, cn, th, 3.0, 20.0, **kw)
        rate = K_POS * cp + K_NEG * np.abs(cn) * np.sin(th) ** 2
        expect = np.exp(-rate * 20 + 2j * np.pi * PHASE * np.abs(cn) * np.cos(th) ** 2 * 20)
        np.testing.assert_allclose(F, expect, rtol=1e-5)          # 0.3 ppm: linear extrapolation
        w = L.weights(np.array([0.0, -0.04, -0.08]))
        np.testing.assert_allclose(w.sum(0), 1.0)
        np.testing.assert_allclose(w[:, 1], np.array([0.8, 0.05, 0.1]) / 0.95)
        # myelin water: its own T2 and exchange shift
        F2 = L.decay(np.array([0.0]), np.array([0.0]), np.array([0.0]), 3.0, 10.0, T2_ms=(1e12, 10.0, 1e12))
        np.testing.assert_allclose(F2, 1.0, rtol=1e-6)            # no myelin at χ- = 0

    def test_powder_average_and_iron_relaxivity(self, luts):
        L = meso.MesoLUT(luts[0])
        assert L.powder_weights().sum() == pytest.approx(1.0)
        z = np.zeros(1)
        kw = dict(T2_ms=(1e12,) * 3, rho=(1, 0, 0))
        # no χ-: no θ dependence, so the orientation average equals any fixed angle
        np.testing.assert_allclose(L.decay(z + 0.1, z, z, 3.0, 15.0, powder=True, **kw),
                                   L.decay(z + 0.1, z, z, 3.0, 15.0, **kw), rtol=1e-6)
        r = L.iron_relaxivity(3.0)
        np.testing.assert_allclose(r["Dr_plus"], 1e3 * (K_POS - K_SE), rtol=1e-4)
        np.testing.assert_allclose(r["R2_per_ppm"], 1e3 * K_SE, rtol=1e-4)
        with pytest.raises(ValueError):
            L.decay(z, z, z, 1.5, 10.0, **kw)

    def test_chi_neg_conventions(self):
        cn = np.array([-0.04, -0.04, -0.04, 0.01])
        th = np.array([0.0, np.arccos(1 / np.sqrt(3)), np.nan, 0.0])
        app = meso.chi_neg_apparent(cn, th)
        np.testing.assert_allclose(app, [-0.02, -0.04, -0.04, 0.0])
        np.testing.assert_allclose(meso.chi_neg_ground_truth(cn), [-0.04, -0.04, -0.04, 0.0])

    def test_signal_reproduces_tissue_r2_in_spin_echo(self, luts):
        L = meso.MesoLUT(luts[0])
        shape = (2, 2, 2)
        seg = np.full(shape, 9)
        seg[0] = 8
        seg[1, 1, 1] = 10                                          # CSF: not in the LUT mask
        cp = np.full(shape, 0.1)
        cn = np.zeros(shape)
        th = np.full(shape, 45.0)
        sig = meso.MesoscopicSignal(L, cp, cn, th, seg, 3.0, se_tes_ms=(10, 30, 50, 70))
        R2 = np.full(shape, 20.0)
        se = np.array([sig.se(t * 1e-3, R2)[1] for t in (10, 30, 50, 70)])
        np.testing.assert_allclose(meso.fit_rate([10, 30, 50, 70], se), 20.0, rtol=1e-6)
        mask, g = sig.gre(0.02, R2)
        assert mask.sum() == 7 and g.shape == (7,)
        # GRE rate = intrinsic (20 - 1) + iron 14 at χ+ = 0.1
        np.testing.assert_allclose(np.abs(g), np.exp(-0.02 * (19.0 + 14.0)), rtol=1e-5)
        assert sig.floored_fraction == 0.0

    def test_intrinsic_r2_floor(self, luts):
        L = meso.MesoLUT(luts[0])
        shape = (2, 2, 2)
        seg = np.full(shape, 9)
        seg[0] = 2
        cp = np.full(shape, 0.1)                                   # LUT SE loss 1 1/s
        sig = meso.MesoscopicSignal(L, cp, np.zeros(shape), np.full(shape, np.nan), seg, 3.0,
                                    se_tes_ms=(10, 30, 50, 70))
        assert sig.r2_floor == pytest.approx(meso.free_water_r2(3.0)) == pytest.approx(0.65 / 1.0296)
        assert meso.free_water_r2(7.0) == pytest.approx(1 / 1.0296)
        R2 = np.where(seg == 2, 1.2, 20.0)                          # pallidus: 1.2 - 1 < floor
        sig.se(0.01, R2)
        rep = sig.floor_report(seg)
        assert rep["ByLabel"]["2"]["floored"] == 4 and rep["ByLabel"]["9"]["floored"] == 0
        np.testing.assert_allclose(rep["ByLabel"]["2"]["median_se_r2_excess_floored"],
                                   sig.r2_floor + 1.0 - 1.2, atol=1e-6)
        assert rep["UnflooredResidualAbove0p1"] == 0 and rep["UnflooredResidualP99"] < 1e-6
        assert rep["FlooredFraction"] == 0.5


class TestSpheres:
    def test_source_params(self):
        b = np.array([0.001, 0.02, 0.0375, 0.045, 0.1, -0.03])
        z, s = meso.source_params(b)
        np.testing.assert_allclose(z * s, np.abs(b))
        assert np.all(z <= 0.35) and np.all(s >= 0.01)
        np.testing.assert_allclose(s[[1, 3]], [0.08, 0.15])

    def test_source_params_inrange(self):
        cap = 0.127
        cp = np.array([0.0, 0.001, 0.02, 0.06, 0.07, 0.10, 0.20, 0.0])
        cn = np.array([0.0, -0.001, -0.01, -0.01, -0.03, -0.04, -0.02, -0.09])
        zp, zm, sp, sm, capped = meso.source_params_inrange(cp, cn, cap)
        np.testing.assert_allclose(zp * sp * (zp > 0), cp, atol=1e-12)
        np.testing.assert_allclose(zm * sm * (zm > 0), -cn, atol=1e-12)
        need = (cp - cn) / cap
        assert np.all(zp + zm <= 0.6 + 1e-12)
        np.testing.assert_array_equal(capped, need > 0.6)
        ok = ~capped
        assert np.all(sp[ok] <= cap + 1e-12) and np.all(sm[ok] <= cap + 1e-12)
        assert np.all(sp[capped & (cp > 0)] > cap) and np.all(sm[capped & (cn < 0)] > cap)
        np.testing.assert_allclose([zp[2], zm[2]], [0.25, 0.25])          # low bulk: ζ_ref, as highchi
        assert sp[3] == pytest.approx(cap) and 0.25 <= zp[3] + zm[3] <= 0.6  # χ+ at the cap, χ- gives way
        m = meso.SourceMapping("in-range", cap)
        flag = m(cp, cn, chi_s_lut_max=0.5)[-1]
        np.testing.assert_array_equal(flag, np.where(capped, 1, 0))
        assert m(np.array([0.5]), np.array([0.0]), chi_s_lut_max=0.75)[-1][0] == 2   # ζ 0.6, χ_s 0.83 > LUT
        with pytest.raises(ValueError):
            meso.SourceMapping("in-range")

    def test_lut_coverage(self, tmp_path):
        z = np.array([0.0, 0.1, 0.2, 0.3, 0.35])
        s, te = np.array([0.01, 0.15]), np.array([0.0, 10.0])
        S = np.ones((5, 5, 2, 2, 2), np.complex64)
        p = str(tmp_path / "v1.npz")
        np.savez(p, zeta=z, chi_s=s, te=te, b0=3.0, S_decompose=S, S_lorentz=S)
        L = meso.SphereLUT(p)
        L.check_mapping(meso.HIGHCHI)
        with pytest.raises(ValueError, match="v2"):
            L.check_mapping(meso.SourceMapping("in-range", 0.127))

    def test_voxelwise_chi_echoes_and_wrap_limit(self, luts):
        L = meso.SphereLUT(luts[1])
        cp, cn = np.array([0.02, 0.05]), np.array([-0.01, 0.0])
        te = np.array([2.7, 20.1, 46.2])
        for interp, atol in (("log", 1e-6), ("complex", 2e-3)):   # log-signal is exact here
            ce = meso.voxelwise_chi_echoes(L, cp, cn, te, "decompose", interp=interp)
            np.testing.assert_allclose(ce, np.broadcast_to(2 * (cp + cn)[:, None] + 0.002, ce.shape), atol=atol)
        assert meso.wrap_limit_ppm(46.2) == pytest.approx(0.127, abs=5e-4)

    def test_star_chimap_echoes(self):
        shape = (4, 4, 4)
        mask = np.zeros(shape, bool)
        mask[1:3, 1:3, 1:3] = True
        rng = np.random.default_rng(1)
        field, gt = rng.normal(0, 0.05, shape), rng.normal(0, 0.05, shape)
        te = np.array([5.0, 10.0, 20.0])
        # linear phase: every echo reconstructs the same χ, chimap-echoes = ground truth
        ref = meso.echo_local_phases(field, te, 3.0)
        out = meso.star_chimap_echoes(gt, ref, ref, mask, te, 3.0, (1, 1, 1), FakeTools(), unwrap="exact")
        np.testing.assert_allclose(out[:, mask], np.broadcast_to(gt[mask], (3, mask.sum())), atol=1e-12)
        assert np.all(out[:, ~mask] == 0)

        class BiasedTools(FakeTools):                               # STAR's own echo dependence
            def star(self, phase, mask, te_ms, b0, voxel):
                return super().star(phase, mask, te_ms, b0, voxel) * (1 + 1.0 / te_ms)

        # an echo-dependent intra-voxel phase shows up as the per-echo minus reference difference,
        # and a reconstruction bias common to both cancels to first order
        pattern = rng.uniform(0, 1, shape)
        intra = np.stack([0.1 * t ** 2 / 100 * pattern for t in te])
        ph = meso.echo_local_phases(field, te, 3.0, intra)
        c = 2 * np.pi * 42.58 * 3.0 * 1e-3
        p = pattern[mask] - pattern[mask].mean()                    # the mask mean is removed
        per = 0.1 * te[:, None] / 100 * p / c
        out, se, sr = meso.star_chimap_echoes(gt, ph, ref, mask, te, 3.0, (1, 1, 1), FakeTools(),
                                              unwrap="exact", return_parts=True)
        np.testing.assert_allclose(out[:, mask] - gt[mask], per, atol=1e-12)
        out = meso.star_chimap_echoes(gt, ph, ref, mask, te, 3.0, (1, 1, 1), BiasedTools(), unwrap="exact")
        np.testing.assert_allclose(out[:, mask] - gt[mask], per * (1 + 1.0 / te[:, None]), atol=1e-12)


def _tiny(tmp):
    from test_qsm_forward import _tiny_chisep_tissue
    paths, arrays = _tiny_chisep_tissue(tmp, n=10)
    tp = qsm_forward.TissueParams(root_dir=tmp, **paths)
    theta = np.full(arrays["seg"].shape, 45.0)
    theta[0, 0, 0] = np.nan
    return tp, arrays, theta


def _maps(tp, arrays, theta, cfg):
    dr_pos, dr_neg = chisep.relaxivity_maps(arrays["seg"], theta, cfg)
    return chisep.ChisepMaps(tp, arrays["chi_pos"], arrays["chi_neg"], theta, dr_pos, dr_neg,
                             qsm_forward.generate_r2prime(arrays["chi_pos"], arrays["chi_neg"], dr=dr_pos, dr_neg=dr_neg))


class TestChisepIntegration:
    def test_presets_and_validation(self):
        for name in ("meso-3t", "wm-micro-7t", "spheres-3t-paper", "spheres-3t-lorentz",
                     "spheres-3t-paper-highchi", "spheres-3t-lorentz-highchi"):
            cfg = chisep.PRESETS[name]
            assert cfg.r2prime_source == "fitted" and cfg.voxel == 1.0 and cfg.se and not cfg.anisotropic
        assert chisep.PRESETS["meso-3t"].signal_model == "mesoscopic"
        assert chisep.PRESETS["meso-3t"].tes_ms[-1] == 46.2 and len(chisep.PRESETS["meso-3t"].tes_ms) == 16
        assert chisep.PRESETS["wm-micro-7t"].b0 == 7.0 and len(chisep.PRESETS["wm-micro-7t"].tes_ms) == 8
        assert chisep.PRESETS["spheres-3t-paper"].sphere_convention == "decompose"
        assert chisep.PRESETS["spheres-3t-lorentz"].sphere_convention == "lorentz"
        assert chisep.PRESETS["spheres-3t-paper"].tr == pytest.approx(0.0593)
        assert chisep.PRESETS["spheres-3t-paper"].sphere_source_range == "in-range"
        assert chisep.PRESETS["spheres-3t-paper-highchi"].sphere_source_range == "highchi"
        assert chisep.PRESETS["spheres-3t-lorentz-highchi"].sphere_convention == "lorentz"
        m = chisep.sphere_mapping(chisep.PRESETS["spheres-3t-paper"])
        assert m.kind == "in-range" and m.chi_s_max == pytest.approx(meso.wrap_limit_ppm(46.2))
        assert chisep.sphere_mapping(chisep.PRESETS["spheres-3t-lorentz-highchi"]) is meso.HIGHCHI
        for bad in (dict(signal_model="mesoscopic"),                                   # relaxation ridani
                    dict(signal_model="spheres", relaxation="linear", dr=300, b0=7.0),
                    dict(relaxation="linear", dr=140, chimap_echoes=True),
                    dict(signal_model="mesoscopic", relaxation="linear", dr=140, meso_table="nope"),
                    dict(signal_model="spheres", relaxation="linear", dr=300, b0=3.0, sphere_convention="nope"),
                    dict(signal_model="spheres", relaxation="linear", dr=300, b0=3.0, sphere_source_range="nope"),
                    dict(signal_model="spheres", relaxation="linear", dr=300, b0=3.0, sphere_zeta_total_max=1.2)):
            with pytest.raises(ValueError):
                chisep.ChisepConfig(**bad)

    def test_mesoscopic_end_to_end(self, luts):
        with tempfile.TemporaryDirectory() as tmp:
            tp, arrays, theta = _tiny(tmp)
            cfg = chisep.preset("meso-3t", voxel=0.0, noiseless=True, tes_ms=(5, 10, 20, 30))
            written = chisep.generate_chisep_bids(tmp, os.path.join(tmp, "bids"), cfg,
                                                  maps=_maps(tp, arrays, theta, cfg))
            anat = os.path.dirname(written["metadata"])
            ld = lambda n: nib.load(os.path.join(anat, f"sub-1_{n}.nii")).get_fdata()
            wm = arrays["seg"] == 8
            app = meso.chi_neg_apparent(arrays["chi_neg"], np.deg2rad(np.where(wm, theta, np.nan)))
            np.testing.assert_allclose(ld("Chimap-neg"), np.abs(arrays["chi_neg"]), atol=1e-6)
            np.testing.assert_allclose(ld("desc-apparent_Chimap-neg"), np.abs(app), atol=1e-6)
            np.testing.assert_allclose(ld("Chimap"), arrays["chi_pos"] + app, atol=1e-6)
            r2p = ld("R2prime")
            assert np.all(np.isfinite(r2p)) and r2p.mean() > 0
            meta = json.load(open(written["metadata"]))
            assert meta["SignalModel"]["Model"] == "mesoscopic" and meta["SignalModel"]["Table"] == "diffusion"
            assert "fibreangle" in written

    def test_spheres_end_to_end(self, luts, monkeypatch):
        monkeypatch.setattr(meso, "QsmrsTools", FakeTools)
        with tempfile.TemporaryDirectory() as tmp:
            tp, arrays, theta = _tiny(tmp)
            cfg = chisep.preset("spheres-3t-paper", voxel=0.0, noiseless=True, tes_ms=(2.7, 5.6, 8.5))
            FakeTools.calls = 0
            written = chisep.generate_chisep_bids(tmp, os.path.join(tmp, "bids"), cfg,
                                                  maps=_maps(tp, arrays, theta, cfg))
            assert FakeTools.calls == 6                                   # 3 echoes + 3 references
            ce = nib.load(written["chimap_echoes"]).get_fdata()
            gt = nib.load(written["chimap_echoes"].replace("_Chimap-echoes", "_Chimap")).get_fdata()
            assert ce.shape == gt.shape + (3,)
            # the synthetic table's intra-voxel phase is linear in TE, c·(bulk + 0.002)·TE with
            # c = (2/3)γB0: per-echo STAR minus the linear-phase reference gives the TE-independent
            # χ offset (2/3)(bulk + 0.002), mask mean removed (up to the complex interpolation)
            mask = nib.load(written["chimap_echoes"].replace("_Chimap-echoes", "_mask")).get_fdata() > 0.5
            brain = np.isin(arrays["seg"], meso.BRAIN_LABELS)
            off = np.where(brain, (2 / 3) * (np.maximum(arrays["chi_pos"], 0) + np.minimum(arrays["chi_neg"], 0)
                                             + 0.002), 0.0)
            off = np.where(mask, off - off[mask].mean(), 0.0)
            np.testing.assert_allclose(ce, (gt + off)[..., None] * np.ones(3), atol=2e-4)
            meta = json.load(open(written["metadata"]))
            assert meta["SignalModel"]["Convention"] == "decompose"
            src = meta["SignalModel"]["Sources"]
            assert src["SourceRange"] == "in-range" and src["ChiSMaxPpm"] == pytest.approx(meso.wrap_limit_ppm(8.5))
            flag = nib.load(written["sourcerange_mask"]).get_fdata()
            assert flag.shape == gt.shape and set(np.unique(flag)) <= {0, 1, 2}
            assert src["Brain"]["n"] == int(brain.sum())
            # in range wherever the fraction cap does not bind: no source phase over π at TE_max
            assert src["Brain"]["either_over_phase_limit"] <= src["Brain"]["flag_above_cap"] + src["Brain"]["flag_clipped"]

    def test_cli_preset(self):
        from unittest.mock import patch
        from qsm_forward.main import main
        with patch.object(chisep, "generate_chisep_bids", return_value={}) as gen, \
                patch.object(sys, "argv", ["qsm-forward", "chi-sep", "data", "bids", "--preset", "spheres-3t-lorentz"]):
            main()
        assert gen.call_args.args[2].sphere_convention == "lorentz"
