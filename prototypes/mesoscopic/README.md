# Mesoscopic χ-separation signal model (prototype)

Simulates the gradient-echo (and spin-echo) signal of a voxel from explicitly placed
sub-voxel magnetic sources. No relaxivity (Dr), pool frequency or R2' formula goes into
the signal. R2', the Gaussian-to-exponential short-TE behaviour, orientation dependence
and non-linear phase come out of the field the sources produce.

## What it models

| Source | Geometry | Susceptibility | Signal contribution |
|---|---|---|---|
| Myelinated axons (χ−) | straight parallel hollow cylinders along z, polydisperse outer radius (Gamma, mean 0.6 µm), g-ratio 0.7, random sequential packing, FVF ≤ 0.6 | sheath tensor χ_I·I + χ_A·diag(1, −½, −½) in (r, φ, z), Wharton & Bowtell 2012: χ_I = χ_A = −0.10 ppm, scaled per χ− node beyond FVF 0.6 so the voxel χ− equals the input map | three water compartments (extra-axonal, myelin at water density 0.5, axonal) |
| Iron clusters (χ+, 83 %) | spheres R = 3 µm at random positions, overlap allowed (120 µm box, 0.47 µm grid) | Δχ = 20 ppm | static-to-intermediate regime: most of R2′ |
| Ferritin-scale iron (χ+, 17 %) | spheres R = 0.5 µm (12 µm box, 0.094 µm grid) | Δχ = 3 ppm | motional narrowing (α = δω R²/D ≈ 0.2): raises R2, R2′ ≈ 0 |

The cluster share is tuned so that, with diffusion, Dr+ = R2′/χ+ from mono-exponential
fits over `meso.PROTOCOL_3T` (GRE 5–40 ms minus Hahn SE 10–40 ms) is ≈ 140 Hz/ppm at
3 T (Shin et al. 2021: 137·B0/2.9). The 0.5 µm spheres stand in for ferritin: in the
narrowing regime only ζ·δω²·R²/D matters, and nanometre cores cannot be gridded.

- Field: the susceptibility-tensor dipole kernel with Lorentz correction and D(0) = 0
  on a periodic box. Frequencies are offsets from the macroscopic field that
  qsm-forward already computes from the voxel-mean χ. B0 is at angle θ to the fibres.
- Ground truth per voxel: χ+ = Σ Δχ·(sphere volume fraction) over both iron
  populations, χ− = χ_I·(myelin volume fraction)·(sheath scale), i.e. volume averages
  of the mean (tr/3) source susceptibility; orientation independent (scored).
  `lut.chi_neg_apparent` gives the secondary orientation-dependent apparent χ−
  (voxel-mean tensor projected on B0: χ−·[1 − ¾(χ_A/χ_I)(cos²θ − ⅓)]; equal to the
  ground truth in grey matter).
- Grey matter: orientation average (`powder=True`): solid-angle-weighted linear
  average of the complex signal over θ.
- Two tables: diffusion Monte Carlo (default) and static dephasing.
- Signal: the mean of exp(iφ) over water "walkers" in each compartment.
  - static: φ = ω(r₀)·t (static dephasing, D = 0).
  - diffusion: Monte Carlo random walk (D = 1 µm²/ms, dt = 2 µs). Myelin membranes are
    impermeable (no exchange), spheres are permeable. Spin echo: φ(TE) − 2φ(TE/2).
- Linearity: for a fixed geometry the field for any θ, B0 and source amplitude is a
  linear combination of 2 myelin and 3·n_levels sphere basis maps. Each walker records
  the time integrals of the basis maps, so a single walk covers every θ, B0 and χ
  amplitude.

Not modelled (by decision or not yet): water exchange between compartments, fibre
dispersion or crossing fibres within a WM voxel, iron placed relative to the axons
(spheres and axons are superposed), vessels and deoxyhaemoglobin, calcifications, and
diffusion anisotropy.

## DECOMPOSE-targeted sphere phantom (A3, `spheres_a3.py`)

After Chen et al. 2021 §3.2: 128³ sub-voxels, random para- and diamagnetic spheres
(radius 6–16 sub-voxels, non-overlapping between signs) that contain water at the
same proton density as the medium, static dephasing, 3 T, signal = mean of
exp(iγB0 f t) over all sub-voxels. DECOMPOSE's multi-pool signal equation is not used.
LUT axes: ζ± ∈ {0…0.35}, χ_s± ∈ {0.01…0.75} ppm, TE 0–50 ms (0.5 ms), 6 realisations.
`source_params` maps a bulk χ to (ζ, χ_s) with ζ·χ_s = bulk (χ_s 0.01–0.15 ppm at
ζ 0.25–0.35 for |bulk| ≤ 0.0525 ppm; above that χ_s rises beyond the paper's range).

Two intra-source conventions, both with zero voxel-mean frequency:
`decompose` (kernel 1 − kz²/k², intra-source water at (2/3)χ_s γB0, the paper's App. A3)
and `lorentz` (1/3 − kz²/k²). `SphereLUT.chi_echoes` returns the per-echo apparent
χ(TE) = χ_bulk + φ_res(TE)/((2/3)γB0·TE) (the `chimap-echoes` input);
`head` writes it for the Ridani anatomy on the Chen in vivo protocol (16 echoes
2.7–46.2 ms) and reports non-linear phase and the 3 T phase-wrap check.

## Files

| File | Purpose |
|---|---|
| `meso.py` | geometry, rasterisation, field basis maps, walkers, Y&H analytic references |
| `validate.py` | static-dephasing checks against analytic limits → `validation.json` |
| `build_lut.py` | one geometry × seed × mode → LUT part (`--list` shows geometries) |
| `lut.py` | `merge` parts into a LUT; `MesoLUT.decay(...)` per-voxel complex factor |
| `calibrate_iron.py` | Dr+ (R2′/χ+) and R2/χ+ for an iron population mix, static and diffusion |
| `demo_slice.py` | head slice: per-tissue emergent R2′, LUT-level Dr+, χ− ground truth vs apparent |
| `spheres_a3.py` | DECOMPOSE-targeted water-containing sphere phantom (A3): build/merge/validate/stats/head |
| `lut_array.slurm`, `calib_array.slurm`, `a3_array.slurm` | Bunya job arrays |

## Usage

```bash
PY=/scratch/user/uqaste15/envs/meso/bin/python   # numpy, scipy, nibabel, matplotlib
$PY validate.py --out OUT/validation
sbatch --export=ALL,MESO_DIR=$PWD,MESO_OUT=OUT/lut_parts lut_array.slurm
$PY lut.py merge --parts OUT/lut_parts --mode diffusion --out OUT/lut_diffusion.npz
sbatch --export=ALL,MESO_DIR=$PWD,MESO_OUT=OUT/a3_parts a3_array.slurm
$PY spheres_a3.py merge --parts OUT/a3_parts --out OUT/a3_lut.npz
$PY spheres_a3.py head --lut OUT/a3_lut.npz --out OUT/a3_head
```

```python
from lut import MesoLUT
L = MesoLUT("lut_diffusion.npz")
F = L.decay(chi_pos, chi_neg, theta, B0=7.0, TE_ms=20.0)            # WM (fixed fibre angle)
F = L.decay(chi_pos, chi_neg, theta, B0=7.0, TE_ms=20.0, powder=True) # dispersed fibres
```

`F` replaces the real `exp(-TE·R2*)` envelope in the GRE signal. The macroscopic field
phase and the SPGR steady-state factor are applied on top. Compartment T2s, water
densities (ρ_myelin = 0.5 by default) and the myelin-water exchange shift E are set at
evaluation time, not stored in the LUT.

## LUT layout

`gre[χ−, χ+, θ, B0, compartment, TE]` (complex64), `se[..., TE_SE]`, plus `frac[χ−,
compartment]`, the standard deviation over realisations, and the achieved FVF/MVF.
Axes: χ+ ∈ {0, .02, .05, .1, .15, .2, .25, .3} ppm, χ− ∈ {0, −.01, …, −.06, −.08}
ppm, θ = 0–90° in 10° steps, B0 ∈ {3, 7} T, TE = 0–80 ms in 0.5 ms steps (SE in
1 ms steps). Interpolation is multilinear in log-signal (log-magnitude plus phase
unwrapped along TE).

## References

- Yablonskiy & Haacke 1994, MRM 32:749, doi:10.1002/mrm.1910320610
- Wharton & Bowtell 2012, PNAS 109:18559, doi:10.1073/pnas.1211075109
- Liu 2010 (susceptibility tensor imaging), MRM 63:1471, doi:10.1002/mrm.22482
- Chen et al. 2021 (DECOMPOSE-QSM), NeuroImage 242:118477, doi:10.1016/j.neuroimage.2021.118477
- Shin et al. 2021 (χ-separation), NeuroImage 240:118371, doi:10.1016/j.neuroimage.2021.118371
- Brooks, Moiny & Gillis 2001, MRM 45:1014, doi:10.1002/mrm.1135
- Boxerman et al. 1995, MRM 34:555, doi:10.1002/mrm.1910340412
- Kiselev & Posse 1998, PRL 81:5696, doi:10.1103/PhysRevLett.81.5696
- He & Yablonskiy 2009, PNAS 106:13558, doi:10.1073/pnas.0904899106
