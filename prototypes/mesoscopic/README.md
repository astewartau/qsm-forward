# Mesoscopic χ-separation signal model (prototype)

Simulates the gradient-echo (and spin-echo) signal of a voxel from explicitly placed
sub-voxel magnetic sources. No relaxivity (Dr), pool frequency or R2' formula goes into
the signal. R2', the Gaussian-to-exponential short-TE behaviour, orientation dependence
and non-linear phase come out of the field the sources produce.

## What it models

| Source | Geometry | Susceptibility | Signal contribution |
|---|---|---|---|
| Myelinated axons (χ−) | straight parallel hollow cylinders along z, polydisperse outer radius (Gamma, mean 0.6 µm), g-ratio 0.7, random sequential packing, FVF ≤ 0.6 | sheath tensor χ_I·I + χ_A·diag(1, −½, −½) in (r, φ, z), Wharton & Bowtell 2012; defaults χ_I = −0.12, χ_A = −0.10 ppm | three water compartments (extra-axonal, myelin, axonal) |
| Iron clusters (χ+) | spheres (default R = 2.5 µm) at random positions, overlap allowed | Δχ_Fe (default 3 ppm) | dephasing of the water around and inside them |

- Field: the susceptibility-tensor dipole kernel with Lorentz correction and D(0) = 0
  on a periodic box. Frequencies are offsets from the macroscopic field that
  qsm-forward already computes from the voxel-mean χ. B0 is at angle θ to the fibres.
- Ground truth per voxel: χ+ = Δχ_Fe·(sphere volume fraction), χ− = χ_I·(myelin volume
  fraction)·(sheath scale), i.e. volume averages of the mean (tr/3) source
  susceptibility.
- Signal: the mean of exp(iφ) over water "walkers" in each compartment.
  - static: φ = ω(r₀)·t (static dephasing, D = 0).
  - diffusion: Monte Carlo random walk (D = 1 µm²/ms, dt = 2 µs). Myelin membranes are
    impermeable (no exchange), spheres are permeable. Spin echo: φ(TE) − 2φ(TE/2).
- Linearity: for a fixed geometry the field for any θ, B0 and source amplitude is a
  linear combination of 2 myelin and 3·n_levels sphere basis maps. Each walker records
  the time integrals of the basis maps, so a single walk covers every θ, B0 and χ
  amplitude.

Not modelled yet: water exchange between compartments, fibre dispersion or crossing
fibres within a voxel, iron placed relative to the axons (spheres and axons are
superposed), vessels and deoxyhaemoglobin, calcifications, and diffusion anisotropy.

## Files

| File | Purpose |
|---|---|
| `meso.py` | geometry, rasterisation, field basis maps, walkers, Y&H analytic references |
| `validate.py` | static-dephasing checks against analytic limits → `validation.json` |
| `build_lut.py` | one geometry × seed × mode → LUT part (`--list` shows geometries) |
| `lut.py` | `merge` parts into a LUT; `MesoLUT.decay(...)` per-voxel complex factor |
| `calibrate_iron.py` | R2′ / R2(SE) per ppm χ+ vs cluster radius and Δχ_Fe, with diffusion |
| `lut_array.slurm`, `calib_array.slurm` | Bunya job arrays |

## Usage

```bash
PY=/home/uqaste15/miniconda3/envs/qsmxt/bin/python3
$PY validate.py --out OUT/validation
sbatch --export=ALL,MESO_DIR=$PWD,MESO_OUT=OUT/lut_parts lut_array.slurm
$PY lut.py merge --parts OUT/lut_parts --mode diffusion --out OUT/lut_diffusion.npz
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
- Boxerman et al. 1995, MRM 34:555, doi:10.1002/mrm.1910340412
- Kiselev & Posse 1998, PRL 81:5696, doi:10.1103/PhysRevLett.81.5696
- He & Yablonskiy 2009, PNAS 106:13558, doi:10.1073/pnas.0904899106
