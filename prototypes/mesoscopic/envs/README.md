# Python environment for the mesoscopic prototype

One self-contained env, `meso`, built with [uv](https://docs.astral.sh/uv/) from the hashed lock file `meso.lock.txt`. It replaces the October venv `/scratch/user/uqaste15/envs/meso`, which ran on the conda-forge Python of `~/miniconda3/envs/seedseg-nnunet` (miniconda3 retired 24 Nov 2026). No conda, no `--system-site-packages`.

Python 3.11.9, numpy 2.4.6, scipy 1.17.1, nibabel 5.4.2, matplotlib 3.11.2 and their dependencies. `meso.in` pins exactly what `pip freeze` showed in the October venv, minus `pip` and `setuptools`; the lock adds sha256 hashes for Linux x86_64 (manylinux_2_28, so Bunya's Rocky 9 works). uv's Python 3.11.9 is a python-build-standalone build, not conda-forge's.

Checked on 24 Nov 2026 (Bunya job 33397190): `calibrate_iron.py --pops ferritin:0.5:3:1 --seed 0 --walkers 8000` and `demo_slice.py --lut lut_static_v2.npz --B0 3` give bit-identical JSON (and PNG) in the old and new env, and match the stored 30 Oct outputs (`calib_v3/`, `demo_v2/`).

## Rebuild

Run in a job, not on a login node. The env lives in `$SCR/projects/chisep/env/meso`; the 90-day scratch purge may delete it, so rebuild it from the lock file.

```bash
UVV=0.13.9; UVD=$SCR/cache/tools/uv-$UVV
[ -x $UVD/uv ] || { mkdir -p $UVD && curl -fsSL https://github.com/astral-sh/uv/releases/download/$UVV/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz --strip-components=1 -C $UVD; }
export UV_CACHE_DIR=$SCR/cache/uv                              # safe to delete
export UV_PYTHON_INSTALL_DIR=$SCR/projects/chisep/env/python    # the interpreter the env links to; keep it with it

ENV=$SCR/projects/chisep/env/meso
$UVD/uv venv --python 3.11.9 "$ENV"
$UVD/uv pip sync --python "$ENV/bin/python" envs/meso.lock.txt
```

The job arrays default to `PY=$SCR/projects/chisep/env/meso/bin/python` (`MESO_PY` overrides it).

To change a version, edit `meso.in`, compile it again and commit both files:

```bash
uv pip compile envs/meso.in -o envs/meso.lock.txt --generate-hashes \
  --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --index-url https://pypi.org/simple --emit-index-url --no-header
```
