"""Mesoscopic GRE / SE voxel-signal simulation from sub-voxel magnetic sources.

A voxel is modelled as a periodic box of tissue holding

* myelinated axons: straight, parallel hollow cylinders along the box z axis with a
  radially anisotropic susceptibility tensor in the sheath (Wharton & Bowtell 2012),
  packed in 2D by random sequential addition;
* iron clusters: randomly placed (Poisson, overlap allowed) uniformly magnetised
  spheres of susceptibility ``dchi_fe`` (Yablonskiy & Haacke 1994).

B0 makes angle ``theta`` with the fibres: b = (sin th, 0, cos th) in box coordinates.
The microscopic field is computed with the Lorentz-corrected susceptibility-tensor
dipole kernel (Liu 2010, STI) with D(0) = 0, i.e. as a deviation from the macroscopic
field that qsm-forward computes from the voxel-averaged susceptibility::

    dB/B0 = FT^-1{ (1/3) b^T X b - (k.b)(k^T X b)/|k|^2 }

Because sources are linear and the geometry is fixed, the field for any theta, any
source amplitude and any B0 is a linear combination of a few theta-independent
"basis" maps:

    myelin (z-invariant, kz = 0):  f = s^2 * M1 + M2
    spheres:                       f = dchi * (s^2 Gxx + 2 s c Gxz + c^2 Gzz)

Water "walkers" record time integrals of the basis maps along their paths (static:
path = fixed point; diffusion: Monte Carlo random walk with impermeable myelin
membranes, permeable spheres), so a single walk serves every theta / B0 / amplitude.

Units: length um, time ms, susceptibility ppm, gamma in rad/ms/T.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import integrate, special

GAMMA = 2.6752218744e5  # rad / ms / T  (1H)

# compartment labels
EXTRA, MYELIN, AXON = 0, 1, 2
COMPARTMENTS = ("extra", "myelin", "axon")


# ----------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------
@dataclass
class AxonPacking:
    L: float                 # box side (um)
    x: np.ndarray
    y: np.ndarray
    r_out: np.ndarray
    r_in: np.ndarray

    @property
    def fvf(self):
        return float(np.pi * np.sum(self.r_out ** 2) / self.L ** 2)

    @property
    def mvf(self):
        return float(np.pi * np.sum(self.r_out ** 2 - self.r_in ** 2) / self.L ** 2)


def pack_axons(L, fvf, g=0.7, r_mean=0.6, r_shape=4.0, r_min=0.25, r_max=2.0,
               gap=0.01, rng=None, max_tries=4000):
    """Random sequential addition of non-overlapping discs in a periodic L x L box.

    Outer radii ~ Gamma(shape=r_shape, mean=r_mean) clipped to [r_min, r_max]; placed
    largest-first (reaches FVF ~0.6-0.65). Inner radius = g * outer radius. Stops when
    the target fibre volume fraction is reached; raises if it cannot be."""
    rng = np.random.default_rng(rng)
    if fvf <= 0:
        e = np.zeros(0)
        return AxonPacking(L, e, e, e, e)
    radii = []
    area = 0.0
    while area < fvf * L ** 2:
        r = float(np.clip(rng.gamma(r_shape, r_mean / r_shape), r_min, r_max))
        radii.append(r)
        area += np.pi * r ** 2
    radii = np.sort(np.array(radii))[::-1]
    X, Y, R = np.empty(len(radii)), np.empty(len(radii)), np.empty(len(radii))
    n = 0
    for r in radii:
        for _ in range(max_tries):
            px, py = rng.uniform(0, L, 2)
            dx = X[:n] - px
            dy = Y[:n] - py
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            if n == 0 or np.all(dx * dx + dy * dy > (R[:n] + r + gap) ** 2):
                X[n], Y[n], R[n] = px, py, r
                n += 1
                break
        else:
            # cannot place this one; try the remaining (smaller) ones
            continue
    pk = AxonPacking(L, X[:n], Y[:n], R[:n], g * R[:n])
    if pk.fvf < 0.97 * fvf:
        raise RuntimeError(f"packing reached FVF {pk.fvf:.3f} < target {fvf:.3f}")
    return pk


def place_spheres(L, zeta, radius, rng=None):
    """Poisson-like placement: N = round(zeta * L^3 / V_sphere) uniform random centres,
    overlap allowed (superposed sources), the setting of the Y&H statistical theory."""
    rng = np.random.default_rng(rng)
    n = int(round(zeta * L ** 3 / (4 / 3 * np.pi * radius ** 3)))
    return rng.uniform(0, L, (n, 3))


# ----------------------------------------------------------------------------------
# rasterisation (linear anti-aliasing on signed distance)
# ----------------------------------------------------------------------------------
def _soft(r, d, dx):
    return np.clip(0.5 + (r - d) / dx, 0.0, 1.0)


def raster_axons(pk: AxonPacking, n, chi_perp, chi_a):
    """Rasterise the myelin tensor on an n x n periodic grid.

    Sheath tensor: chi = chi_perp*I + chi_a * r r^T (r = in-plane radial unit vector),
    i.e. chi_rr = chi_perp + chi_a, chi_phiphi = chi_zz = chi_perp (Wharton & Bowtell
    2012: chi_I = chi_perp, chi_A = chi_par - chi_perp, parallel = radial).

    Returns (chi_xx, chi_xy, chi_zz, label) with label in {EXTRA, MYELIN, AXON}
    (label by pixel centre)."""
    L = pk.L
    dx = L / n
    cxx = np.zeros((n, n))
    cxy = np.zeros((n, n))
    czz = np.zeros((n, n))
    lab = np.zeros((n, n), np.int8)
    coords = (np.arange(n) + 0.5) * dx
    for x0, y0, ro, ri in zip(pk.x, pk.y, pk.r_out, pk.r_in):
        h = int(np.ceil(ro / dx)) + 2
        ix0 = int(np.floor(x0 / dx))
        iy0 = int(np.floor(y0 / dx))
        ix = np.arange(ix0 - h, ix0 + h + 1)
        iy = np.arange(iy0 - h, iy0 + h + 1)
        X = (ix + 0.5) * dx - x0
        Y = (iy + 0.5) * dx - y0
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        d = np.hypot(XX, YY)
        occ = _soft(ro, d, dx) - _soft(ri, d, dx)
        dd = np.maximum(d, 1e-12)
        c, s = XX / dd, YY / dd
        sl = np.ix_(ix % n, iy % n)
        cxx[sl] += occ * (chi_perp + chi_a * c * c)
        cxy[sl] += occ * (chi_a * c * s)
        czz[sl] += occ * chi_perp
        lab_loc = lab[sl]
        lab_loc[d <= ro] = MYELIN
        lab_loc[d <= ri] = AXON
        lab[sl] = lab_loc
    del coords
    return cxx, cxy, czz, lab


def raster_spheres(centres, radius, L, n):
    """Anti-aliased indicator sum of spheres on an n^3 periodic grid."""
    dx = L / n
    S = np.zeros((n, n, n), np.float32)
    h = int(np.ceil(radius / dx)) + 2
    off = np.arange(-h, h + 1)
    for c in centres:
        i0 = np.floor(c / dx).astype(int)
        ix, iy, iz = (i0[k] + off for k in range(3))
        X = (ix + 0.5) * dx - c[0]
        Y = (iy + 0.5) * dx - c[1]
        Z = (iz + 0.5) * dx - c[2]
        d = np.sqrt(X[:, None, None] ** 2 + Y[None, :, None] ** 2 + Z[None, None, :] ** 2)
        S[np.ix_(ix % n, iy % n, iz % n)] += _soft(radius, d, dx).astype(np.float32)
    return S


# ----------------------------------------------------------------------------------
# field basis maps
# ----------------------------------------------------------------------------------
def _kgrid(n, L, ndim):
    k = np.fft.fftfreq(n, d=L / n)
    if ndim == 2:
        return np.meshgrid(k, k, indexing="ij")
    return np.meshgrid(k, k, k, indexing="ij")


def myelin_basis(cxx, cxy, czz, L):
    """2D basis maps (ppm) for fibres along z: field(theta) = s^2*M1 + M2.

    With kz = 0 and b = (s, 0, c):  b^T X b /3 = (s^2 Xxx + c^2 Xzz)/3,
    (k.b)(k^T X b)/k^2 = s^2 kx (kx Xxx + ky Xxy)/k^2. D(0) = 0 (zero mean)."""
    n = cxx.shape[0]
    kx, ky = _kgrid(n, L, 2)
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = np.inf
    Fxx = np.fft.fft2(cxx)
    Fxy = np.fft.fft2(cxy)
    A = Fxx / 3 - kx * (kx * Fxx + ky * Fxy) / k2
    A[0, 0] = 0
    A = np.real(np.fft.ifft2(A))
    zz = czz / 3 - czz.mean() / 3
    return (A - zz).astype(np.float32), zz.astype(np.float32)


def sphere_basis(S, L):
    """3D basis maps for unit-susceptibility spheres: Gxx, Gxz, Gzz (D(0)=0)."""
    n = S.shape[0]
    kx, ky, kz = _kgrid(n, L, 3)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = np.inf
    F = np.fft.fftn(S)
    out = []
    for kern in (1 / 3 - kx ** 2 / k2, -kx * kz / k2, 1 / 3 - kz ** 2 / k2):
        G = kern * F
        G[0, 0, 0] = 0
        out.append(np.real(np.fft.ifftn(G)).astype(np.float32))
    return out


# ----------------------------------------------------------------------------------
# walkers
# ----------------------------------------------------------------------------------
class Sampler2D:
    """Nearest-pixel lookup on a periodic 2D grid of side L."""

    def __init__(self, maps, L):
        self.maps = np.stack(maps, -1)  # (n, n, m)
        self.n = self.maps.shape[0]
        self.L = L

    def idx(self, xy):
        i = np.floor(xy / self.L * self.n).astype(np.int64) % self.n
        return i[:, 0], i[:, 1]

    def __call__(self, xy):
        i, j = self.idx(xy)
        return self.maps[i, j]


class Sampler3D:
    """Trilinear lookup on a periodic 3D grid (maps stacked last-axis)."""

    def __init__(self, maps, L):
        n = maps[0].shape[0]
        self.n = n
        self.L = L
        self.flat = np.stack([m.ravel() for m in maps], -1)  # (n^3, m)

    def __call__(self, xyz):
        n = self.n
        u = xyz / self.L * n - 0.5
        i0 = np.floor(u).astype(np.int64)
        f = u - i0
        out = 0.0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            ix = (i0[:, 0] + dx) % n
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                iy = (i0[:, 1] + dy) % n
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    iz = (i0[:, 2] + dz) % n
                    w = (wx * wy * wz)[:, None]
                    out = out + w * self.flat[(ix * n + iy) * n + iz]
        return out


@dataclass
class Voxel:
    """A simulated tissue box with its basis maps."""
    L: float
    packing: AxonPacking
    label2d: np.ndarray
    myelin2d: Sampler2D                 # maps (M1, M2), ppm per unit sheath scale
    sphere3d: Sampler3D | None          # maps (Gxx,Gxz,Gzz) x n_levels, unit chi
    sphere_levels: list = field(default_factory=list)   # zeta per nested level
    chi_sheath_mean: float = 0.0        # tr/3 of sheath tensor (ppm)


def build_voxel(L, fvf, zetas, sphere_radius, n2d=4096, n3d=128, g=0.7,
                chi_i=-0.12, chi_a=-0.10, rng=None, axon_kw=None):
    """Build geometry + basis maps.

    Sheath tensor in the Wharton & Bowtell (2012) parameterisation,
    chi = chi_i*I + chi_a*diag(1, -1/2, -1/2) in (r, phi, z), so tr/3 = chi_i (the
    volume-averaged mean susceptibility of the sheath is chi_i).

    ``zetas``: increasing list of sphere volume fractions; the sphere sets are nested
    (level k uses the first N_k spheres), and basis maps are stored per *increment*
    so that the cumulative field at level k is the sum over increments <= k."""
    rng = np.random.default_rng(rng)
    pk = pack_axons(L, fvf, g=g, rng=rng, **(axon_kw or {}))
    cxx, cxy, czz, lab = raster_axons(pk, n2d, chi_i - chi_a / 2, 1.5 * chi_a)
    M1, M2 = myelin_basis(cxx, cxy, czz, L)
    del cxx, cxy, czz
    s2d = Sampler2D([M1, M2], L)
    s3d = None
    zetas = [z for z in zetas if z > 0]
    if zetas:
        allc = place_spheres(L, max(zetas), sphere_radius, rng)
        nk = [int(round(z * L ** 3 / (4 / 3 * np.pi * sphere_radius ** 3))) for z in zetas]
        maps = []
        prev = 0
        for k in nk:
            S = raster_spheres(allc[prev:k], sphere_radius, L, n3d)
            maps += sphere_basis(S, L)
            prev = k
        s3d = Sampler3D(maps, L)
    chi_mean = chi_i   # tr/3 of the sheath tensor
    return Voxel(L, pk, lab, s2d, s3d, zetas, chi_mean)


def init_walkers(vox: Voxel, n_per_comp, rng=None):
    """Uniform random positions, n_per_comp walkers in each present compartment.
    Returns (pos (W,3), comp (W,), volume fractions dict)."""
    rng = np.random.default_rng(rng)
    lab = vox.label2d
    frac = {c: float(np.mean(lab == i)) for i, c in enumerate(COMPARTMENTS)}
    pos, comp = [], []
    for ci, cname in enumerate(COMPARTMENTS):
        if frac[cname] == 0:
            continue
        got = 0
        while got < n_per_comp:
            p = rng.uniform(0, vox.L, (4 * n_per_comp, 3))
            i, j = vox.myelin2d.idx(p[:, :2])
            p = p[lab[i, j] == ci][: n_per_comp - got]
            pos.append(p)
            comp.append(np.full(len(p), ci, np.int8))
            got += len(p)
    return np.concatenate(pos), np.concatenate(comp), frac


def _basis_values(vox, pos):
    v = vox.myelin2d(pos[:, :2])
    if vox.sphere3d is not None:
        v = np.concatenate([v, vox.sphere3d(pos)], axis=1)
    return v.astype(np.float64)


def walk(vox: Voxel, pos, comp, t_out, D=0.0, dt=2e-3, rng=None, progress=False):
    """Phase integrals of every basis map along each walker path.

    Returns I (n_t, W, n_basis): I[k] = int_0^{t_out[k]} map(r(t')) dt'  (ppm*ms).
    D in um^2/ms. D = 0 is static dephasing (I = t * map(r0)). Myelin membranes are
    impermeable (a step that leaves the walker's compartment is rejected); spheres are
    permeable and do not affect the walk."""
    t_out = np.asarray(t_out, float)
    if D == 0:
        v = _basis_values(vox, pos)
        return t_out[:, None, None] * v[None].astype(np.float32)
    rng = np.random.default_rng(rng)
    sig = np.sqrt(2 * D * dt)
    pos = pos.copy()
    W = len(pos)
    v = _basis_values(vox, pos)
    I = np.zeros_like(v)
    out = np.zeros((len(t_out), W, v.shape[1]), np.float32)
    steps = np.round(t_out / dt).astype(int)
    k_out = 0
    while k_out < len(steps) and steps[k_out] == 0:
        k_out += 1
    lab = vox.label2d
    nsteps = steps.max()
    for s in range(1, nsteps + 1):
        new = pos + rng.normal(0, sig, pos.shape)
        i, j = vox.myelin2d.idx(new[:, :2])
        ok = lab[i, j] == comp
        pos[ok] = new[ok]
        vn = _basis_values(vox, pos)
        I += 0.5 * (v + vn) * dt   # trapezoid
        v = vn
        while k_out < len(steps) and steps[k_out] == s:
            out[k_out] = I
            k_out += 1
        if progress and s % 5000 == 0:
            print(f"    step {s}/{nsteps}", flush=True)
    return out


def phase_coeffs(vox: Voxel, theta, level=None, dchi_fe=0.0, sheath_scale=1.0):
    """Coefficient vector c such that field (ppm) = basis . c, for fibre-to-B0 angle
    theta, sheath susceptibility scale and iron level (index into vox.sphere_levels,
    None = no iron)."""
    s, c = np.sin(theta), np.cos(theta)
    coef = [sheath_scale * s * s, sheath_scale]
    if vox.sphere3d is not None:
        nl = len(vox.sphere_levels)
        for k in range(nl):
            on = level is not None and k <= level
            w = dchi_fe if on else 0.0
            coef += [w * s * s, w * 2 * s * c, w * c * c]
    return np.array(coef)


def compartment_signals(I, comp, coef, B0, se_index=None):
    """Mean complex signal per compartment exp(i*gamma*B0*phi), phi = I . coef (1e-6).

    I: (n_t, W, n_basis). Returns dict comp -> complex array (n_t,). If ``se_index``
    is given ((k_te, k_half) pairs) also returns spin-echo signals using
    phi_SE(TE) = phi(TE) - 2 phi(TE/2)."""
    phi = (I @ coef) * (GAMMA * B0 * 1e-6)  # (n_t, W) rad
    gre, se = {}, {}
    for ci, cname in enumerate(COMPARTMENTS):
        m = comp == ci
        if not m.any():
            continue
        gre[cname] = np.exp(1j * phi[:, m]).mean(1)
        if se_index is not None:
            kt, kh = se_index
            p = phi[kt][:, m] - 2 * phi[kh][:, m]
            se[cname] = np.exp(1j * p).mean(1)
    return gre, se


# ----------------------------------------------------------------------------------
# analytic references (static dephasing, Yablonskiy & Haacke 1994)
# ----------------------------------------------------------------------------------
def dw_sphere(dchi_ppm, B0):
    """Characteristic frequency of a sphere, rad/ms: gamma*dchi*B0/3."""
    return GAMMA * B0 * dchi_ppm * 1e-6 / 3


def dw_cylinder(dchi_ppm, B0, theta):
    """Characteristic frequency of a cylinder, rad/ms: gamma*dchi*B0*sin^2(theta)/2."""
    return GAMMA * B0 * dchi_ppm * 1e-6 * np.sin(theta) ** 2 / 2


def yh_f_sphere(x):
    """f(x) with S = exp(-zeta f(dw t)) for Poisson spheres (exterior + interior at 0
    frequency). f(x) = int_0^1 dv/v^2 int_0^1 du [1 - exp(i x v (3u^2-1))] (complex)."""
    x = np.atleast_1d(x)
    out = np.empty(len(x), complex)
    for k, xx in enumerate(x):
        def fr(u, v):
            return (1 - np.cos(xx * v * (3 * u * u - 1))) / (v * v)

        def fi(u, v):
            return -np.sin(xx * v * (3 * u * u - 1)) / (v * v)
        re = integrate.dblquad(fr, 0, 1, 0, 1, epsabs=1e-10)[0]
        im = integrate.dblquad(fi, 0, 1, 0, 1, epsabs=1e-10)[0]
        out[k] = re + 1j * im
    return out


def yh_f_cylinder(x):
    """f(x) for Poisson parallel cylinders, exterior only: int_0^1 dy/y^2 [1-J0(x y)]."""
    x = np.atleast_1d(x)
    return np.array([integrate.quad(lambda y: (1 - special.j0(xx * y)) / y ** 2, 0, 1,
                                    limit=400)[0] for xx in x])


def yh_r2p_sphere(zeta, dchi_ppm, B0):
    """Y&H static-dephasing R2' for spheres (1/ms): 2pi/(9 sqrt3) gamma dchi B0 zeta."""
    return 2 * np.pi / (9 * np.sqrt(3)) * GAMMA * dchi_ppm * 1e-6 * B0 * zeta


def yh_r2p_cylinder(zeta, dchi_ppm, B0, theta):
    """Y&H static-dephasing R2' for parallel cylinders (1/ms): zeta*dw_c."""
    return zeta * dw_cylinder(dchi_ppm, B0, theta)
