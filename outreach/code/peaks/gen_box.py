#!/usr/bin/env python3
"""Precompute panels 5 and 6 of outreach/cmb-peaks.html.

    OMP_NUM_THREADS=2 NPROC=4 python3 gen_box.py

Panel 5 ("the box") is the k-space standing-wave picture: delta_gamma(k, eta)
over a grid of wavenumbers and times, so the page can show a continuum of modes
all released at rest together and all sampled at one instant. The comb of peaks
is what that sampling produces -- selection by phase, not by any boundary
condition. There are no walls; the "box length" is the sound horizon.

Panel 6 ("the ruler") is the same physics in real space, Fourier dual to it:
the response to a single point overdensity, after Eisenstein, Seo & White
(2007). A delta function at eta_i is decomposed into modes, each evolved, and
transformed back,

    profile_X(r) = 1/(2 pi^2) Int dk k^2 [delta_X(k,eta)/delta_cdm(k,eta_i)]
                                        j0(k r) exp(-k^2 sigma^2 / 2)

normalising by the superhorizon adiabatic growing mode so the initial condition
is a unit point perturbation rather than unit primordial curvature (CAMB's
convention, under which delta grows as (k eta)^2 while superhorizon). That
normalisation is taken as C k^2 analytically rather than read off CAMB at an
early time: CAMB begins each mode's integration only once k eta is small, so at
eta = 0.2 the modes below k = 0.08 have not started and come back wrong -- which
is precisely the range carrying acoustic peaks 2 to 4. The pressure wave
runs out as a shell, stalls at r_s when the photons leave, and the dark matter
is drawn to it -- the BAO feature, which is the same acoustic scale the CMB
peaks measure, carried in galaxies instead of photons.

Both run on the matter axis only, at the Planck baryon density, which keeps the
payload to a few hundred KB. omm is the same 9-point axis gen_peaks.py uses.
"""
import os, json, base64, sys
import numpy as np, camb
from multiprocessing import Pool

OMB0 = 0.02237                                  # Planck baryon density, fixed
OMM  = np.linspace(0.112, 0.172, 9)             # same axis as gen_peaks.py
NT   = 36                                       # shared time grid, log in eta
ETA_0 = 5.0                                     # start of the grid; nothing has happened before
ETA_I = 0.2                                     # "initial" time, for the k^2 calibration only
KCAL  = (0.2, 0.6)                              # k window where CAMB is clean at ETA_I

KFT   = np.arange(0.0005, 0.7001, 0.0007)       # fine grid for the transform
SIGMA = 4.0                                     # Mpc, smoothing in the transform
RMAX, NR = 270.0, 90                            # Mpc, radial grid for panel 6
RG    = np.linspace(0.0, RMAX, NR)

NKA   = 160                                     # panel 5 wavenumber grid
KA    = np.linspace(0.002, 0.14, NKA)

_bbn = None
def yhe(ob):
    global _bbn
    if _bbn is None: _bbn = camb.bbn.get_predictor()
    return float(np.asarray(_bbn.Y_He(ob, 0.0)).ravel()[0])

def one(args):
    im, om = args
    oc = om - OMB0
    try:
        p = camb.set_params(thetastar=1.04118/100, ombh2=OMB0, omch2=oc, tau=0.0544,
                            As=np.exp(3.044)*1e-10, ns=0.9649, YHe=yhe(OMB0), lmax=30)
        r = camb.get_results(p); d = r.get_derived_params()
        eta0 = float(r.conformal_time(0))
        ea = np.logspace(np.log10(ETA_0), np.log10(eta0 * 0.999), NT)
        # a first pass for a(eta), so eta_* and eta_eq can be snapped onto the
        # grid: panel 5 at last scattering must be the same curve as panel 3,
        # which it is not if the nearest grid point sits 6% late.
        a0 = r.get_time_evolution(np.array([0.01]), ea, ['a'])[0, :, 0]
        zs, ze = d['zstar'], d['zeq']
        et_st = float(np.interp(1/(1+zs), a0, ea))
        et_eq = float(np.interp(1/(1+ze), a0, ea))
        for t in (et_eq, et_st):
            ea[int(np.argmin(np.abs(ea - t)))] = t
        ea = np.sort(ea)
        av = r.get_time_evolution(np.array([0.01]), ea, ['a'])[0, :, 0]

        # eta_i first, to calibrate the k^2 normalisation, then dropped
        ev = r.get_time_evolution(KFT, np.concatenate([[ETA_I], ea]),
                                  ['delta_cdm', 'delta_baryon', 'delta_photon'])
        cal = (KFT > KCAL[0]) & (KFT < KCAL[1])
        C = float(np.median((ev[cal, 0, 0] / KFT[cal]**2)))
        norm = C * KFT**2                       # the unit point source at eta_i
        ev = ev[:, 1:, :]

        # 6. back to real space. r^2 x profile, as Eisenstein plots it.
        w  = (KFT**2 * np.exp(-0.5 * (KFT*SIGMA)**2) / (2*np.pi**2) / norm)[:, None]
        kr = np.outer(KFT, RG)
        j0 = np.ones_like(kr); nz = kr > 0
        j0[nz] = np.sin(kr[nz]) / kr[nz]
        dk = KFT[1] - KFT[0]
        prof = np.einsum('kts,kr->tsr', ev * w[:, :, None], j0) * dk
        prof *= (RG**2)[None, None, :]

        # 5. the oscillator across k, on its own grid. Left per unit primordial
        # curvature -- the same normalisation panel 3 uses, so at eta_* the comb
        # here and the peaks there are the same curve.
        dgk = np.empty((NT, NKA))
        for it in range(NT):
            dgk[it] = np.interp(KA, KFT, ev[:, it, 2])

        # redshift and sound horizon along the same time grid, so the page can
        # rule a z axis and draw the acoustic ladder n pi / r_s(eta) as it moves
        zz = 1/av - 1
        rs = np.array([r.sound_horizon(max(z, 0.0)) for z in zz])
        # CAMB keeps integrating c_s d(eta) after decoupling, where there is no
        # longer a coupled fluid to carry sound. The acoustic scale is frozen at
        # last scattering, so clamp it there -- otherwise the ladder the page
        # draws would keep marching left through the whole matter era.
        rs = np.minimum(rs, float(np.interp(et_st, ea, rs)))
        return (im, prof.astype(np.float64), dgk.astype(np.float64), ea.astype(np.float32),
                zz.astype(np.float32), rs.astype(np.float32),
                dict(eta_st=et_st, eta_eq=et_eq, eta0=eta0,
                     rs=d['rstar'], zstar=zs, zeq=ze, oc=oc, H0=r.Params.H0))
    except Exception as e:
        sys.stderr.write(f"  FAILED om={om:.4f}: {type(e).__name__}: {e}\n")
        return None

def q16(a, axes):
    """int16 with a float32 scale per slice over `axes` -- the profiles span
    orders of magnitude in time, so one global scale would quantise the early
    frames to nothing."""
    s = np.max(np.abs(a), axis=axes, keepdims=True)
    s[s == 0] = 1.0
    return np.round(a / s * 32000).astype(np.int16), s.astype(np.float32)

def main():
    if 'OMP_NUM_THREADS' not in os.environ:
        sys.stderr.write("WARNING: set OMP_NUM_THREADS (CAMB oversubscribes otherwise)\n")
    jobs = list(enumerate(OMM))
    with Pool(int(os.environ.get('NPROC', 4))) as pool:
        res = [x for x in pool.map(one, jobs) if x is not None]
    nm = len(OMM)
    prof = np.zeros((nm, NT, 3, NR)); dgk = np.zeros((nm, NT, NKA))
    eta  = np.zeros((nm, NT), np.float32); zed = np.zeros((nm, NT), np.float32)
    rse  = np.zeros((nm, NT), np.float32); meta = [None]*nm
    for im, pr, dg, ea, zz, rs, m in res:
        prof[im], dgk[im], eta[im], zed[im], rse[im] = pr, dg, ea, zz, rs
        meta[im] = {k: round(float(v), 4) for k, v in m.items()}
    pq, ps = q16(prof, (3,))            # per model, time, species
    dq, ds = q16(dgk,  (2,))            # per model, time
    b = lambda x, t: base64.b64encode(np.ascontiguousarray(x, t).tobytes()).decode()
    out = dict(omm=[float(x) for x in OMM], omb=OMB0, rgrid=[float(x) for x in RG],
               kgrid=[float(x) for x in KA], nt=NT, nr=NR, nka=NKA, sigma=SIGMA,
               prof=b(pq, np.int16), profs=b(ps, np.float32),
               dgk=b(dq, np.int16),  dgks=b(ds, np.float32),
               eta=b(eta, np.float32), zed=b(zed, np.float32), rse=b(rse, np.float32),
               meta=meta)
    js = json.dumps(out, separators=(',', ':'))
    open('box_data.json', 'w').write(js)
    print(f"  {len(res)}/{len(jobs)} models -> box_data.json, {len(js)/1024:.0f} KB")

if __name__ == '__main__':
    main()
