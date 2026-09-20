#!/usr/bin/env python3
"""CAMB training set for the matter power spectrum, over 8 parameters.

    OMP_NUM_THREADS=2 NPROC=4 python3 gen_pk.py 10000 7 train_pk.npz

THREADING. Set BOTH. CAMB uses OpenMP and defaults to one thread per core, so
NPROC=4 alone gives 4 processes x 8 threads = 32 threads on 8 cores, which
spin-waits rather than computing. NPROC * OMP_NUM_THREADS should equal the
core count. Getting this wrong cost a run of over an hour.

tau is deliberately absent: it changes P(k) by 4e-5, acting on CMB photons
rather than on clustering. H0 rather than theta*, since k and P are in h units.

Sum m_nu is sampled from 0 upward only. A negative neutrino mass is not a model
CAMB can solve; the page extrapolates below zero by reflecting the suppression,
as the literature does, and labels it as an extrapolation.

Writes a chunk every CHUNK models so progress is visible and a crash does not
cost the whole run -- both learned the hard way.
"""
import os, sys, time, glob
import numpy as np, camb
from scipy.signal import savgol_filter
from scipy.stats import qmc
from multiprocessing import Pool

KMIN, KMAX, NK = 1e-4, 5.0, 500
ZS    = [2.0, 1.0, 0.5, 0.0]          # descending: CAMB's own ordering
CHUNK = 250

PARAMS = [
    ("ombh2", 0.019,  0.026,  0.02237),
    ("omch2", 0.090,  0.150,  0.1200 ),
    ("H0",    55.0,   80.0,   67.36  ),
    ("logAs", 2.70,   3.30,   3.044  ),
    ("ns",    0.920,  1.010,  0.9649 ),
    ("mnu",   0.0,    0.50,   0.06   ),
    ("w0",   -1.50,  -0.50,  -1.0    ),
    ("wa",   -1.50,   0.50,   0.0    ),
]
LO  = np.array([p[1] for p in PARAMS])
HI  = np.array([p[2] for p in PARAMS])
FID = np.array([p[3] for p in PARAMS])

_bbn = None
def yhe(ob):
    global _bbn
    if _bbn is None: _bbn = camb.bbn.get_predictor()
    return float(np.asarray(_bbn.Y_He(ob, 0.0)).ravel()[0])

def one(theta):
    ob, oc, H0, lA, ns, mnu, w0, wa = theta
    try:
        p = camb.set_params(H0=H0, ombh2=ob, omch2=oc, tau=0.0544, mnu=mnu,
                            As=np.exp(lA)*1e-10, ns=ns, YHe=yhe(ob), lmax=30,
                            w=w0, wa=wa, dark_energy_model='ppf')
        p.set_matter_power(redshifts=ZS, kmax=KMAX*1.6)
        p.NonLinear = camb.model.NonLinear_none
        r  = camb.get_results(p)
        k, z, Pl = r.get_matter_power_spectrum(minkh=KMIN, maxkh=KMAX, npoints=NK)
        p.NonLinear = camb.model.NonLinear_both
        _, _, Pn = camb.get_results(p).get_matter_power_spectrum(
            minkh=KMIN, maxkh=KMAX, npoints=NK)
        if not (np.all(np.isfinite(Pl)) and np.all(Pl > 0)): return None
        if not (np.all(np.isfinite(Pn)) and np.all(Pn > 0)): return None
        # BAO wiggles: divide out a Savitzky-Golay smooth of log P in log k.
        # The window is fixed here, at build time, so the convention cannot
        # drift with whatever the page's sliders happen to be set to.
        lp  = np.log(Pl[-1])                      # z = 0 is last (ZS descending)
        wig = np.exp(lp - savgol_filter(lp, 101, 3)) - 1.0
        d   = r.get_derived_params()
        return (Pl.astype(np.float32), Pn[-1].astype(np.float32),
                wig.astype(np.float32), float(r.get_sigma8()[-1]), float(d['rdrag']))
    except Exception:
        return None

def main(n, seed, out):
    if 'OMP_NUM_THREADS' not in os.environ:
        sys.stderr.write("WARNING: OMP_NUM_THREADS unset -- CAMB will oversubscribe. "
                         "See the docstring.\n")
    X = LO + qmc.LatinHypercube(d=8, seed=seed).random(n) * (HI - LO)
    nproc = int(os.environ.get('NPROC', 4))
    base  = out[:-4] if out.endswith('.npz') else out
    t0, done = time.time(), 0
    with Pool(nproc) as pool:
        for c0 in range(0, n, CHUNK):
            part = list(X[c0:c0+CHUNK])
            res  = pool.map(one, part, chunksize=4)
            ok   = [i for i, v in enumerate(res) if v is not None]
            np.savez_compressed(f"{base}.part{c0//CHUNK:04d}.npz",
                X=X[c0:c0+CHUNK][ok],
                Plin=np.array([res[i][0] for i in ok]),
                Pnl =np.array([res[i][1] for i in ok]),
                wig =np.array([res[i][2] for i in ok]),
                s8  =np.array([res[i][3] for i in ok]),
                rd  =np.array([res[i][4] for i in ok]))
            done += len(ok)
            el = time.time() - t0
            sys.stderr.write(f"  {done}/{n} in {el/60:.1f} min "
                             f"(eta {el/max(done,1)*(n-done)/60:.0f} min)\n")
            sys.stderr.flush()
    parts = sorted(glob.glob(f"{base}.part*.npz"))
    acc = {k: [] for k in ('X','Plin','Pnl','wig','s8','rd')}
    for f in parts:
        d = np.load(f)
        for k in acc: acc[k].append(d[k])
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in acc.items()},
        k=np.geomspace(KMIN, KMAX, NK), zs=np.array(ZS),
        lo=LO, hi=HI, fid=FID, names=np.array([p[0] for p in PARAMS]))
    for f in parts: os.remove(f)
    print(f"  {done}/{n} succeeded in {(time.time()-t0)/60:.1f} min -> {out}")

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
