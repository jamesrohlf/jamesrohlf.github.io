#!/usr/bin/env python3
"""Generate a CAMB training set of TT spectra over the 6 LCDM parameters.

Ranges are deliberately wider than Planck's error bars: the point of the page
is to see the spectrum move, which +-1 sigma does not show.
"""
import os, sys, time
import numpy as np, camb
from scipy.stats import qmc
from multiprocessing import Pool

LMIN, LMAX = 2, 2500
CAMB_LMAX  = 2800          # solve past LMAX; the top of a CAMB run is inaccurate

# name, low, high, Planck 2018 fiducial
PARAMS = [
    ("ombh2",  0.017,  0.028,  0.02237),
    ("omch2",  0.080,  0.160,  0.1200 ),
    ("H0",     55.0,   80.0,   67.36  ),
    ("tau",    0.010,  0.120,  0.0544 ),
    ("logAs",  2.70,   3.30,   3.044  ),   # ln(1e10 As)
    ("ns",     0.900,  1.020,  0.9649 ),
]
LO = np.array([p[1] for p in PARAMS])
HI = np.array([p[2] for p in PARAMS])
FID= np.array([p[3] for p in PARAMS])

_bbn = None
def yhe(ombh2):
    global _bbn
    if _bbn is None:
        _bbn = camb.bbn.get_predictor()
    return float(np.asarray(_bbn.Y_He(ombh2, 0.0)).ravel()[0])

def spectrum(theta):
    """Return D_l^TT in muK^2 on l = LMIN..LMAX, or None if CAMB fails."""
    ombh2, omch2, H0, tau, logAs, ns = theta
    try:
        pars = camb.set_params(
            H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
            As=np.exp(logAs)*1e-10, ns=ns,
            YHe=yhe(ombh2),                 # BBN-consistent, tracks ombh2
            lmax=CAMB_LMAX, WantTensors=False)
        res = camb.get_results(pars)
        cl  = res.get_cmb_power_spectra(pars, CMB_unit='muK', spectra=['total'])['total']
        d   = cl[LMIN:LMAX+1, 0]
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            return None
        return d.astype(np.float64)
    except Exception:
        return None

def main(n, seed, out):
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    X = LO + sampler.random(n) * (HI - LO)
    t0 = time.time()
    # 7 workers exhausted 8 GB when other things were running; each CAMB
    # process holds a few hundred MB. Override with NPROC if you have more.
    with Pool(int(os.environ.get('NPROC', 4))) as pool:
        out_list = pool.map(spectrum, list(X), chunksize=4)
    ok = [i for i, d in enumerate(out_list) if d is not None]
    X  = X[ok]
    Y  = np.array([out_list[i] for i in ok])
    ell = np.arange(LMIN, LMAX+1)
    np.savez_compressed(out, X=X, Y=Y, ell=ell,
                        names=np.array([p[0] for p in PARAMS]),
                        lo=LO, hi=HI, fid=FID)
    print(f"  {len(ok)}/{n} succeeded in {time.time()-t0:.0f}s -> {out}")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
