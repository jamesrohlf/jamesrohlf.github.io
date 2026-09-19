#!/usr/bin/env python3
"""Generate a CAMB training set of TT, EE and TE spectra over the 6 LCDM parameters.

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
# theta* rather than H0: it is what the CMB actually constrains, and the peak
# positions follow it directly. The box is trimmed so every corner stays
# physical -- wider in theta* drives H0 past CAMB's ceiling of 100 at high
# ombh2/low omch2, or negative Omega_Lambda at the opposite corner.
PARAMS = [
    ("ombh2",  0.018,  0.027,  0.02237),
    ("omch2",  0.090,  0.150,  0.1200 ),
    ("th100",  1.020,  1.065,  1.04118),
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
    """Return (D_l^TT in muK^2 on l = LMIN..LMAX, derived H0), or None."""
    ombh2, omch2, th100, tau, logAs, ns = theta
    try:
        pars = camb.set_params(
            thetastar=th100/100.0, ombh2=ombh2, omch2=omch2, tau=tau,
            As=np.exp(logAs)*1e-10, ns=ns,
            YHe=yhe(ombh2),                 # BBN-consistent, tracks ombh2
            lmax=CAMB_LMAX, WantTensors=False)
        res = camb.get_results(pars)
        cl  = res.get_cmb_power_spectra(pars, CMB_unit='muK', spectra=['total'])['total']
        # CAMB column order is TT, EE, BB, TE -- verified, not assumed.
        # BB is zero without tensors, so it is dropped.
        tt = cl[LMIN:LMAX+1, 0]
        ee = cl[LMIN:LMAX+1, 1]
        te = cl[LMIN:LMAX+1, 3]
        if not np.all(np.isfinite(tt)) or np.any(tt <= 0):
            return None
        if not np.all(np.isfinite(ee)) or np.any(ee <= 0):
            return None            # EE is positive definite; log-emulated like TT
        if not np.all(np.isfinite(te)):
            return None            # TE changes sign and is emulated linearly
        return (tt.astype(np.float32), ee.astype(np.float32),
                te.astype(np.float32), float(res.Params.H0))
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
    Y  = np.array([out_list[i][0] for i in ok])        # TT
    YEE= np.array([out_list[i][1] for i in ok])
    YTE= np.array([out_list[i][2] for i in ok])
    H0 = np.array([out_list[i][3] for i in ok])
    ell = np.arange(LMIN, LMAX+1)
    np.savez_compressed(out, X=X, Y=Y, YEE=YEE, YTE=YTE, H0=H0, ell=ell,
                        names=np.array([p[0] for p in PARAMS]),
                        lo=LO, hi=HI, fid=FID)
    print(f"  {len(ok)}/{n} succeeded in {time.time()-t0:.0f}s -> {out}")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
