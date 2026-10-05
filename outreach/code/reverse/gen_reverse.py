#!/usr/bin/env python3
"""Precompute the lookup grid for outreach/cmb-reverse.html.

    OMP_NUM_THREADS=2 NPROC=4 python3 gen_reverse.py

The other CMB pages run forwards: move a parameter, watch the curve. This one
runs backwards -- the reader measures features on the Planck data and the page
solves for the cosmology. That needs the map from parameters to observables
tabulated densely enough to invert:

    peak spacing      -> theta*    (the acoustic scale)
    peak 1 / peak 2   -> omega_b   (baryon loading displaces the oscillation)
    peak 3 / peak 2   -> omega_m   (radiation driving sets the envelope)

Height RATIOS, never absolute heights, because the overall normalisation is
A_s exp(-2 tau) and the page must not need it.

theta* is held fixed across the grid. The peak ratios depend on it only through
projection, very weakly, so the shape grid is effectively a function of the two
densities alone; theta* then enters separately as the measured quantity that
converts the computed sound horizon into a distance, and from there into H0.
That last step is done in the browser, by integrating the flat-LCDM distance and
root-finding on h, so it is not limited to this grid's cosmologies.
"""
import os, json, base64, sys
import numpy as np, camb
from multiprocessing import Pool

NPEAK = 5          # the Planck data shows a 6th and 7th maximum, but at its own
                   # sampling the theory has no 7th: that one is noise, and
                   # including it drags the comb fit by a full percent in spacing,
                   # which is five percent in H0. Five teeth is where it settles.
LFIT  = (120, 1700)
OG, ONU, NEFF = 2.47282e-5, 0.06/93.14, 3.044

OMB = np.linspace(0.0180, 0.0270, 7)
OMM = np.linspace(0.1000, 0.1900, 9)
THETA = 1.04109/100
LMAX = 2500
LNODES = np.unique(np.concatenate([np.arange(2, 60), np.arange(60, LMAX+1, 10), [LMAX]]))

_bbn = None
def yhe(ob):
    global _bbn
    if _bbn is None: _bbn = camb.bbn.get_predictor()
    return float(np.asarray(_bbn.Y_He(ob, 0.0)).ravel()[0])

def one(args):
    ib, im, ob, om = args
    try:
        p = camb.set_params(thetastar=THETA, ombh2=ob, omch2=om-ob, tau=0.0544,
                            As=np.exp(3.044)*1e-10, ns=0.9649, YHe=yhe(ob), lmax=LMAX)
        r = camb.get_results(p); d = r.get_derived_params()
        cl = r.get_cmb_power_spectra(p, CMB_unit='muK', spectra=['total'])['total'][:, 0]
        return (ib, im, cl[LNODES].astype(np.float32),
                dict(rs=d['rstar'], DA=d['DAstar']*1000, H0=r.Params.H0,
                     zstar=d['zstar'], zeq=d['zeq'], zdrag=d['zdrag'], rdrag=d['rdrag'],
                     omb=ob, omm=om, theta=d['thetastar']/100))
    except Exception as e:
        sys.stderr.write(f"  FAILED ob={ob:.4f} om={om:.4f}: {type(e).__name__}: {e}\n")
        return None

def observables(y, ell):
    """Locate the acoustic maxima and fit a uniform comb to them. Applied to the
    models on the data's own ell sampling, so that whatever bias the sampling and
    the finite bin width introduce is shared by model and measurement and cancels
    in the ratio."""
    pk, hh = [], []
    for i in range(1, len(ell)-1):
        if not (LFIT[0] <= ell[i] <= LFIT[1]): continue
        if y[i] > y[i-1] and y[i] >= y[i+1]:
            den = y[i-1] - 2*y[i] + y[i+1]
            d = (y[i-1] - y[i+1])/(2*den) if den else 0.0
            pk.append(ell[i] + d*(ell[i+1] - ell[i]))
            hh.append(y[i] - 0.25*(y[i-1] - y[i+1])*d)      # parabola vertex
    pk = np.array(pk[:NPEAK]); n = np.arange(1, len(pk)+1)
    dl, off = np.polyfit(n, pk, 1)
    # Heights at the located maxima, NOT at the fitted comb teeth. The page reads
    # the data's heights at the data's own peaks, and a ratio is only meaningful
    # if model and measurement are read the same way; taking the model's heights
    # off the comb instead put omega_m out by two and a half percent.
    h = [float(x) for x in hh[:3]]
    return dict(dl=float(dl), off=float(off), R12=h[0]/h[1], R32=h[2]/h[1],
                peaks=[float(x) for x in pk])

NZ_INT = 800
def DA_flat(h, omm, zst, nz=NZ_INT):
    """Flat-LCDM comoving distance to last scattering -- bit for bit the integral
    the page evaluates in the browser, so that `cal` below corrects the integral
    that is actually used.

    Substituting x = ln(1+z) matters more than it looks. Sampled linearly in z,
    the integrand is almost all concentrated in the first few units of redshift
    and the quadrature is still 0.1% wrong at 4000 steps -- which, because H0 is
    about five times more sensitive to distance than distance is to anything
    else, is half a unit of H0. In x the same integral is exact by 800 steps."""
    Om = (omm + ONU)/h**2; Or = OG*(1 + 0.2271*2.044)/h**2
    x = np.linspace(0.0, np.log(1.0 + zst), nz + 1)
    z = np.exp(x) - 1.0
    E = np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + 1 - Om - Or)
    return float(np.trapezoid((1.0 + z)/E, x)*2997.92458/h)

def main():
    if 'OMP_NUM_THREADS' not in os.environ:
        sys.stderr.write("WARNING: set OMP_NUM_THREADS (CAMB oversubscribes otherwise)\n")
    jobs = [(ib, im, ob, om) for ib, ob in enumerate(OMB) for im, om in enumerate(OMM)]
    with Pool(int(os.environ.get('NPROC', 4))) as pool:
        res = [x for x in pool.map(one, jobs) if x is not None]
    nb, nm = len(OMB), len(OMM)
    cl = np.zeros((nb, nm, len(LNODES)), np.float32)
    meta = [[None]*nm for _ in range(nb)]
    planck = json.load(open('../cmb-emulator/planck3.json'))['TT']
    pell = np.array([p[0] for p in planck])
    for ib, im, c, m in res:
        cl[ib, im] = c
        d = {k: round(float(v), 6) for k, v in m.items()}
        # measured the way the reader will measure, on the data's ell grid
        d.update({k: (round(v, 5) if not isinstance(v, list) else [round(x, 2) for x in v])
                  for k, v in observables(np.interp(pell, LNODES, c.astype(float)), pell).items()})
        # and the factor that makes the browser's distance integral agree with CAMB
        d['cal'] = round(m['DA']/DA_flat(m['H0']/100, m['omm'], m['zstar']), 6)
        meta[ib][im] = d
    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x, np.float32).tobytes()).decode()
    out = dict(omb=[float(x) for x in OMB], omm=[float(x) for x in OMM],
               lnodes=[int(x) for x in LNODES], nl=len(LNODES),
               cl=b64(cl), meta=meta, theta_grid=THETA, planck=planck)
    js = json.dumps(out, separators=(',', ':'))
    open('reverse_data.json', 'w').write(js)
    print(f"  {len(res)}/{len(jobs)} models -> reverse_data.json, {len(js)/1024:.0f} KB")

if __name__ == '__main__':
    main()
