#!/usr/bin/env python3
"""Precompute the acoustic-driving story for outreach/cmb-peaks.html.

    OMP_NUM_THREADS=2 NPROC=4 python3 gen_peaks.py

Varies total matter and baryon density on a small grid and, for each model,
records three things that together make the argument:

  1. the Weyl potential and the photon overdensity against conformal time,
     for a few fixed wavenumbers -- the DRIVER and the OSCILLATOR;
  2. the photon overdensity frozen at last scattering, across k -- the
     acoustic peaks, before any projection onto the sky;
  3. the observed TT spectrum.

theta* is held fixed throughout, so the peaks do not slide sideways and any
change in height is unambiguous.

Omega_b h^2 is a separate axis because baryons and cold dark matter do
opposite things to the first peak -- baryons raise it by displacing the
oscillation's zero point, dark matter lowers it by moving matter-radiation
equality earlier and so cutting short the potential decay that drives the
oscillation. A single "total matter" slider that scales both at fixed baryon
fraction largely cancels the first-peak effect (-7% instead of -23%), which
is why the matter axis here moves the CDM and pins the baryons at the BBN
value, as the real analyses do.
"""
import os, json, base64, sys
import numpy as np, camb
from multiprocessing import Pool

OMB  = np.linspace(0.019, 0.026, 5)         # baryon axis
OMM  = np.linspace(0.112, 0.172, 9)         # TOTAL matter axis (omega_b + omega_c)
KSHOW = np.array([0.0187, 0.0408, 0.0622])  # the first three acoustic peaks in k
NETA, NK, LMAX = 160, 220, 2500
KGRID = np.linspace(0.002, 0.10, NK)
LNODES = np.unique(np.concatenate([np.arange(2, 60),
                                   np.arange(60, LMAX + 1, 10), [LMAX]]))

_bbn = None
def yhe(ob):
    global _bbn
    if _bbn is None: _bbn = camb.bbn.get_predictor()
    return float(np.asarray(_bbn.Y_He(ob, 0.0)).ravel()[0])

def one(args):
    ib, im, ob, om = args
    oc = om - ob                            # CDM carries the variation
    try:
        p = camb.set_params(thetastar=1.04118/100, ombh2=ob, omch2=oc, tau=0.0544,
                            As=np.exp(3.044)*1e-10, ns=0.9649, YHe=yhe(ob), lmax=LMAX)
        r = camb.get_results(p); d = r.get_derived_params()
        eta = np.logspace(-0.5, 3.0, 600)
        a   = r.get_time_evolution(np.array([0.01]), eta, ['a'])[0, :, 0]
        eta_st = float(np.interp(1/(1 + d['zstar']), a, eta))
        eta_eq = float(np.interp(1/(1 + d['zeq']),   a, eta))
        # 1. driver and oscillator, on a grid that ends at last scattering
        te = np.logspace(np.log10(0.3), np.log10(eta_st), NETA)
        ev = r.get_time_evolution(KSHOW, te, ['delta_photon', 'Weyl'])
        # CAMB's 'Weyl' is k^2 (Phi+Psi)/2, not the potential: Weyl/k^2 comes back
        # flat at -2/3 across k and across time while all modes are superhorizon,
        # which is the known value of the real potential there. Plotted raw, the
        # three modes start a factor of k^2 apart -- 11x across the range shown --
        # and look like three unrelated quantities. Divided through, they start
        # together, as adiabatic initial conditions require, and the only thing
        # that separates them is how much decay each one caught.
        ev[:, :, 1] /= (KSHOW**2)[:, None]
        # the scale factor on the same grid, so the page can carry a redshift axis
        av = r.get_time_evolution(np.array([0.01]), te, ['a'])[0, :, 0]
        # 2. the frozen pattern across k
        fr = r.get_time_evolution(KGRID, np.array([eta_st]), ['delta_photon'])[:, 0, 0]
        # 3. the observed spectrum
        cl = r.get_cmb_power_spectra(p, CMB_unit='muK', spectra=['total'])['total'][:, 0]
        return (ib, im,
                ev[:, :, 0].astype(np.float32), ev[:, :, 1].astype(np.float32),
                te.astype(np.float32), av.astype(np.float32), fr.astype(np.float32),
                cl[LNODES].astype(np.float32),
                dict(zeq=d['zeq'], zstar=d['zstar'], eta_eq=eta_eq, eta_st=eta_st,
                     rs=d['rstar'], DA=d['DAstar']*1000, H0=r.Params.H0, oc=oc))
    except Exception as e:
        sys.stderr.write(f"  FAILED ob={ob:.4f} om={om:.4f}: {type(e).__name__}\n")
        return None

def main():
    if 'OMP_NUM_THREADS' not in os.environ:
        sys.stderr.write("WARNING: set OMP_NUM_THREADS (CAMB oversubscribes otherwise)\n")
    jobs = [(ib, im, ob, om) for ib, ob in enumerate(OMB) for im, om in enumerate(OMM)]
    with Pool(int(os.environ.get('NPROC', 4))) as pool:
        res = [x for x in pool.map(one, jobs) if x is not None]
    nb, nm = len(OMB), len(OMM)
    dg = np.zeros((nb, nm, len(KSHOW), NETA), np.float32)
    we = np.zeros_like(dg)
    te = np.zeros((nb, nm, NETA), np.float32)
    av = np.zeros((nb, nm, NETA), np.float32)
    fr = np.zeros((nb, nm, NK), np.float32)
    cl = np.zeros((nb, nm, len(LNODES)), np.float32)
    meta = [[None]*nm for _ in range(nb)]
    for ib, im, a, b, t, sa, f, c, m in res:
        dg[ib, im], we[ib, im], te[ib, im] = a, b, t
        av[ib, im], fr[ib, im], cl[ib, im] = sa, f, c
        meta[ib][im] = {k: round(float(v), 4) for k, v in m.items()}
    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x, np.float32).tobytes()).decode()
    out = dict(omb=[float(x) for x in OMB], omm=[float(x) for x in OMM],
               kshow=[float(x) for x in KSHOW], kgrid=b64(KGRID),
               lnodes=[int(x) for x in LNODES],
               dgam=b64(dg), weyl=b64(we), eta=b64(te), avec=b64(av),
               frozen=b64(fr), cl=b64(cl),
               meta=meta, neta=NETA, nk=NK, nl=len(LNODES))
    js = json.dumps(out, separators=(',', ':'))
    open('peaks_data.json', 'w').write(js)
    print(f"  {len(res)}/{len(jobs)} models -> peaks_data.json, {len(js)/1024:.0f} KB")

if __name__ == '__main__':
    main()
