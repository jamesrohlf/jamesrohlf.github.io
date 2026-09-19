#!/usr/bin/env python3
"""Turn the joint TT/EE/TE emulator into the page payload, and validate the
shipped path -- node grid plus linear interpolation -- against held-out CAMB."""
import base64, itertools as it, json
import numpy as np

d  = np.load('emulator3.npz'); te = np.load('test3.npz'); tr = np.load('train3.npz')
lo, hi, ell, nd = np.array(d['lo']), np.array(d['hi']), d['ell'], d['nodes']
NL = len(nd); K = int(d['K']); H = int(d['H'])

def fwd(X):
    z  = 2*(X - lo)/(hi - lo) - 1
    h1 = np.tanh(z@d['W1'] + d['b1']); h2 = np.tanh(h1@d['W2'] + d['b2'])
    return (h2@d['W3'] + d['b3']) * d['cs']

rec = fwd(te['X']) @ d['basis']
p_tt, p_ee, p_te = np.split(rec, 3, axis=1)
p_tt = np.exp(p_tt*d['s0'] + d['mean0'])
p_ee = np.exp(p_ee*d['s1'] + d['mean1'])
p_te =        p_te*d['s2'] + d['mean2']
I = lambda R: np.array([np.interp(ell, nd, r) for r in R])
f_tt, f_ee, f_te = I(p_tt), I(p_ee), I(p_te)

band = ell >= 30
e_tt = np.abs(f_tt/te['Y']   - 1)
e_ee = np.abs(f_ee/te['YEE'] - 1)
a_te = np.abs(f_te - te['YTE'])
te_rms = float(np.sqrt((te['YTE']**2).mean()))
print("  SHIPPED-path accuracy vs 600 held-out CAMB models, l >= 30")
print(f"    TT  median {np.median(e_tt[:,band])*100:.4f}%   95th {np.percentile(e_tt[:,band],95)*100:.4f}%   max {e_tt[:,band].max()*100:.3f}%")
print(f"    EE  median {np.median(e_ee[:,band])*100:.4f}%   95th {np.percentile(e_ee[:,band],95)*100:.4f}%   max {e_ee[:,band].max()*100:.3f}%")
print(f"    TE  median {np.median(a_te[:,band]):.4f}      95th {np.percentile(a_te[:,band],95):.4f}      max {a_te[:,band].max():.3f} uK^2")
print(f"        (TE rms amplitude {te_rms:.1f} uK^2, so max error is {a_te[:,band].max()/te_rms*100:.2f}% of rms)")

# derived H0, degree 4 in the two densities and theta*
POW=[tuple(sum(1 for i in c if i==v) for v in range(3))
     for t in range(5) for c in it.combinations_with_replacement(range(3),t)]
def des(X):
    z=[2*(X[:,i]-lo[i])/(hi[i]-lo[i])-1 for i in range(3)]
    return np.column_stack([np.prod([z[i]**e[i] for i in range(3)],axis=0) for e in POW])
h0,*_ = np.linalg.lstsq(des(tr['X']), tr['H0'], rcond=None)
r = des(te['X'])@h0 - te['H0']
print(f"  derived H0: max |error| {np.abs(r).max():.4f} km/s/Mpc")

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
pay = dict(nodes=b64(nd), basis=b64(d['basis']),
           mean0=b64(d['mean0']), mean1=b64(d['mean1']), mean2=b64(d['mean2']),
           W1=b64(d['W1']), b1=b64(d['b1']), W2=b64(d['W2']), b2=b64(d['b2']),
           W3=b64(d['W3']), b3=b64(d['b3']),
           cs=float(d['cs'][0]), s0=float(d['s0']), s1=float(d['s1']), s2=float(d['s2']),
           K=K, H=H, nl=NL,
           lo=[float(x) for x in lo], hi=[float(x) for x in hi],
           fid=[float(x) for x in d['fid']],
           h0coef=[float(x) for x in h0], h0pow=[list(e) for e in POW],
           err_med=float(np.median(e_tt[:,band])*100), err_max=float(e_tt[:,band].max()*100),
           ee_med=float(np.median(e_ee[:,band])*100),  ee_max=float(e_ee[:,band].max()*100),
           te_max=float(a_te[:,band].max()), te_rms=te_rms)
js=json.dumps(pay, separators=(',',':')); open('payload3.json','w').write(js)
print(f"  payload3.json {len(js)/1024:.0f} KB  (K={K} H={H} nodes={NL})")
