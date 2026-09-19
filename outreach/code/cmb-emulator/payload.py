#!/usr/bin/env python3
"""Turn a trained emulator into the base64 float32 payload the page embeds,
then validate the *shipped* path (node subsampling + linear interp) against
held-out CAMB, so the quoted accuracy is what the browser actually delivers."""
import base64, json, sys
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else 'emulator_mlp.npz'
d  = np.load(src); te = np.load('test.npz')
ell, lo, hi = d['ell'], d['lo'], d['hi']

# l nodes: every l to 60 (plateau + first rise), then every 8. Linear interp
# between them is far finer than the plot's pixel grid.
nd  = np.unique(np.concatenate([np.arange(2,61), np.arange(64, ell[-1]+1, 8), [ell[-1]]]))
idx = np.searchsorted(ell, nd)

def predict_nodes(X):
    Z  = 2*(X - lo)/(hi - lo) - 1
    h1 = np.tanh(Z@d['W1'] + d['b1']); h2 = np.tanh(h1@d['W2'] + d['b2'])
    return np.exp(d['mean'][idx] + ((h2@d['W3'] + d['b3'])*d['cs']) @ d['basis'][:, idx])

shipped = np.array([np.interp(ell, nd, row) for row in predict_nodes(te['X'])])
err = np.abs(shipped/te['Y'] - 1)
hi_l = ell >= 30
print(f"  SHIPPED-path error vs held-out CAMB ({len(te['X'])} models)")
print(f"    all l:   median {np.median(err)*100:.4f}%  95th {np.percentile(err,95)*100:.4f}%  max {err.max()*100:.3f}%")
print(f"    l >= 30: median {np.median(err[:,hi_l])*100:.4f}%  95th {np.percentile(err[:,hi_l],95)*100:.4f}%  max {err[:,hi_l].max()*100:.3f}%")

# H0 is a background-only quantity: it depends on ombh2, omch2 and theta* alone,
# and not at all on tau, As or ns. A quadratic in those three is plenty -- the
# residual is printed below so it cannot be assumed.
import itertools as _it
tr = np.load('train.npz')
H0DEG = 4                                    # quadratic left 0.21 km/s/Mpc; this leaves 0.004
H0POW = [tuple(sum(1 for i in c if i == v) for v in range(3))
         for t in range(H0DEG+1) for c in _it.combinations_with_replacement(range(3), t)]
def h0design(X):
    z = [2*(X[:,i]-lo[i])/(hi[i]-lo[i])-1 for i in range(3)]
    cols = []
    for e in H0POW:
        col = np.ones(len(X))
        for i, k in enumerate(e): col *= z[i]**k
        cols.append(col)
    return np.column_stack(cols)
h0coef,*_ = np.linalg.lstsq(h0design(tr['X']), tr['H0'], rcond=None)
h0res = h0design(te['X'])@h0coef - te['H0']
print(f"  derived H0 (degree {H0DEG}, {len(H0POW)} terms): max |error| = {np.abs(h0res).max():.4f} "
      f"km/s/Mpc (median {np.median(np.abs(h0res)):.5f}) over held-out models")

def b64(a, dt=np.float32):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()

payload = dict(
    nodes   = b64(nd, np.float32),
    mean    = b64(d['mean'][idx]),
    basis   = b64(d['basis'][:, idx]),
    W1=b64(d['W1']), b1=b64(d['b1']),
    W2=b64(d['W2']), b2=b64(d['b2']),
    W3=b64(d['W3']), b3=b64(d['b3']),
    cs      = float(d['cs'][0]),
    K       = int(d['basis'].shape[0]),
    H       = int(d['W1'].shape[1]),
    nl      = int(len(nd)),
    lo      = [float(x) for x in lo],
    hi      = [float(x) for x in hi],
    fid     = [float(x) for x in d['fid']],
    h0coef  = [float(x) for x in h0coef],
    h0pow   = [list(e) for e in H0POW],
    err_med = float(np.median(err[:,hi_l])*100),
    err_max = float(err[:,hi_l].max()*100),
)
js = json.dumps(payload, separators=(',',':'))
open('payload.json','w').write(js)
print(f"  payload.json: {len(js)/1024:.0f} KB  (K={payload['K']} H={payload['H']} nodes={payload['nl']})")
