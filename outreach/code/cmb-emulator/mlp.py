#!/usr/bin/env python3
"""Emulator: PCA on log D_l + a small tanh MLP on the 6 normalised parameters.

The PCA basis is orthonormal, so MSE on the PCA coefficients is exactly the L2
error of log D_l -- the training loss is the quantity we actually care about,
with no reweighting needed.
"""
import sys, numpy as np

K      = int(sys.argv[1]) if len(sys.argv) > 1 else 48
H      = int(sys.argv[2]) if len(sys.argv) > 2 else 64
ITERS  = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
SEED   = 0

tr = np.load('train.npz'); te = np.load('test.npz')
lo, hi, ell = tr['lo'], tr['hi'], tr['ell']
nrm = lambda X: 2*(X - lo)/(hi - lo) - 1

Ltr = np.log(tr['Y'])
mean = Ltr.mean(axis=0)
U, S, Vt = np.linalg.svd(Ltr - mean, full_matrices=False)
basis = Vt[:K]
C = (Ltr - mean) @ basis.T
# A SINGLE global scale, deliberately not per-component: dividing each
# component by its own sigma would reweight the loss by 1/sigma^2 and destroy
# the property that MSE on coefficients == L2 error of log D_l.
cs = np.full(C.shape[1], C.std())
Cs = C / cs

X = nrm(tr['X'])
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
nval = 400
vi, ti = perm[:nval], perm[nval:]
Xt, Ct = X[ti], Cs[ti]
Xv, Cv = X[vi], Cs[vi]

def init(a, b):
    return rng.normal(0, np.sqrt(2.0/(a+b)), (a, b)), np.zeros(b)
W1,b1 = init(6,H); W2,b2 = init(H,H); W3,b3 = init(H,K)
params = [W1,b1,W2,b2,W3,b3]
m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]

def fwd(x, ps):
    W1,b1,W2,b2,W3,b3 = ps
    h1 = np.tanh(x@W1+b1); h2 = np.tanh(h1@W2+b2)
    return h1, h2, h2@W3+b3

best, best_ps, bad = np.inf, None, 0
lr, b1m, b2m, eps = 3e-3, 0.9, 0.999, 1e-8
for it in range(1, ITERS+1):
    h1, h2, out = fwd(Xt, params)
    d = (out - Ct) / len(Xt)
    gW3 = h2.T@d; gb3 = d.sum(0)
    d2 = (d@params[4].T) * (1-h2**2)
    gW2 = h1.T@d2; gb2 = d2.sum(0)
    d1 = (d2@params[2].T) * (1-h1**2)
    gW1 = Xt.T@d1; gb1 = d1.sum(0)
    grads = [gW1,gb1,gW2,gb2,gW3,gb3]
    cur = lr * (0.15 ** (it/ITERS))            # decay to ~15% of lr
    for i,(p,g) in enumerate(zip(params,grads)):
        m[i] = b1m*m[i] + (1-b1m)*g
        v[i] = b2m*v[i] + (1-b2m)*g*g
        mh = m[i]/(1-b1m**it); vh = v[i]/(1-b2m**it)
        p -= cur*mh/(np.sqrt(vh)+eps)
    if it % 500 == 0:
        vl = ((fwd(Xv, params)[2]-Cv)**2).mean()
        if vl < best*0.9995:
            best, best_ps, bad = vl, [p.copy() for p in params], 0
        else:
            bad += 1
            if bad >= 30: break
params = best_ps

pred = np.exp(mean + (fwd(nrm(te['X']), params)[2]*cs) @ basis)
err = np.abs(pred/te['Y'] - 1); band = ell <= 2000
nw = sum(p.size for p in params)
print(f"  K={K} H={H} iters_used<={it}  weights={nw}")
print(f"  held-out fractional error: median {np.median(err)*100:.4f}%  "
      f"95th {np.percentile(err,95)*100:.4f}%  99.9th {np.percentile(err,99.9)*100:.4f}%  max {err.max()*100:.4f}%")
print(f"    l<=2000:                 median {np.median(err[:,band])*100:.4f}%  max {err[:,band].max()*100:.4f}%")
for a,b in [(2,500),(500,1500),(1500,2500)]:
    msk=(ell>=a)&(ell<b); print(f"    l {a}-{b}: max {err[:,msk].max()*100:.3f}%")
print(f"  payload: weights {nw} + basis {K*365} + mean 365 = {(nw+K*365+365)*4/1024:.0f} KB float32")
np.savez('emulator_mlp.npz', mean=mean, basis=basis, cs=cs,
         W1=params[0],b1=params[1],W2=params[2],b2=params[3],W3=params[4],b3=params[5],
         lo=lo, hi=hi, fid=tr['fid'], ell=ell, names=tr['names'])
