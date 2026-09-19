#!/usr/bin/env python3
"""Joint TT/EE/TE emulator: one PCA across all three spectra, one MLP.

TT and EE are emulated in log (both are positive definite). TE changes sign
1900 times and cannot be, so it is carried linearly -- which also means its
accuracy has to be quoted in absolute units, not as a fraction.
"""
import sys, numpy as np

K     = int(sys.argv[1]) if len(sys.argv) > 1 else 96
H     = int(sys.argv[2]) if len(sys.argv) > 2 else 192
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 90000
rng = np.random.default_rng(0)

tr = np.load('train3.npz'); te = np.load('test3.npz')
lo, hi, ell = np.array(tr['lo']), np.array(tr['hi']), tr['ell']
nd  = np.unique(np.concatenate([np.arange(2,61), np.arange(64, ell[-1]+1, 8), [ell[-1]]]))
idx = np.searchsorted(ell, nd)
NL  = len(nd)
nrm = lambda X: 2*(X - lo)/(hi - lo) - 1

def blocks(d):
    return [np.log(d['Y'][:, idx]), np.log(d['YEE'][:, idx]), d['YTE'][:, idx].astype(np.float64)]
Btr, Bte = blocks(tr), blocks(te)
mean  = [b.mean(axis=0) for b in Btr]
scale = [float(np.std(b - m)) for b, m in zip(Btr, mean)]     # one scalar per block
Xtr = np.hstack([(b - m)/s for b, m, s in zip(Btr, mean, scale)])
print(f"  design matrix {Xtr.shape}, block scales {['%.3f'%s for s in scale]}")

U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
basis = Vt[:K]
resid = np.sqrt((S[K:]**2).sum()/(S**2).sum())
C = Xtr @ basis.T
cs = np.full(K, C.std())          # single global scale: see mlp.py
Cs = C / cs
print(f"  PCA K={K}: residual {resid*100:.4f}% of variance")

Z = nrm(tr['X'])
perm = rng.permutation(len(Z)); vi, ti = perm[:400], perm[400:]
Xt, Ct, Xv, Cv = Z[ti], Cs[ti], Z[vi], Cs[vi]
def init(a,b): return rng.normal(0, np.sqrt(2.0/(a+b)), (a,b)), np.zeros(b)
W1,b1 = init(6,H); W2,b2 = init(H,H); W3,b3 = init(H,K)
P = [W1,b1,W2,b2,W3,b3]
m_ = [np.zeros_like(p) for p in P]; v_ = [np.zeros_like(p) for p in P]
def fwd(x, ps):
    h1 = np.tanh(x@ps[0]+ps[1]); h2 = np.tanh(h1@ps[2]+ps[3])
    return h1, h2, h2@ps[4]+ps[5]
best, bestP, bad, lr = np.inf, None, 0, 3e-3
for it in range(1, ITERS+1):
    h1,h2,out = fwd(Xt,P); d = (out-Ct)/len(Xt)
    g = [None]*6
    g[4] = h2.T@d; g[5] = d.sum(0)
    d2 = (d@P[4].T)*(1-h2**2); g[2] = h1.T@d2; g[3] = d2.sum(0)
    d1 = (d2@P[2].T)*(1-h1**2); g[0] = Xt.T@d1; g[1] = d1.sum(0)
    cur = lr*(0.15**(it/ITERS))
    for i,(p,gr) in enumerate(zip(P,g)):
        m_[i] = .9*m_[i] + .1*gr; v_[i] = .999*v_[i] + .001*gr*gr
        p -= cur*(m_[i]/(1-.9**it))/(np.sqrt(v_[i]/(1-.999**it))+1e-8)
    if it % 500 == 0:
        vl = ((fwd(Xv,P)[2]-Cv)**2).mean()
        if vl < best*0.9995: best, bestP, bad = vl, [p.copy() for p in P], 0
        else:
            bad += 1
            if bad >= 30: break
P = bestP
nw = sum(p.size for p in P)
print(f"  MLP H={H}, {nw} weights, stopped at iteration {it}")

# ---- accuracy, held out, through the shipped path ------------------------
rec = (fwd(nrm(te['X']), P)[2]*cs) @ basis
parts = np.split(rec, 3, axis=1)
pred_tt = np.exp(parts[0]*scale[0] + mean[0])
pred_ee = np.exp(parts[1]*scale[1] + mean[1])
pred_te =        parts[2]*scale[2] + mean[2]
def interp(rows): return np.array([np.interp(ell, nd, r) for r in rows])
ftt, fee, fte = interp(pred_tt), interp(pred_ee), interp(pred_te)
band = ell >= 30
e_tt = np.abs(ftt/tr['Y'][:0].reshape(0,-1).shape[0] if False else ftt/te['Y'] - 1)
e_ee = np.abs(fee/te['YEE'] - 1)
d_te = np.abs(fte - te['YTE'])
print(f"  TT  l>=30 : median {np.median(e_tt[:,band])*100:.4f}%  max {e_tt[:,band].max()*100:.3f}%")
print(f"  EE  l>=30 : median {np.median(e_ee[:,band])*100:.4f}%  max {e_ee[:,band].max()*100:.3f}%")
print(f"  TE  l>=30 : median |err| {np.median(d_te[:,band]):.4f}  max {d_te[:,band].max():.3f} uK^2"
      f"   (TE spans +-{np.abs(te['YTE']).max():.0f})")
np.savez('emulator3.npz', basis=basis, cs=cs, mean0=mean[0], mean1=mean[1], mean2=mean[2],
         s0=scale[0], s1=scale[1], s2=scale[2],
         W1=P[0],b1=P[1],W2=P[2],b2=P[3],W3=P[4],b3=P[5],
         lo=lo, hi=hi, fid=tr['fid'], nodes=nd, ell=ell, K=K, H=H)
print(f"  payload: {(nw + basis.size + 3*NL)*4/1024:.0f} KB float32")
