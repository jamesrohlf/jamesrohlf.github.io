# CMB power spectrum emulator

Builds the emulator embedded in `outreach/cmb-spectrum.html`. A Boltzmann code
cannot run in a browser, so CAMB is run offline and compressed into something a
page can evaluate on every slider move.

## Pipeline

    NPROC=4 python3 gen_theta.py 4000 7 train.npz   # CAMB over a Latin hypercube (~55 min)
    NPROC=4 python3 gen_theta.py  600 99 test.npz   # independent held-out set    (~8 min)
    python3 mlp.py  64 160 90000        # PCA(64) + 6->160->160->64 tanh MLP
    python3 payload.py emulator_mlp.npz # base64 float32 + accuracy numbers
    python3 assemble.py cmb-spectrum.html

`assemble.py` substitutes the payload and the Planck points into `cmb.js.html`
and concatenates `cmb.head.html` + `cmb.prose.html` + the script.

`gen.py` is the older H0-parameterised generator, kept for reference.
`gen_theta.py` is the one that built what ships.

## Three things that will bite

**CAMB 1.6.6 against numpy >= 2.** `camb.set_params` raises
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
because the BBN predictor hands back an array. `gen.py` calls the predictor
itself and passes `YHe` explicitly. Note it still *tracks* `ombh2` through the
BBN table rather than being frozen -- that matters, since `ombh2` is a slider.

**Do not standardise the PCA targets per component.** The basis is orthonormal,
so MSE on the coefficients is exactly the L2 error of log D_l -- which is the
thing worth minimising. Dividing each component by its own sigma reweights the
loss by 1/sigma^2 and spends the network's capacity on components that carry no
variance. Doing that cost a factor of ten in accuracy (median 0.53% against
0.054%). `mlp.py` uses a single global scale for conditioning only.

## Accuracy

Measured against the 600 held-out CAMB models, through the shipped path
(node subsampling and linear interpolation included), for l >= 30:
median 0.03%, worst case 0.48%. Planck's own fractional uncertainty is 0.8% at
l ~ 1000 and 2.3% at l ~ 2000, so emulator error sits inside measurement error
across the plot. Whatever `payload.py` prints is what the page displays -- the
numbers are injected, not typed in.

A polynomial regression was tried first and abandoned: the damping tail's
parameter dependence is too nonlinear, and degree 6 still left 3.8% error at
404 KB against the MLP's 0.48% at 238 KB.

**Do not run 7 workers on 8 GB.** Each CAMB process holds a few hundred MB and
the machine was OOM-killed mid-run, losing a held-out set. Worker count is the
`NPROC` environment variable, default 4.

## theta* rather than H0

theta* is what the CMB measures, so it is the slider; H0 is derived from the two
densities and theta* by a degree-4 polynomial (35 terms, max error 0.004
km/s/Mpc against held-out CAMB -- a quadratic left 0.21, too coarse for the two
decimals the page prints).

Reparameterising forced the box to shrink. In (ombh2, omch2, theta*) the old
corners stop being physical: high ombh2 with low omch2 at 100*theta*=1.08 drives
H0 past CAMB's ceiling of 100, and the opposite corner gives Omega_Lambda < 0.
The shipped box -- ombh2 0.018-0.027, omch2 0.090-0.150, 100*theta* 1.020-1.065
-- was checked corner by corner: H0 48-93, Omega_m 0.14-0.72, no failures.

## Data

Planck 2018, from the ESA Planck Legacy Archive: Commander for l = 2-29
(`COM_PowerSpect_CMB-TT-full_R3.01.txt`) and binned Plik above
(`COM_PowerSpect_CMB-TT-binned_R3.01.txt`).
