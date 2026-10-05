# cmb-reverse.html

The inverse problem. Every other CMB page here runs forwards — pick parameters,
compute a curve. This one takes the Planck measurement and solves backwards for
the densities, the sound horizon, the distance to last scattering and H₀.

## Build

    OMP_NUM_THREADS=2 NPROC=4 python3 gen_reverse.py    # 63 CAMB models, ~60 s
    python3 assemble.py                                 # -> ../../cmb-reverse.html

Edit the fragments, never the built page: an edit to `cmb-reverse.html` is
silently lost on the next rebuild.

## The chain

| | from | to |
|---|---|---|
| 1 | comb spacing Δℓ | θ\* = acoustic scale |
| 2 | peak 1 / peak 2 | Ω_b h² (baryon loading shifts the zero point) |
| 3 | peak 3 / peak 2 | Ω_m h² (radiation driving sets the envelope) |
| 4 | Ω_b h², Ω_m h² | r_\* ≈ 145 Mpc — **computed, not measured** |
| 5 | r_\*/θ\* | D_A ≈ 13.9 Gpc |
| 6 | D_A, Ω_m h² | H₀ |

Steps 5 and 6 run in the browser: a flat-ΛCDM distance integral and a bisection
on h, with a per-node calibration factor reconciling it with CAMB.

## Things that bit, and must not be undone

- **The data's seventh peak near ℓ = 2000 is noise.** At the data's own sampling
  the best-fit model has no seventh maximum. Including it drags the fitted comb
  spacing by 1%, which is 5% in H₀. `NPEAK = 5`.
- **The distance integral must be taken in x = ln(1+z), not in z.** Linear
  sampling is 1.1% wrong at 1200 steps and still 0.1% wrong at 4000; in x it is
  exact by 800. Worse, the page and the calibration once used different step
  counts, so `cal` was correcting an integral nobody evaluated — about 5% of H₀.
  `NZ_INT` must stay identical in `gen_reverse.py` and `js.html`.
- **Model and data must be measured identically.** Peak positions and heights
  are located on the Planck ℓ binning, by the same routine, with heights taken
  at the located maxima (not at the comb teeth). Locating peaks on 30-wide bins
  biases the spacing by half a percent and the bias only cancels if both sides
  are treated alike.
- **Peak-finding must not use fixed ℓ bands.** They break the moment θ\* moves
  the peaks, silently returning the wrong ones.
- **Heights are read at the data's own peaks, once.** Reading them at the comb
  teeth let the reader's comb corrupt the density measurement.

## Why H₀ is the hard one

H₀ is about five times more sensitive to the acoustic scale than the acoustic
scale is to anything else: one unit of Δℓ out of ~300 is 1.2 in H₀. Hence the
page's spine — a ruler gives H₀ = 72.2 ± 1.2, four sigma from Planck; fitting
the whole curve gives 68.9 ± 1.2, which is 1.3σ. Uncertainties are propagated
from Planck's error bars on the three peak heights, and the table reports
offsets in sigma rather than percent, because a 2% offset that is half a sigma
is agreement.

## Accuracy

Against Planck 2018, after fitting the whole curve: θ\* exact to four decimals,
Ω_b h² 0.8σ, Ω_m h² 0.7σ, r_\* 0.4σ, H₀ 1.3σ, χ²/N = 1.30 over 64 bands with
the amplitude free. Ω_m h² is the loosest link: R32 moves only ~3% across an 8%
change in the matter density.
