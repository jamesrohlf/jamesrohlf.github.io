# cmb-peaks.html

How the matter density sets the heights of the CMB acoustic peaks, from the
sound wave in the early universe through to the observed spectrum, and then the
same acoustic scale seen as a standing-wave ladder and as the BAO ruler.

## Build

    OMP_NUM_THREADS=2 NPROC=4 python3 gen_peaks.py    # panels 1-4   (~35 s, 45 CAMB models)
    OMP_NUM_THREADS=2 NPROC=4 python3 gen_box.py      # panels 5-6   (~20 s,  9 CAMB models)
    python3 assemble.py                               # -> ../../cmb-peaks.html

`assemble.py` composes `head.html` + `mid.html` (with `box.html` spliced in at
`<!--BOX-->`) + `js.html` (with `boxjs.html` at `__BOXJS__` and the two data
files at `__DATA__` and `__BOX__`). Edit the fragments, never the built page:
an edit made to `cmb-peaks.html` directly is silently lost on the next rebuild.

Both JSON payloads are committed so prose can be revised without CAMB
installed. Re-run the generators only when the physics or the grids change.

## What the panels show

| | |
|---|---|
| 1 | Weyl potential against conformal time, three wavenumbers — the driver |
| 2 | photon overdensity, same three — the oscillator |
| 3 | photon overdensity squared at last scattering across k — the peaks, before projection |
| 4 | the computed TT spectrum |
| 5 | the same three modes as waves in space, over the comb across k, on a time slider |
| 6 | radial profile around one point overdensity — the sound shell and the BAO bump |

## Things that bit, and must not be undone

- **CAMB's `Weyl` is k²(Φ+Ψ)/2, not the potential.** `Weyl/k²` is flat at −2/3
  across k and time while superhorizon, which is the real potential's value
  there. Plotted raw the three modes start a factor of 11 apart and look like
  unrelated quantities. `gen_peaks.py` divides the k² out.
- **CAMB has not started a mode at η = 0.2 if k ≲ 0.08.** Normalising the point
  source in `gen_box.py` by `delta_cdm` at an early time therefore corrupts
  exactly the range carrying acoustic peaks 2–4, with ±0.4 ringing out to
  240 Mpc that looks plausible. The normalisation is analytic, C·k².
- **`sound_horizon(z)` keeps integrating past decoupling** and reaches 1223 Mpc
  today. It is clamped at last scattering; the acoustic scale is frozen there.
- **Panels 5a/5b freeze at η\*.** Past it the photons free-stream and damp, and
  a per-frame normalisation magnifies the remnant into full-scale noise.
- θ\* is held fixed across the whole grid, so H₀ moves instead (≈80 to ≈58).
  That is deliberate: it stops the peaks sliding sideways so the change in
  height is unambiguous.

## Accuracy notes

- Panel 5 at last scattering reproduces panel 3's peak positions to 0.6%; the
  time grid carries exact points at equality and at last scattering so the two
  can be compared directly.
- The measured peaks sit at k·r_s = (n − 0.13)π, not nπ: a constant acoustic
  phase shift from radiation driving and neutrino free-streaming. Consecutive
  spacings are 1.017, 0.991, 1.014 in units of π/r_s.
- The baryon shell settles at ≈149 Mpc, outside r\* = 144.5, because baryons are
  released at the drag epoch rather than at photon decoupling.
