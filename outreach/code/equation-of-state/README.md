# equation-of-state.html

From Newton to the equation of state. A uniform ball of dust and the shell
theorem give the Friedmann equation; the first law of thermodynamics gives the
fluid equation; together they give the acceleration equation, whose
ρ + 3P/c² term is the one thing Newton cannot supply. Then P = wρc², and the
universe's own w_eff(a) under ΛCDM or a DESI-like w₀wₐ.

## Build

    python3 assemble.py                                 # -> ../../equation-of-state.html

Edit the fragments, never the built page: an edit to `equation-of-state.html`
is silently lost on the next rebuild. There is no data file; everything is
computed in the browser.

## Fragments

| file | holds |
|---|---|
| `head.html` | `<head>`, page-specific CSS (shared styles come from `assets/css/site.css`) |
| `mid.html`  | `<main>`: the derivation, both panels' markup, sources |
| `js.html`   | both panels: single-component a(t) for a chosen w, and the real mixture |

## Numbers, and where they come from

- Planck 2018 VI: h = 0.674, Ω_m = 0.315. Ω_r h² = 4.18e-5 (photons plus three
  massless neutrinos), Ω_DE = 1 − Ω_m − Ω_r, flat.
- DESI preset: w₀ = −0.75, wₐ = −0.86, near the DR2 BAO + CMB + DESY5 best fit
  (arXiv:2503.14738). Ω_m is kept at the Planck value, so the preset shows the
  shape of the effect, not the joint best fit. The page says so.
- Cosmic time: ∫ d ln a / H on 4000 log-spaced steps from a = 1e-9, started at
  t = 1/(2H) (radiation era). ΛCDM gives 13.79 Gyr, matter–radiation equality
  at ~50 kyr (z ≈ 3420), acceleration from 6.1 Gyr ago (z ≈ 0.63).

## Things that must not be undone

- Display equations are MathJax: `<div class="eq">\[ ... \]</div>`, unnumbered. The three main ones carry a name label, `<span class="eqname">Friedmann equation</span>` before the `\[`, and the text refers to them by name. (A `\tag{name}` at the right margin made them too wide for a phone.)
  `assets/js/chrome.js` loads MathJax on any page whose `<main>` contains TeX,
  and `.eq` is styled in `site.css`. Don't add a MathJax `<script>` here.
- Inline math is MathJax too: time derivatives are `\dot a`, `\ddot a`, `\dot\rho`, never HTML entities.
