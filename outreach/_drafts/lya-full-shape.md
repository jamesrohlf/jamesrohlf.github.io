# Draft: adding the DR2 Lyman-α full shape to the w₀–wₐ plane

**Not published.** Unlinked, absent from sitemap.xml, disallowed in robots.txt.
Note that this repository sets `.nojekyll`, so GitHub Pages serves every file
verbatim — this file is reachable by anyone who types its exact URL. It is
hidden, not private. Nothing secret belongs here.

## The change

Four numbers. In `w0wa.html`, the Lyman-α entry is the fifth element of each
BAO array. Replace the DR1 BAO values with the DR2 Lyman-α full-shape +
BAO combination at z = 2.33:

```js
//  index 4 is the Lyman-alpha point, z = 2.330
//  now  (DR1 BAO):            D_M/r_d = 39.71 +- 0.94 ,  D_H/r_d = 8.52  +- 0.17
//  swap (DR2 Lya full shape): D_M/r_d = 39.32 +- 0.33 ,  D_H/r_d = 8.600 +- 0.066
var DMo = [13.62, 16.85, 21.71, 27.79, 39.32], SMo = [0.25, 0.32, 0.28, 0.69, 0.33];
var DHo = [20.98, 20.08, 17.88, 13.82, 8.600], SHo = [0.61, 0.60, 0.35, 0.42, 0.066];
```

Source: the numbers quoted on `full-shape.html`, from the DR2 Lyman-α
full-shape paper — the AP effect measured from the full shape of the DR2
Lyman-α correlation functions rather than the BAO peak alone, combined with
the Lyman-α BAO result.

## What it does, measured

Run before building anything, against the twelve-point fit with r_d = 147.1:

| | ΛCDM Δχ² from best-in-box | unbounded minimum | 68% cells | 95% cells |
|---|---|---|---|---|
| DR1 BAO Lyα | 2.60 | w₀=+0.90, wₐ=−6.8 | 85/625 | 319/625 |
| DR2 Lyα full shape | 3.07 | w₀=+0.72, wₐ=−6.0 | 62/625 | 257/625 |

Three conclusions:

1. **The allowed region shrinks by about a quarter** (68% region −27%,
   95% −19%). This is the real, showable effect: a 2.6× tighter high-z
   anchor visibly tightens the constraint.
2. **It does not move toward ΛCDM.** ΛCDM goes from Δχ² = 2.60 to 3.07 —
   marginally *worse*. Small enough to mean little, but definitely not the
   "pulls back toward ΛCDM" story. It agrees in direction with
   `full-shape.html`, which has the w₀wₐCDM preference *rising* to 2.7σ.
3. **It does not cure the runaway.** With w₀ and wₐ free and only BAO-like
   distances to constrain them, the χ² valley still has no interior
   minimum; it runs to w₀ > 0. Only the CMB and supernovae close it.

## How to present it

A **toggle on the existing plane**, not a second plot: `DR1 BAO` /
`with DR2 Lyα full shape`. Two plots side by side invite the reader to
compare centres, which is the misleading comparison. One plane that visibly
tightens while the ΛCDM marker stays put shows precision improving without
implying the answer moved.

Implementation sketch: hold two grids, rebuild or precompute both at load
(each takes ~200 ms), and have the toggle swap which one `GRID` points at,
then re-run `draw()`. `chiMin` must be recomputed per dataset.

## Caveats that must appear on the page

- This mixes eleven DR1 BAO points with one DR2 measurement.
- The DR1 correlation coefficient ρ = −0.477 is reused for the Lyman-α pair,
  where a DR2 full-shape covariance belongs instead.

Fine for showing how precision propagates. Not a published constraint.
