"""
desi_light_path_ltr.py
======================
Animated diagram of the full DESI light path, from galaxy to CCD —
horizontally flipped from desi_light_path_v2.py so that time flows
LEFT → RIGHT (galaxy on the left, CCDs on the right).

The scene is defined in the same data coordinates as desi_light_path_v2.py
(galaxy at high x, CCDs at low x); the axis is flipped at draw time via
ax.invert_xaxis(), which mirrors every patch, beam, and 45° dichroic
(a `/` mirror becomes `\\`, correctly reflecting the now rightward-
travelling optical beam downward toward the blue / red arms).

Stages animated in sequence:
  0. Galaxy emits light
  1. Mayall 4-metre telescope collects and focuses
  2. Focal plane — 5000 fiber positioners
  3. Optical fiber routes light to spectrograph room
  4. Collimator makes parallel beam
  5. Dichroic 1 reflects blue (360–593 nm), transmits red + NIR
  6. Dichroic 2 reflects red (566–772 nm), transmits NIR (747–980 nm)
  7. VPH grating disperses blue arm → CCD B
  8. VPH grating disperses red arm → CCD R
  9. VPH grating disperses NIR arm → CCD N
 10. All three CCDs illuminated simultaneously

Saves (via CLI flags):
  --save-gif   — writes desi_light_path_ltr.gif
  --save-png   — writes desi_light_path_ltr.png

Usage:
    python desi_light_path.py

Requirements:
    numpy, matplotlib
    (Optional: Pillow for GIF export — pip install Pillow)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Arc
import matplotlib.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT  (all coordinates in axes units 0–1 normalised to figure)
#  We use a single large Axes with no ticks, and place everything by hand.
# ─────────────────────────────────────────────────────────────────────────────

FIG_W, FIG_H = 16, 10
BG  = '#07070f'
FG  = '#ccccdd'

BLUE_COL  = '#6699ff'
RED_COL   = '#ff7722'       # LTR: shifted toward orange for visual pop
NIR_COL   = '#ee3322'       # LTR: brightened from '#661111' — proper red
WHITE_COL = '#ffffff'       # actually white
DIM_ALPHA = 0.18

# Stage durations in animation frames (at 25 fps → seconds = frames/25)
STAGE_FRAMES = [30, 35, 35, 45, 30, 40, 40, 40, 40, 40, 50]
STAGE_LABELS = [
    "Galaxy emits light across the visible spectrum",
    "Mayall 4-m telescope (Kitt Peak) collects and focuses light",
    "Focal plane: robotic fiber positioner captures the galaxy",
    "107-μm optical fiber routes light 50 m to spectrograph",
    "Collimator lens produces a parallel beam",
    "Dichroic 1 reflects blue (360–593 nm), transmits red+NIR",
    "Dichroic 2 reflects red (566–772 nm), transmits NIR (747–980 nm)",
    "VPH grating disperses blue arm — wavelength → CCD pixel",
    "VPH grating disperses red arm — [OII] 746 nm lands here",
    "VPH grating disperses NIR arm — Hα at high-z lands here",
    "All three CCDs record 500 fiber spectra simultaneously",
]
TOTAL_FRAMES = sum(STAGE_FRAMES)

# Global animation frame counter — set by draw_frame(); read by draw_beam /
# draw_fiber_beam so the sine wave keeps flowing after a stage completes.
_ANIM_FRAME = 0

# ─────────────────────────────────────────────────────────────────────────────
#  COMPONENT POSITIONS  (x, y in data coords;  axes xlim 0–16, ylim 0–10)
# ─────────────────────────────────────────────────────────────────────────────

# Galaxy — upper-right area (canvas is 16 × 10).  Kept a bit inside the
# canvas so the glow pulse and spiral arms don't get clipped.
GAL_XY   = (14.85, 9.15)
GAL_R    = 0.22
GAL_ARM_SCALE = 0.07

# Telescope primary mirror — tilted toward (but not directly at) the galaxy
TEL_XY   = (11.0, 5.5)
TEL_W, TEL_H = 2.0, 0.9
# angle of optical axis (from vertical), with a small CCW rotation offset
_dx, _dy = GAL_XY[0] - TEL_XY[0], GAL_XY[1] - TEL_XY[1]
TEL_TILT_CCW_DEG = 7                     # rotate telescope+FP ~7° CCW
TEL_TILT = np.arctan2(_dx, _dy) - np.radians(TEL_TILT_CCW_DEG)
_ax_x = np.sin(TEL_TILT); _ax_y = np.cos(TEL_TILT)   # unit vector along axis
_perp_x = _ax_y; _perp_y = -_ax_x                    # perpendicular to axis

# Focal plane (prime focus) — along optical axis, between mirror and galaxy
FP_DIST  = 2.6
FP_XY    = (TEL_XY[0] + _ax_x * FP_DIST, TEL_XY[1] + _ax_y * FP_DIST)
FP_R     = 0.5                             # radius of the little DESI disc
FP_TILT_SCALE = 0.40                       # 3D foreshortening (minor/major axis)

# Corrector lens — between the primary mirror and the focal plane, on the
# optical axis.  DESI's prime-focus corrector re-shapes the wavefront just
# before it lands on the fiber-positioner plate; we draw it as a thin lens
# perpendicular to the axis and bend the animated ray as it passes through.
CORR_FRAC  = 0.68                          # fraction of FP_DIST from mirror
CORR_XY    = (TEL_XY[0] + _ax_x * FP_DIST * CORR_FRAC,
              TEL_XY[1] + _ax_y * FP_DIST * CORR_FRAC)
CORR_HALFW = FP_R                          # same diameter as the focal plane
CORR_HALFT = 0.07                          # thickness along axis

# Fiber run (bezier control point) — now starts near the focal plane
# Spectrograph box
SPEC_BOX = (1.0, 1.2, 8.0, 8.0)  # x, y, w, h

# HORIZONTAL optical layout (light travels LEFT along y = OPTICAL_Y):
#   fiber → V-groove → collimator → D1 → D2 → red CCD
#   D1 reflects blue DOWN to blue arm
#   D2 reflects red DOWN to red arm
OPTICAL_Y = 6.6                     # spectrograph raised ~0.6 so the fiber
                                    # cable doesn't brush past the telescope

# V-groove block (right end, fiber enters from its RIGHT face)
VG_W, VG_H = 0.7, 1.0
VG_XY    = (8.4, OPTICAL_Y)
# Collimator lens — SAME width as V-groove; small gap between them
COL_W, COL_H = VG_W, VG_H
COL_XY   = (VG_XY[0] - VG_W - 0.35, OPTICAL_Y)   # 0.35-unit gap between the two

# 11 horizontal grooves inside the V-groove block — middle groove sits
# on the optical axis so the fibre → D1 beam is one straight horizontal line.
N_GROOVES = 11
GROOVE_YS = np.linspace(VG_XY[1] - VG_H/2 + 0.08,
                        VG_XY[1] + VG_H/2 - 0.08, N_GROOVES)
FIBER_GROOVE_IDX = N_GROOVES // 2                # middle groove (at OPTICAL_Y)
FIB_GROOVE_Y = GROOVE_YS[FIBER_GROOVE_IDX]

# Fiber run (bezier) — starts at the EDGE of the tilted DESI disc, ends
# aligned horizontally with the V-groove at the chosen groove height.
# Pick the rim point of the tilted ellipse in the direction of the V-groove.
_vg_right = VG_XY[0] + VG_W / 2
_vdx = _vg_right - FP_XY[0]
_vdy = FIB_GROOVE_Y - FP_XY[1]
# project V-groove direction into local (perp, ax) frame
_lp = _vdx * _perp_x + _vdy * _perp_y
_la = _vdx * _ax_x   + _vdy * _ax_y
# parametric angle on the tilted ellipse pointing that way; nudged so the
# fiber exits a little higher on the rim and doesn't clip the corrector lens
_t = np.arctan2(_la / FP_TILT_SCALE, _lp) - 0.45
_rx = FP_R * np.cos(_t)
_ry = FP_R * FP_TILT_SCALE * np.sin(_t)
FIB_EDGE_XY = (FP_XY[0] + _rx * _perp_x + _ry * _ax_x,
               FP_XY[1] + _rx * _perp_y + _ry * _ax_y)

FIB_P0   = FIB_EDGE_XY
FIB_P1   = (FIB_EDGE_XY[0] - 1.4, FIB_EDGE_XY[1] - 0.5)
FIB_P2   = (_vg_right + 1.2, FIB_GROOVE_Y)   # horizontal approach to V-groove
FIB_P3   = (_vg_right, FIB_GROOVE_Y)

# Dichroic 1 (blue split) — space between collimator and D1
D1_XY    = (COL_XY[0] - COL_W/2 - 1.05, OPTICAL_Y)
# Dichroic 2 (red/NIR split) — further left of D1, same y
D2_XY    = (D1_XY[0] - 1.55, OPTICAL_Y)

# --- arm geometry: equal grating→CCD distance for all three arms ---
ARM_LEN = 1.4                       # grating → CCD centre distance
# Blue arm — light reflects DOWN off D1  (gratings shifted up with OPTICAL_Y)
GB_XY    = (D1_XY[0], OPTICAL_Y - 2.0)
CB_XY    = (D1_XY[0], GB_XY[1] - ARM_LEN)
# Red arm — light reflects DOWN off D2
GR_XY    = (D2_XY[0], OPTICAL_Y - 2.0)
CR_XY    = (D2_XY[0], GR_XY[1] - ARM_LEN)
# NIR arm — light passes LEFT through D2, continuing along the optical axis
GN_XY    = (D2_XY[0] - 1.25, OPTICAL_Y)
CN_XY    = (GN_XY[0] - ARM_LEN, OPTICAL_Y)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)

def ease(t):
    """Smooth step."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def stage_progress(frame, stage_idx):
    """Return (local_frac 0→1) for given stage at given global frame."""
    start = sum(STAGE_FRAMES[:stage_idx])
    end   = start + STAGE_FRAMES[stage_idx]
    if frame < start:  return 0.0
    if frame >= end:   return 1.0
    return (frame - start) / STAGE_FRAMES[stage_idx]

def is_active(frame, stage_idx):
    start = sum(STAGE_FRAMES[:stage_idx])
    end   = start + STAGE_FRAMES[stage_idx]
    return start <= frame < end

def is_done(frame, stage_idx):
    return frame >= sum(STAGE_FRAMES[:stage_idx + 1])

def beam_alpha(frame, stage_idx):
    """Alpha for a beam: fades in during its stage, stays on after."""
    p = stage_progress(frame, stage_idx)
    if is_done(frame, stage_idx): return 0.85
    return ease(p) * 0.85

def particle_pos(p0, p1, t):
    """Linear interpolation along a beam, t ∈ [0,1]."""
    return (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t))

def bezier3(p0, p1, p2, p3, t):
    """Cubic bezier point."""
    t = np.clip(t, 0, 1)
    mt = 1 - t
    x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
    y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
    return x, y

def draw_component_box(ax, xy, w, h, color, label, label_below=True, alpha=1.0):
    """Draw a filled rounded box with label."""
    from matplotlib.patches import FancyBboxPatch
    fc = (*plt.matplotlib.colors.to_rgb(color), alpha * 0.22)
    ec = (*plt.matplotlib.colors.to_rgb(color), alpha * 0.85)
    box = FancyBboxPatch((xy[0]-w/2, xy[1]-h/2), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=fc, edgecolor=ec, linewidth=1.2,
                          zorder=3)
    ax.add_patch(box)
    ty = xy[1] - h/2 - 0.18 if label_below else xy[1] + h/2 + 0.18
    ax.text(xy[0], ty, label, ha='center', va='top' if label_below else 'bottom',
            fontsize=7.5, color=(*plt.matplotlib.colors.to_rgb(color), alpha*0.9),
            fontfamily='monospace', zorder=4)

def _draw_camera_lens(ax, xy, half_w, half_h, color, alpha=1.0, horizontal=False):
    """Draw a small lens element (two converging arcs).
       horizontal=True  → lens axis is horizontal (beam enters the side face).
       horizontal=False → lens axis is vertical (beam enters the top face)."""
    cx, cy = xy
    rgb = plt.matplotlib.colors.to_rgb(color)
    fc = (*rgb, alpha * 0.25)
    ec = (*rgb, alpha * 0.9)
    # bounding body
    from matplotlib.patches import FancyBboxPatch
    body = FancyBboxPatch((cx - half_w, cy - half_h),
                          2 * half_w, 2 * half_h,
                          boxstyle="round,pad=0.02",
                          facecolor=fc, edgecolor=ec, lw=0.8, zorder=3)
    ax.add_patch(body)
    # two converging arcs indicating a lens
    t = np.linspace(-np.pi * 0.5, np.pi * 0.5, 30)
    if horizontal:
        for side in (-1, +1):
            lx = cx + side * half_w * 0.7 * np.cos(t) * 0.8
            ly = cy + (half_h - 0.02) * np.sin(t)
            ax.plot(lx, ly, color=(*rgb, alpha * 0.75), lw=0.9, zorder=4)
    else:
        for side in (-1, +1):
            lx = cx + (half_w - 0.02) * np.sin(t)
            ly = cy + side * half_h * 0.7 * np.cos(t) * 0.8
            ax.plot(lx, ly, color=(*rgb, alpha * 0.75), lw=0.9, zorder=4)


def draw_dichroic(ax, xy, color, label, alpha=1.0):
    """Draw a 45° dichroic mirror line."""
    cx, cy = xy
    sz = 0.55
    lc = (*plt.matplotlib.colors.to_rgb(color), alpha*0.9)
    ax.plot([cx - sz*0.7, cx + sz*0.7], [cy - sz*0.7, cy + sz*0.7],
            color=lc, linewidth=3, solid_capstyle='round', zorder=4)
    ax.plot([cx - sz*0.7, cx + sz*0.7], [cy - sz*0.7, cy + sz*0.7],
            color=(*plt.matplotlib.colors.to_rgb(color), alpha*0.25),
            linewidth=6, solid_capstyle='round', zorder=3)
    label_c = (*plt.matplotlib.colors.to_rgb(color), min(alpha * 1.0, 1.0))
    ax.text(cx, cy + 0.55, label, fontsize=11, color=label_c,
            fontfamily='monospace', ha='center', va='bottom', zorder=5)

def draw_beam(ax, p0, p1, color, alpha, frame_t, n_parts=6, lw=2.0):
    """Draw animated beam as a propagating sine wave from p0 toward p1."""
    p0v = np.asarray(p0, dtype=float)
    p1v = np.asarray(p1, dtype=float)
    d = p1v - p0v
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-9:
        return
    u = d / L
    perp = np.array([-u[1], u[0]])

    end_L = L * float(np.clip(frame_t, 0.0, 1.0))
    if end_L < 1e-4:
        return

    amp = 0.055
    wavelength = 0.28
    k = 2 * np.pi / wavelength
    # phase drifts with the global animation frame so the wave "flows"
    phase = _ANIM_FRAME * 0.35

    n = max(60, int(end_L / 0.012))
    s = np.linspace(0.0, end_L, n)
    edge = min(0.18, end_L * 0.30)
    taper = np.ones_like(s)
    if edge > 1e-6:
        near_start = s < edge
        taper[near_start] = s[near_start] / edge
        near_end = s > (end_L - edge)
        taper[near_end] = (end_L - s[near_end]) / edge

    wave = amp * taper * np.sin(k * s - phase)
    pts = p0v[None, :] + s[:, None] * u[None, :] + wave[:, None] * perp[None, :]
    ax.plot(pts[:, 0], pts[:, 1],
            color=color, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=5)

def draw_fiber_cladding(ax, p0, p1, p2, p3, alpha=0.9):
    """Draw the two thin cladding walls along the full fiber bezier (always
       visible — static piece of apparatus, not animated)."""
    n = 120
    ts_full = np.linspace(0, 1, n)
    pts = np.array([bezier3(p0, p1, p2, p3, t) for t in ts_full])
    d = np.zeros_like(pts)
    d[:-1] = pts[1:] - pts[:-1]
    d[-1]  = d[-2]
    tn = np.linalg.norm(d, axis=1)[:, None]
    d  = d / (tn + 1e-12)
    perp = np.stack([-d[:, 1], d[:, 0]], axis=1)
    FIBER_R = 0.08
    left  = pts + perp * FIBER_R
    right = pts - perp * FIBER_R
    wall_col = (0.85, 0.85, 0.95, alpha * 0.65)
    ax.plot(left[:, 0],  left[:, 1],  color=wall_col, lw=0.9, zorder=4)
    ax.plot(right[:, 0], right[:, 1], color=wall_col, lw=0.9, zorder=4)


def draw_fiber_beam(ax, p0, p1, p2, p3, color, alpha, frame_t, n_parts=8):
    """Animated light inside the fiber, drawn as a sine wave along the bezier
       whose amplitude is bounded by the cladding radius."""
    frame_t = float(np.clip(frame_t, 0.0, 1.0))
    if frame_t < 1e-4:
        return

    n = 220
    ts_full = np.linspace(0.0, frame_t, n)
    pts = np.array([bezier3(p0, p1, p2, p3, ti) for ti in ts_full])

    # tangent + perpendicular at each sample
    d = np.zeros_like(pts)
    d[:-1] = pts[1:] - pts[:-1]
    d[-1] = d[-2] if n > 1 else np.array([1.0, 0.0])
    tn = np.linalg.norm(d, axis=1)[:, None]
    d = d / (tn + 1e-12)
    perp = np.stack([-d[:, 1], d[:, 0]], axis=1)

    # arc length along the sampled bezier
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1] if s[-1] > 0 else 1e-9

    amp = 0.035                          # a bit smaller than fiber core radius
    wavelength = 0.20
    k = 2 * np.pi / wavelength
    phase = _ANIM_FRAME * 0.35

    edge = min(0.18, total * 0.25)
    taper = np.ones_like(s)
    if edge > 1e-6:
        near_start = s < edge
        taper[near_start] = s[near_start] / edge
        near_end = s > (total - edge)
        taper[near_end] = (total - s[near_end]) / edge

    wave = amp * taper * np.sin(k * s - phase)
    pts_wave = pts + wave[:, None] * perp
    ax.plot(pts_wave[:, 0], pts_wave[:, 1],
            color=color, lw=1.8, alpha=alpha,
            solid_capstyle='round', zorder=5)

def draw_spectrum_fan(ax, ccd_xy, direction, alpha):
    """Draw fiber-spectrum strips on a CCD.
       blue / red — 10 horizontal strips, wavelength gradient runs across each
                    strip (horizontal).
       nir        — colors rotated 90°: each strip is a solid wavelength band,
                    color changing from strip to strip (vertical)."""
    from matplotlib.colors import LinearSegmentedColormap

    colors_b = ['#3311aa','#3355dd','#3399ff','#22ccff','#00ffff']
    # LTR: red arm gradient shifted toward orange, NIR gradient brightened
    colors_r = ['#ffee44','#ffbb33','#ff8822','#ff5511','#ee3300']
    colors_n = ['#ff6655','#ee4433','#cc2211','#aa1100','#661100']
    cols = {'blue': colors_b, 'red': colors_r, 'nir': colors_n}[direction]
    cmap = LinearSegmentedColormap.from_list(f'spec_{direction}', cols)

    cx, cy = ccd_xy
    N_STRIPS = 10
    strip_h = 0.06
    gap_h = (0.80 - N_STRIPS * strip_h) / (N_STRIPS - 1)   # fills CCD height
    step = strip_h + gap_h
    x_half = 0.40
    y_bot = cy - 0.40

    if direction == 'nir':
        for i in range(N_STRIPS):
            y0 = y_bot + i * step
            r_, g_, b_, _ = cmap(i / (N_STRIPS - 1))
            fc = (r_, g_, b_, alpha * 0.85)
            ax.add_patch(plt.Rectangle((cx - x_half, y0),
                                       2 * x_half, strip_h,
                                       facecolor=fc, edgecolor='none',
                                       zorder=6))
    else:
        # LTR: draw each strip as a row of discrete solid-color Rectangles.
        # imshow (even with interpolation='nearest') suffers when the strip's
        # data extent doesn't land on integer pixel boundaries — bins render
        # slightly-different widths and GIF palette quantisation compounds
        # the effect into visibly non-uniform stripes. Discrete rectangles
        # each take a fixed data-extent, so aliasing is consistent across
        # bins and every strip renders identically.
        N_BINS = 10
        bin_w = 2 * x_half / N_BINS
        for i in range(N_STRIPS):
            y0 = y_bot + i * step
            for j in range(N_BINS):
                rr, gg, bb, _ = cmap(j / (N_BINS - 1))
                ax.add_patch(plt.Rectangle(
                    (cx - x_half + j * bin_w, y0),
                    bin_w, strip_h,
                    facecolor=(rr, gg, bb, alpha * 0.85),
                    edgecolor='none', zorder=6, snap=True))


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW FUNCTION  (called per frame)
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame(frame, ax, title_ax):
    global _ANIM_FRAME
    _ANIM_FRAME = frame
    ax.cla()
    title_ax.cla()

    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.set_aspect('equal'); ax.axis('off')
    ax.invert_xaxis()   # LTR variant: mirror the whole scene horizontally
    ax.set_facecolor(BG)
    title_ax.axis('off'); title_ax.set_facecolor(BG)

    # ── Current stage ──────────────────────────────────────────────────────
    stage = 0
    elapsed = 0
    for i, dur in enumerate(STAGE_FRAMES):
        if frame < elapsed + dur:
            stage = i
            break
        elapsed += dur
    else:
        stage = len(STAGE_FRAMES) - 1

    local_frac = stage_progress(frame, stage)
    ef = ease(local_frac)

    # ── Credit + title ──────────────────────────────────────────────────────
    title_ax.text(0.5, 0.95, "J. Rohlf and Claude Opus 4.7 (2026)",
                  ha='center', va='top', fontsize=14, color='#9aabcc',
                  fontfamily='monospace', transform=title_ax.transAxes)
    title_ax.text(0.5, 0.30, "Light path through DESI",
                  ha='center', va='center', fontsize=30, color=FG,
                  fontfamily='monospace', fontweight='bold',
                  transform=title_ax.transAxes)

    # alpha helper: full if done, fading in if active, dim if future
    def ca(s):
        # render every piece of apparatus at full alpha from frame 0 —
        # only the light beams are animated
        return 0.9

    # ── Galaxy — golden core + blue spiral arms ────────────────────────────
    gal_a = ca(0)
    _core_rgb = (1.00, 0.78, 0.35)          # warm gold (old stars)
    _arm_rgb  = (0.55, 0.75, 1.00)          # pale blue (young stars)
    circle = plt.Circle(GAL_XY, GAL_R,
                         facecolor=(*_core_rgb, gal_a*0.55),
                         edgecolor=(*_core_rgb, gal_a*0.9),
                         linewidth=1.2, zorder=3)
    ax.add_patch(circle)
    for arm in range(2):
        ts = np.linspace(0, 3*np.pi, 80)
        rs = ts * GAL_ARM_SCALE
        xs = GAL_XY[0] + rs * np.cos(ts + arm*np.pi)
        ys = GAL_XY[1] + rs * np.sin(ts + arm*np.pi)
        ax.plot(xs, ys, color=(*_arm_rgb, gal_a*0.75), lw=0.8, zorder=3)

    # ── Telescope primary mirror — shallow 3D dish, tilted toward galaxy ─
    tel_a = ca(1)
    from matplotlib.patches import FancyBboxPatch, Wedge
    TEL_DISH_SCALE = 0.28               # foreshortening of the dish (shallow)
    _mirror_angle_deg = np.degrees(np.arctan2(-_ax_x, _ax_y))

    # 1. Shadow behind the dish (offset backward along -_ax), suggests depth
    _sh_off = 0.12
    _sh_xy = (TEL_XY[0] - _ax_x * _sh_off, TEL_XY[1] - _ax_y * _sh_off)
    ax.add_patch(mpatches.Ellipse(
        _sh_xy, width=TEL_W * 0.98, height=TEL_W * 0.98 * TEL_DISH_SCALE,
        angle=_mirror_angle_deg,
        facecolor=(0.08, 0.08, 0.14, tel_a * 0.65),
        edgecolor='none', zorder=2))

    # 2. Main reflective dish (tilted ellipse, no edge — rim is drawn as
    #    two separate arcs below so the FRONT is bright and the BACK is dim)
    ax.add_patch(mpatches.Ellipse(
        TEL_XY, width=TEL_W, height=TEL_W * TEL_DISH_SCALE,
        angle=_mirror_angle_deg,
        facecolor=(0.45, 0.50, 0.62, tel_a * 0.55),
        edgecolor='none', zorder=3))
    # 2b. Rim drawn as two SOLID arcs:
    #     bottom (closer to viewer) — full brightness
    #     top    (farther away)     — dimmer
    _near_ts = np.linspace(np.pi, 2 * np.pi, 50)
    _nrx = (TEL_W / 2) * np.cos(_near_ts)
    _nry = (TEL_W / 2 * TEL_DISH_SCALE) * np.sin(_near_ts)
    _nx = TEL_XY[0] + _nrx * _perp_x + _nry * _ax_x
    _ny = TEL_XY[1] + _nrx * _perp_y + _nry * _ax_y
    ax.plot(_nx, _ny, color=(0.92, 0.95, 1.0, tel_a * 0.95),
            lw=1.8, solid_capstyle='round', zorder=3.3)
    _far_ts = np.linspace(0, np.pi, 50)
    _frx = (TEL_W / 2) * np.cos(_far_ts)
    _fry = (TEL_W / 2 * TEL_DISH_SCALE) * np.sin(_far_ts)
    _fx = TEL_XY[0] + _frx * _perp_x + _fry * _ax_x
    _fy = TEL_XY[1] + _frx * _perp_y + _fry * _ax_y
    ax.plot(_fx, _fy, color=(0.72, 0.78, 0.92, tel_a * 0.55),
            lw=1.4, solid_capstyle='round', zorder=3.2)

    # 3. Upper-interior shadow crescent (dark fill) — concavity cue
    _sh_ts = np.linspace(0, np.pi, 50)
    _shlx = (TEL_W / 2 * 0.96) * np.cos(_sh_ts)
    _shly = (TEL_W / 2 * TEL_DISH_SCALE * 0.96) * np.sin(_sh_ts)
    # closing curve at ~20% height so the shadow is a thin arc at the top rim
    _clx = (TEL_W / 2 * 0.96) * np.cos(_sh_ts[::-1])
    _cly = (TEL_W / 2 * TEL_DISH_SCALE * 0.30) * np.sin(_sh_ts[::-1])
    _crescent_lx = np.concatenate([_shlx, _clx])
    _crescent_ly = np.concatenate([_shly, _cly])
    _cx = TEL_XY[0] + _crescent_lx * _perp_x + _crescent_ly * _ax_x
    _cy = TEL_XY[1] + _crescent_lx * _perp_y + _crescent_ly * _ax_y
    ax.add_patch(mpatches.Polygon(
        np.column_stack([_cx, _cy]),
        facecolor=(0.08, 0.10, 0.18, tel_a * 0.45),
        edgecolor='none', zorder=3.5))

    # 4. Concentric rings inside — solid faint lines
    for _frac in (0.72, 0.45, 0.22):
        ax.add_patch(mpatches.Ellipse(
            TEL_XY, width=TEL_W * _frac,
            height=TEL_W * _frac * TEL_DISH_SCALE,
            angle=_mirror_angle_deg,
            facecolor='none',
            edgecolor=(0.80, 0.86, 1.0, tel_a * 0.45),
            lw=0.7, zorder=4))

    # 5. Bright highlight arc on the near (viewer-facing, bottom) rim
    _arc_ts = np.linspace(np.pi * 1.15, np.pi * 1.85, 40)
    _hlx = (TEL_W / 2 * 0.98) * np.cos(_arc_ts)
    _hly = (TEL_W / 2 * TEL_DISH_SCALE * 0.98) * np.sin(_arc_ts)
    _hx = TEL_XY[0] + _hlx * _perp_x + _hly * _ax_x
    _hy = TEL_XY[1] + _hlx * _perp_y + _hly * _ax_y
    ax.plot(_hx, _hy, color=(0.97, 0.98, 1.0, tel_a * 0.9),
            lw=1.8, zorder=5)
    # LTR: hero call-out CENTRED under the mirror — extra-large, bold.
    lb_x = TEL_XY[0]
    lb_y = TEL_XY[1] - TEL_W * TEL_DISH_SCALE / 2 - 0.70
    ax.text(lb_x, lb_y, "Mayall 4-m",
            ha='center', va='top', fontsize=17, fontweight='bold',
            color=(0.92, 0.95, 1.0, tel_a*1.0),
            fontfamily='monospace', zorder=4)

    # ── Focal plane — drawn as a DESI disc with 10 petal spokes ───────────
    fp_a = ca(2)
    # background disc
    # 3D tilt: compress the disc along the optical-axis direction so it
    # looks like a plate viewed slightly from the side (FP_TILT_SCALE is
    # defined at module scope).
    # ellipse in world coords, major along _perp = (_ax_y, -_ax_x),
    # minor along _ax.  matplotlib Ellipse rotates by `angle` (deg).
    _fp_angle_deg = np.degrees(np.arctan2(-_ax_x, _ax_y))
    disc = mpatches.Ellipse(FP_XY,
                            width=2 * FP_R,
                            height=2 * FP_R * FP_TILT_SCALE,
                            angle=_fp_angle_deg,
                            facecolor=(0.3, 0.4, 0.85, fp_a * 0.18),
                            edgecolor=(0.6, 0.75, 1.0, fp_a * 0.9),
                            lw=1.2, zorder=3)
    ax.add_patch(disc)
    # 10 petal spokes — foreshortened along the axis
    for k in range(10):
        local_a = k * 2 * np.pi / 10
        lx = FP_R * np.sin(local_a)                       # along _perp
        ly = FP_R * np.cos(local_a) * FP_TILT_SCALE       # along _ax
        x1 = FP_XY[0] + lx * _perp_x + ly * _ax_x
        y1 = FP_XY[1] + lx * _perp_y + ly * _ax_y
        ax.plot([FP_XY[0], x1], [FP_XY[1], y1],
                color=(0.55, 0.7, 1.0, fp_a * 0.55), lw=0.6, zorder=4)
    # a few positioner dots inside (same foreshortening)
    for fi in range(12):
        r = 0.12 + (fi % 3) * 0.15
        local_a = fi * (2 * np.pi / 12)
        lx = r * np.sin(local_a)
        ly = r * np.cos(local_a) * FP_TILT_SCALE
        fx = FP_XY[0] + lx * _perp_x + ly * _ax_x
        fy = FP_XY[1] + lx * _perp_y + ly * _ax_y
        ax.plot(fx, fy, 'o', ms=2.0,
                color=(0.6, 0.8, 1.0, fp_a * 0.85), zorder=5)
    # label above the disc (facing the galaxy)
    ax.text(FP_XY[0], FP_XY[1] + FP_R + 0.02,
            "focal plane\n(5000 fibers)",
            ha='center', fontsize=10.5, color=(0.80,0.88,1.0,fp_a*1.0),
            fontfamily='monospace', zorder=4, va='bottom')
    # fiber-exit cable boot at the EDGE of the focal-plane disc
    _boot_inner = (FIB_EDGE_XY[0] + (FP_XY[0] - FIB_EDGE_XY[0]) * 0.25,
                   FIB_EDGE_XY[1] + (FP_XY[1] - FIB_EDGE_XY[1]) * 0.25)
    ax.plot([_boot_inner[0], FIB_EDGE_XY[0]],
            [_boot_inner[1], FIB_EDGE_XY[1]],
            color=(0.15, 0.18, 0.25, fp_a * 0.95),
            lw=6, solid_capstyle='round', zorder=6)

    # ── Corrector lens — between primary mirror and focal plane ───────────
    # drawn as a thin rotated lens (aligned perpendicular to the optical axis)
    corr_alpha = 0.9
    _corr_rgb = (0.55, 0.75, 1.0)
    # local frame: half_w along perp (_ax_y, -_ax_x), half_t along _ax
    _px, _py = _ax_y, -_ax_x
    # rectangular body
    _corners_local = np.array([
        [-CORR_HALFT, -CORR_HALFW],
        [ CORR_HALFT, -CORR_HALFW],
        [ CORR_HALFT,  CORR_HALFW],
        [-CORR_HALFT,  CORR_HALFW],
    ])
    _corners = np.array([
        (CORR_XY[0] + lx * _ax_x + ly * _px,
         CORR_XY[1] + lx * _ax_y + ly * _py) for lx, ly in _corners_local
    ])
    ax.add_patch(mpatches.Polygon(
        _corners,
        facecolor=(*_corr_rgb, corr_alpha * 0.18),
        edgecolor=(*_corr_rgb, corr_alpha * 0.9),
        lw=1.0, zorder=3))
    # lens curvature arcs (two converging arcs along the perp axis)
    _arc_ts = np.linspace(-np.pi/2, np.pi/2, 30)
    for side in (-1, +1):
        lx = side * (CORR_HALFT - 0.01) * np.cos(_arc_ts) * 0.7
        ly = (CORR_HALFW - 0.04) * np.sin(_arc_ts)
        xs = CORR_XY[0] + lx * _ax_x + ly * _px
        ys = CORR_XY[1] + lx * _ax_y + ly * _py
        ax.plot(xs, ys, color=(*_corr_rgb, corr_alpha * 0.7), lw=0.9, zorder=4)
    # label — horizontal, placed just below-and-to-the-left of the lens
    ax.text(CORR_XY[0] - 0.55, CORR_XY[1] - 0.05, "corrector",
            ha='center', va='top',
            fontsize=10.5, color=(*_corr_rgb, corr_alpha * 1.0),
            fontfamily='monospace', zorder=4)

    # ── Fiber cladding (always visible, light flows inside it later) ──────
    draw_fiber_cladding(ax, FIB_P0, FIB_P1, FIB_P2, FIB_P3, alpha=0.9)
    # fiber label — placed above the fiber at its midpoint
    _bx, _by = bezier3(FIB_P0, FIB_P1, FIB_P2, FIB_P3, 0.5)
    ax.text(_bx, _by + 0.55, "optical fiber\n(107 μm core)",
            ha='center', va='bottom', fontsize=10.5,
            color=(1.0, 0.94, 0.70, 1.0),
            fontfamily='monospace', zorder=5)

    # ── Spectrograph enclosure ─────────────────────────────────────────────
    spec_a = max(ca(4), 0.12)
    sx0, sy0, sw, sh = SPEC_BOX
    spec_rect = plt.Rectangle((sx0, sy0), sw, sh,
                               facecolor='none',
                               edgecolor=(0.35,0.35,0.55,spec_a*0.35),
                               linewidth=0.7, zorder=2)
    ax.add_patch(spec_rect)
    # LTR: hero call-out above the D1/D2 dichroics — same size as Mayall
    _spec_lbl_x = (D1_XY[0] + D2_XY[0]) / 2
    ax.text(_spec_lbl_x, OPTICAL_Y + 1.15, "spectrograph",
            ha='center', va='bottom',
            fontsize=17, fontweight='bold',
            color=(0.85, 0.85, 0.95, spec_a * 1.0),
            fontfamily='monospace', zorder=3)

    # ── V-groove block  (top) ─────────────────────────────────────────────
    col_a = ca(4)
    vg_box = FancyBboxPatch((VG_XY[0]-VG_W/2, VG_XY[1]-VG_H/2),
                             VG_W, VG_H, boxstyle="round,pad=0.02",
                             facecolor=(0.15, 0.18, 0.25, col_a*0.55),
                             edgecolor=(0.55, 0.65, 0.85, col_a*0.85),
                             lw=1.0, zorder=3)
    ax.add_patch(vg_box)
    for gy in GROOVE_YS:
        ax.plot([VG_XY[0]-VG_W/2+0.04, VG_XY[0]+VG_W/2-0.04],
                [gy, gy],
                color=(0.55, 0.7, 1.0, col_a*0.5), lw=0.55, zorder=4)
    ax.text(VG_XY[0], VG_XY[1] + VG_H/2 + 0.02,
            "V-groove\nblock",
            ha='center', va='bottom',
            fontsize=10.5, color=(0.80,0.88,1.0, col_a*1.0),
            fontfamily='monospace', zorder=4)

    # ── Collimator lens  (bottom, SAME WIDTH & COLOR as V-groove) ─────────
    col_box = FancyBboxPatch((COL_XY[0]-COL_W/2, COL_XY[1]-COL_H/2),
                              COL_W, COL_H, boxstyle="round,pad=0.04",
                              facecolor=(0.15, 0.18, 0.25, col_a*0.55),
                              edgecolor=(0.55, 0.65, 0.85, col_a*0.85),
                              lw=1.0, zorder=3)
    ax.add_patch(col_box)
    # lens curvature marks: two converging arcs inside the collimator box
    lens_ts = np.linspace(-np.pi*0.5, np.pi*0.5, 40)
    for side in (-1, +1):
        lx = COL_XY[0] + side * (COL_W/2 - 0.08) * np.cos(lens_ts) * 0.9
        ly = COL_XY[1] + (COL_H/2 - 0.08) * np.sin(lens_ts)
        ax.plot(lx, ly, color=(0.55, 0.7, 1.0, col_a*0.6), lw=0.9, zorder=4)
    ax.text(COL_XY[0], COL_XY[1] + COL_H/2 + 0.12, "collimator",
            ha='center', fontsize=11, color=(0.80, 0.88, 1.0, col_a*1.0),
            fontfamily='monospace', va='bottom', zorder=4)

    # ── Dichroics ──────────────────────────────────────────────────────────
    draw_dichroic(ax, D1_XY, '#88aaff', 'D1', alpha=ca(5))
    draw_dichroic(ax, D2_XY, '#ffaa55', 'D2', alpha=ca(6))

    # unified square CCDs
    CCD_SIDE = 0.8
    CCD_W = CCD_H = CCD_SIDE
    # grating "thick line" half-length in the direction perpendicular to the beam
    GR_HALF = 0.55
    # fraction of grating→CCD distance at which to place the camera lens
    LENS_FRAC = 0.30

    # ── Blue arm: grating (thick line) + lens + CCD ───────────────────────
    blue_alpha = ca(7)
    _rgb = plt.matplotlib.colors.to_rgb(BLUE_COL)
    ax.plot([GB_XY[0] - GR_HALF, GB_XY[0] + GR_HALF],
            [GB_XY[1], GB_XY[1]],
            color=(*_rgb, blue_alpha * 0.95), lw=6,
            solid_capstyle='round', zorder=4)
    ax.text(GB_XY[0], GB_XY[1] + 0.20, "grating B",
            ha='center', fontsize=11,
            color=(*_rgb, blue_alpha * 1.0),
            fontfamily='monospace', zorder=4)
    lens_y = GB_XY[1] + LENS_FRAC * (CB_XY[1] - GB_XY[1])
    _draw_camera_lens(ax, (GB_XY[0], lens_y),
                      half_w=0.38, half_h=0.10,
                      color=BLUE_COL, alpha=blue_alpha, horizontal=False)
    _rgb = plt.matplotlib.colors.to_rgb(BLUE_COL)
    ax.text(CB_XY[0], CB_XY[1] - CCD_H/2 - 0.20, "CCD  B",
            ha='center', va='top', fontsize=11,
            color=(*_rgb, blue_alpha * 1.0),
            fontfamily='monospace', zorder=4)
    ax.text(CB_XY[0], CB_XY[1] - CCD_H/2 - 0.50, "360–593 nm",
            ha='center', va='top', fontsize=9.5,
            color=(*_rgb, blue_alpha * 1.0),
            fontfamily='monospace', zorder=4)

    # ── Red arm: grating (thick line) + lens + CCD — DOWN off D2 ──────────
    red_alpha = ca(8)
    _rgb = plt.matplotlib.colors.to_rgb(RED_COL)
    ax.plot([GR_XY[0] - GR_HALF, GR_XY[0] + GR_HALF],
            [GR_XY[1], GR_XY[1]],
            color=(*_rgb, red_alpha * 0.95), lw=6,
            solid_capstyle='round', zorder=4)
    ax.text(GR_XY[0], GR_XY[1] + 0.20, "grating R",
            ha='center', fontsize=11,
            color=(*_rgb, red_alpha * 1.0),
            fontfamily='monospace', zorder=4)
    lens_y = GR_XY[1] + LENS_FRAC * (CR_XY[1] - GR_XY[1])
    _draw_camera_lens(ax, (GR_XY[0], lens_y),
                      half_w=0.38, half_h=0.10,
                      color=RED_COL, alpha=red_alpha, horizontal=False)
    _rgb = plt.matplotlib.colors.to_rgb(RED_COL)
    ax.text(CR_XY[0], CR_XY[1] - CCD_H/2 - 0.20, "CCD  R",
            ha='center', va='top', fontsize=11,
            color=(*_rgb, red_alpha * 1.0),
            fontfamily='monospace', zorder=4)
    ax.text(CR_XY[0], CR_XY[1] - CCD_H/2 - 0.50, "566–772 nm",
            ha='center', va='top', fontsize=9.5,
            color=(*_rgb, red_alpha * 1.0),
            fontfamily='monospace', zorder=4)

    # ── NIR arm: grating (thick line) + lens + CCD — LEFT along axis ──────
    nir_alpha = ca(9)
    _rgb = plt.matplotlib.colors.to_rgb(NIR_COL)
    ax.plot([GN_XY[0], GN_XY[0]],
            [GN_XY[1] - GR_HALF, GN_XY[1] + GR_HALF],
            color=(*_rgb, nir_alpha * 0.95), lw=6,
            solid_capstyle='round', zorder=4)
    _nir_lbl = plt.matplotlib.colors.to_rgb('#ff8866')      # NIR labels — slightly brighter than the bar
    ax.text(GN_XY[0], GN_XY[1] + GR_HALF + 0.18, "grating NIR",
            ha='center', fontsize=11,
            color=(*_nir_lbl, nir_alpha * 1.0),
            fontfamily='monospace', zorder=4)
    lens_x = GN_XY[0] + LENS_FRAC * (CN_XY[0] - GN_XY[0])
    _draw_camera_lens(ax, (lens_x, GN_XY[1]),
                      half_w=0.10, half_h=0.38,
                      color=NIR_COL, alpha=nir_alpha, horizontal=True)
    _rgb = plt.matplotlib.colors.to_rgb(NIR_COL)
    ax.text(CN_XY[0], CN_XY[1] - CCD_H/2 - 0.20, "CCD  NIR",
            ha='center', va='top', fontsize=11,
            color=(*_nir_lbl, nir_alpha * 1.0),
            fontfamily='monospace', zorder=4)
    ax.text(CN_XY[0], CN_XY[1] - CCD_H/2 - 0.50, "747–980 nm",
            ha='center', va='top', fontsize=9.5,
            color=(*_nir_lbl, nir_alpha * 1.0),
            fontfamily='monospace', zorder=4)

    # ── ANIMATED BEAMS ─────────────────────────────────────────────────────

    # Stage 0: galaxy glow pulse
    if stage == 0:
        pulse = 0.4 + 0.3*np.sin(frame * 0.4)
        glow = plt.Circle(GAL_XY, GAL_R + pulse,
                           facecolor='none',
                           edgecolor=(1.0, 0.85, 0.55, ef*0.4),
                           lw=1.0, zorder=4)
        ax.add_patch(glow)

    # Stage 1: galaxy CENTRE → primary mirror EDGE (clears the focal plane disc)
    _offset = FP_R + 0.25
    # mirror-edge hit point — along-axis depth computed from the bowl's
    # surface at this perpendicular offset so the beam actually TOUCHES
    # the mirror (bowl: lx = TEL_W/2*cos(ts), ly = 0.35 + TEL_H/2*sin(ts))
    _ratio = _offset / (TEL_W / 2)
    _ratio = max(-1.0, min(1.0, _ratio))
    _sin_ts = -np.sqrt(max(0.0, 1.0 - _ratio ** 2))
    _mirror_depth = 0.35 + (TEL_H / 2) * _sin_ts
    _mx_hit = TEL_XY[0] + _ax_x * _mirror_depth + _perp_x * _offset
    _my_hit = TEL_XY[1] + _ax_y * _mirror_depth + _perp_y * _offset
    # unit vector from galaxy centre toward that hit point
    _vdx = _mx_hit - GAL_XY[0]; _vdy = _my_hit - GAL_XY[1]
    _vln = np.hypot(_vdx, _vdy)
    _ux, _uy = _vdx / _vln, _vdy / _vln
    if frame >= sum(STAGE_FRAMES[:1]):
        p = stage_progress(frame, 1)
        # start at the galaxy edge along that line (so galaxy centre is on the beam)
        gx = GAL_XY[0] + _ux * (GAL_R + 0.03)
        gy = GAL_XY[1] + _uy * (GAL_R + 0.03)
        draw_beam(ax, (gx, gy), (_mx_hit, _my_hit),
                  WHITE_COL, beam_alpha(frame, 1), p, n_parts=5)

    # Stage 2: primary mirror EDGE → corrector lens → focal plane
    # The ray hits the corrector off-axis then BENDS toward the FP centre
    # (mimicking the refractive correction just before prime focus).
    if frame >= sum(STAGE_FRAMES[:2]):
        p = stage_progress(frame, 2)
        # SINGLE bend point at the centre of the lens (no gap inside it)
        _corr_entry_offset = 0.15
        corr_bend_x = CORR_XY[0] + _perp_x * _corr_entry_offset
        corr_bend_y = CORR_XY[1] + _perp_y * _corr_entry_offset
        # focal plane entry (on the mirror-facing edge of the disc)
        fx_end = FP_XY[0] - _ax_x * FP_R * 0.25
        fy_end = FP_XY[1] - _ax_y * FP_R * 0.25
        # split progress 60/40 between the two segments
        if p <= 0.6:
            draw_beam(ax, (_mx_hit, _my_hit), (corr_bend_x, corr_bend_y),
                      WHITE_COL, beam_alpha(frame, 2), p / 0.6, n_parts=4)
        else:
            draw_beam(ax, (_mx_hit, _my_hit), (corr_bend_x, corr_bend_y),
                      WHITE_COL, beam_alpha(frame, 2), 1.0, n_parts=4)
            draw_beam(ax, (corr_bend_x, corr_bend_y), (fx_end, fy_end),
                      WHITE_COL, beam_alpha(frame, 2),
                      (p - 0.6) / 0.4, n_parts=4)

    # Stage 3: fiber run (light inside the cladding)
    if frame >= sum(STAGE_FRAMES[:3]):
        p = stage_progress(frame, 3)
        draw_fiber_beam(ax, FIB_P0, FIB_P1, FIB_P2, FIB_P3,
                         WHITE_COL, beam_alpha(frame,3), p, n_parts=8)

    # Stage 4: light enters V-groove at the fiber endpoint, travels along
    # the groove, then continues straight through the collimator to D1.
    if frame >= sum(STAGE_FRAMES[:4]):
        p = stage_progress(frame, 4)
        fiber_tip = (VG_XY[0] + VG_W/2, FIB_GROOVE_Y)   # right face of V-groove
        draw_beam(ax, fiber_tip, D1_XY,
                  WHITE_COL, beam_alpha(frame, 4),
                  p, n_parts=6, lw=2.2)

    # Stage 5: at D1, blue reflects DOWN to grating B and red+NIR transmits
    # LEFT toward D2 — both outputs propagate simultaneously.
    if frame >= sum(STAGE_FRAMES[:5]):
        p = stage_progress(frame, 5)
        draw_beam(ax, D1_XY, (D1_XY[0], GB_XY[1]),
                  BLUE_COL, beam_alpha(frame, 5) * 0.95, p, n_parts=4)
        draw_beam(ax, D1_XY, D2_XY,
                  '#ffaa55', beam_alpha(frame, 5),
                  p, n_parts=4)

    # Stage 6: at D2, red reflects DOWN to grating R and NIR transmits LEFT
    # to grating N — both outputs propagate simultaneously.
    if frame >= sum(STAGE_FRAMES[:6]):
        p = stage_progress(frame, 6)
        draw_beam(ax, D2_XY, (GR_XY[0], GR_XY[1]),
                  RED_COL, beam_alpha(frame, 6), p, n_parts=3)
        draw_beam(ax, D2_XY, (GN_XY[0], GN_XY[1]),
                  NIR_COL, beam_alpha(frame, 6), p, n_parts=3)

    # CCD half-size (square CCD: CCD_H = CCD_SIDE)
    _ccd_half = CCD_H / 2

    # Grating dispersion fans — each begins the instant its arm beam reaches
    # the grating, so there is no idle pause between arrival and dispersion.
    _blue_arr = sum(STAGE_FRAMES[:6])          # end of stage 5
    _red_arr  = sum(STAGE_FRAMES[:7])          # end of stage 6
    _nir_arr  = sum(STAGE_FRAMES[:7])          # end of stage 6
    _fan_dur  = STAGE_FRAMES[7]                # keep original expansion time

    def _fan_alpha(pp):
        return 0.85 if pp >= 1 else ease(pp) * 0.85

    # Blue grating diffracts → fan → CCD B
    if frame >= _blue_arr:
        p = float(np.clip((frame - _blue_arr) / _fan_dur, 0, 1))
        a = _fan_alpha(p) * 0.9
        impact = (GB_XY[0], GB_XY[1])
        blues = ['#5522cc','#3355dd','#3399ff','#22ccff','#00ffff']
        ccd_top = CB_XY[1] + _ccd_half
        for i, bc in enumerate(blues):
            tx = CB_XY[0] + (i - 2) * 0.18
            ex = lerp(impact[0], tx, p)
            ey = lerp(impact[1], ccd_top, p)
            ax.plot([impact[0], ex], [impact[1], ey],
                    color=bc, lw=1.3, alpha=a, zorder=5)

    # Red grating → fan → CCD R  (light DOWN — red reflects off D2)
    if frame >= _red_arr:
        p = float(np.clip((frame - _red_arr) / _fan_dur, 0, 1))
        a = _fan_alpha(p) * 0.9
        impact = (GR_XY[0], GR_XY[1])
        reds = ['#ffee44','#ffbb33','#ff8822','#ff5511','#ee3300']
        ccd_top = CR_XY[1] + _ccd_half
        for i, rc in enumerate(reds):
            tx = CR_XY[0] + (i - 2) * 0.18
            ex = lerp(impact[0], tx, p)
            ey = lerp(impact[1], ccd_top, p)
            ax.plot([impact[0], ex], [impact[1], ey],
                    color=rc, lw=1.3, alpha=a, zorder=5)

    # NIR grating → fan → CCD N  (light LEFT — NIR passes through D2)
    if frame >= _nir_arr:
        p = float(np.clip((frame - _nir_arr) / _fan_dur, 0, 1))
        a = _fan_alpha(p) * 0.9
        impact = (GN_XY[0], GN_XY[1])
        nirs = ['#ff8866','#ee5544','#dd3322','#bb1100','#661100']
        ccd_right = CN_XY[0] + CCD_W/2
        for i, nc in enumerate(nirs):
            ty = CN_XY[1] + (i - 2) * 0.18
            ex = lerp(impact[0], ccd_right, p)
            ey = lerp(impact[1], ty, p)
            ax.plot([impact[0], ex], [impact[1], ey],
                    color=nc, lw=1.3, alpha=a, zorder=5)

    # CCD illumination — spectrum strips appear as soon as each arm's fan
    # reaches its CCD (no separate end-of-animation stage).
    _CCD_FADE = 20                            # frames of fade-in per CCD
    if frame >= _blue_arr + _fan_dur:
        p = float(np.clip((frame - (_blue_arr + _fan_dur)) / _CCD_FADE, 0, 1))
        draw_spectrum_fan(ax, CB_XY, 'blue', p)
    if frame >= _red_arr + _fan_dur:
        p = float(np.clip((frame - (_red_arr + _fan_dur)) / _CCD_FADE, 0, 1))
        draw_spectrum_fan(ax, CR_XY, 'red', p)
    if frame >= _nir_arr + _fan_dur:
        p = float(np.clip((frame - (_nir_arr + _fan_dur)) / _CCD_FADE, 0, 1))
        draw_spectrum_fan(ax, CN_XY, 'nir', p)



# ─────────────────────────────────────────────────────────────────────────────
#  BUILD FIGURE & ANIMATION
# ─────────────────────────────────────────────────────────────────────────────



# ======================================================================
# Entry point
# ======================================================================

def main():
    import os, sys

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 11], hspace=0)
    title_ax = fig.add_subplot(gs[0])
    main_ax  = fig.add_subplot(gs[1])
    for a in (title_ax, main_ax):
        a.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    def update(fr):
        draw_frame(fr, main_ax, title_ax)
        return main_ax, title_ax

    anim = FuncAnimation(fig, update, frames=range(TOTAL_FRAMES),
                         interval=50, blit=False, repeat=False,
                         cache_frame_data=False)
    fig._keep = anim

    _here = os.path.dirname(os.path.abspath(__file__))
    if "--save-png" in sys.argv:
        draw_frame(TOTAL_FRAMES - 10, main_ax, title_ax)
        _png = os.path.join(_here, "desi_light_path_ltr.png")
        fig.savefig(_png, dpi=130, facecolor=BG, bbox_inches='tight')
        print(f"  Saved: {_png}")

    if "--save-gif" in sys.argv:
        print(f"  Generating {TOTAL_FRAMES}-frame GIF…")
        try:
            _gif = os.path.join(_here, "desi_light_path_ltr.gif")
            anim_save = FuncAnimation(
                fig,
                lambda fr: (draw_frame(fr, main_ax, title_ax), main_ax, title_ax)[1:],
                frames=TOTAL_FRAMES, interval=50, blit=False)
            anim_save.save(_gif, writer='pillow', fps=20,
                           savefig_kwargs=dict(facecolor=BG))
            print(f"  Saved: {_gif}")
        except Exception as e:
            print(f"  GIF export skipped ({e})")

    plt.show()


if __name__ == "__main__":
    main()
