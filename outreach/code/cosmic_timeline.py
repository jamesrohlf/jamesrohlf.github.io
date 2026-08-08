"""
Quantitative cosmic timeline - flat version (no trumpet).
Era bands on top, twin axes (time / z / 1+z) below.
Two-segment x: log early, linear late.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import quad

# Cosmology
H0 = 67.36; h = H0/100; Om = 0.3153; OL = 1-Om
Ogamma_h2 = 2.473e-5; Neff = 3.046
Or = (Ogamma_h2 + Neff*(7/8)*(4/11)**(4/3)*Ogamma_h2) / h**2
Mpc_km, yr_s = 3.0857e19, 3.1557e7
tH_yr = Mpc_km/H0/yr_s

def E(z): zp=1+z; return np.sqrt(Or*zp**4 + Om*zp**3 + OL)
def age_yr(z):
    val,_ = quad(lambda zp: 1/((1+zp)*E(zp)), z, np.inf, limit=200)
    return val*tH_yr

t0_yr = age_yr(0)

def t_for_z(z):
    if z < 1e6: return age_yr(z)
    return 0.5*tH_yr/(np.sqrt(Or)*(1+z)**2)

# Two-segment axis
T_BREAK = 1e9; T_MIN = 1e-44/yr_s; T_MAX = t0_yr
W_LOG = 0.55; W_LIN = 0.45
LOG_MIN = np.log10(T_MIN); LOG_MAX = np.log10(T_BREAK)

def tx(t):
    if t <= T_BREAK:
        return (np.log10(t)-LOG_MIN)/(LOG_MAX-LOG_MIN)*W_LOG
    return W_LOG + (t-T_BREAK)/(T_MAX-T_BREAK)*W_LIN

# Figure - single panel for eras, axes below
fig = plt.figure(figsize=(15, 8.5), facecolor='white')
ax_mid = fig.add_axes([0.07, 0.30, 0.90, 0.50])

# ============== Era bands ==============
eras = [
    (1e-43/yr_s, 1e-36/yr_s, 'Planck era',             '#ffe680'),
    (1e-36/yr_s, 1e-32/yr_s, 'Inflation',              '#fff2cc'),
    (1e-32/yr_s, 1e-6/yr_s,  'Quark–gluon\nplasma',    '#ffd9b3'),
    (1e-6/yr_s, 3*60/yr_s,   'Big\nBang\nNucleo-\nsynthesis', '#ffb380'),
    (3*60/yr_s, 50e3,        'Radiation\ndomination',  '#ffaaaa'),
    (50e3, 370e3,            'Matter dom.\n(pre-CMB)', '#aaccff'),
    (370e3, 150e6,           'Dark Ages',              '#3a3a5c'),
    (150e6, 1e9,             'Copious Star Formation', '#7a5a9f'),
    (1e9, 4e9,               'Galaxy\nassembly',       '#5a8fbf'),
    (4e9, 9.5e9,             'Mature galaxies',        '#8fc88f'),
    (9.5e9, t0_yr,           'Dark-energy era',        '#ffcc88'),
]

for t_s, t_e, label, color in eras:
    x_s = tx(t_s); x_e = tx(t_e)
    rect = mpatches.Rectangle((x_s, 0.0), x_e-x_s, 1.0,
                              facecolor=color, edgecolor='#222', lw=0.7, alpha=0.92)
    ax_mid.add_patch(rect)
    if x_e - x_s > 0.022:
        text_color = '#fff' if color in ['#3a3a5c', '#7a5a9f'] else '#1a1a1a'
        # Planck era gets a 2-line vertical label: heading + ℓ_P, m_P, T_P units subtitle
        if label == 'Planck era':
            ax_mid.text((x_s+x_e)/2 - 0.006, 0.5, label,
                        ha='center', va='center', fontsize=8, color=text_color,
                        fontweight='bold', rotation=90)
            ax_mid.text((x_s+x_e)/2 + 0.010, 0.5,
                        r'$\ell_P,\ m_P,\ T_P$',
                        ha='center', va='center', fontsize=9, color=text_color,
                        rotation=90, style='italic')
        # Narrow boxes get vertical labels (but only those wide enough for the text)
        elif label in ('Dark Ages', 'Inflation'):
            ax_mid.text((x_s+x_e)/2, 0.5, label,
                        ha='center', va='center', fontsize=8, color=text_color,
                        fontweight='bold', rotation=90)
        else:
            ax_mid.text((x_s+x_e)/2, 0.5, label,
                        ha='center', va='center', fontsize=9, color=text_color,
                        fontweight='bold')

ax_mid.set_xlim(-0.005, 1.005); ax_mid.set_ylim(-0.55, 2.0)
ax_mid.set_yticks([]); ax_mid.set_xticks([])
for s in ax_mid.spines.values(): s.set_visible(False)

# Transition markers with leader lines above the band
transitions = [
    # (t_yr, label, color, label_x, tick_h_above_band, label_y)
    (5.1e4,  'z_eq ≈ 3400\n(matter–rad eq.)',  '#aa0000', 0.45, 0.18, 1.28),
    (3.71e5, 'z ≈ 1090\n(recombination)',       '#cc4400', 0.51, 0.62, 1.72),
    (1.8e8,  'z ≈ 20\n(1st stars)',             '#5a3a8f', 0.56, 0.20, 1.28),
    (3.27e9, 'z ≈ 2\n(cosmic noon)',            '#1a6633', None, 0.20, 1.28),
    (9.85e9, 'z ≈ 0.3\n(matter–Λ eq.)',         '#996600', None, 0.20, 1.28),
]
for t_tr, label, color, label_x, h, y_label in transitions:
    x_tr = tx(t_tr)
    ax_mid.plot([x_tr, x_tr], [1.02, 1.0 + h], color=color, lw=1.4, zorder=5)
    if label_x is None:
        ax_mid.text(x_tr, y_label, label, ha='center', va='bottom',
                    fontsize=8, color=color, fontweight='bold')
    else:
        ax_mid.plot([x_tr, label_x], [1.0 + h, y_label - 0.04],
                    color=color, lw=0.7, zorder=5)
        ax_mid.text(label_x, y_label, label, ha='center', va='bottom',
                    fontsize=8, color=color, fontweight='bold')

# "Now" label at right edge of timeline
x_now = tx(t0_yr)
ax_mid.plot([x_now, x_now], [1.02, 1.20], color='#225522', lw=1.4, zorder=5)
ax_mid.text(x_now, 1.30, 'Now\n(z = 0)', ha='center', va='bottom',
            fontsize=9, color='#225522', fontweight='bold')

# Planck time pointer to the yellow Planck era box
# t_P ≈ 5.4×10⁻⁴⁴ s; the Planck era is defined as starting at t_P
x_planck = tx(1e-43/yr_s)  # left edge of Planck era box
ax_mid.plot([x_planck, x_planck], [1.02, 1.20], color='#aa7700', lw=1.4, zorder=5)
ax_mid.text(x_planck, 1.30, r'$t_P \approx 5.4 \times 10^{-44}$ s'+'\n(Planck time)',
            ha='left', va='bottom',
            fontsize=8, color='#aa7700', fontweight='bold')

# Below-band labels for narrow era boxes that can't fit in-line text
below_labels = [
    (5.1e4,  3.71e5,  'Matter dom.\n(pre-CMB)',    '#3366aa', 0.57, -0.28),
]
for t_s, t_e, label, color, label_x, label_y in below_labels:
    x_c = tx(0.5*(t_s + t_e))
    ax_mid.plot([x_c, x_c], [-0.02, -0.18], color=color, lw=1.2, zorder=5)
    ax_mid.plot([x_c, label_x], [-0.18, label_y + 0.03], color=color, lw=0.7, zorder=5)
    ax_mid.text(label_x, label_y, label, ha='center', va='top',
                fontsize=8, color=color, fontweight='bold')

# Title above
fig.suptitle('Cosmic Timeline  —  Age, Redshift, and Major Eras',
             fontsize=15, fontweight='bold', y=0.94)
fig.text(0.5, 0.89,
         r'flat $\Lambda$CDM, Planck 2018:      '
         r'$H_0 = 67.36$ km s$^{-1}$ Mpc$^{-1}$      '
         r'$\Omega_m = 0.3153$      '
         r'$\Omega_\Lambda = 0.6847$      '
         r'$t_0 = 13.80$ Gyr',
         ha='center', fontsize=12, style='italic', color='#444')

# ============== Quad-row axis ==============
y_t = 0.27; y_z = 0.19; y_1z = 0.11; y_kT = 0.03
ax_L, ax_R = 0.07, 0.97
def xf(x): return ax_L + x*(ax_R-ax_L)

for y in (y_t, y_z, y_1z, y_kT):
    fig.add_artist(plt.Line2D([ax_L, ax_R], [y, y], color='#000', lw=1.0,
                              transform=fig.transFigure))

t_ticks = [(1e-43/yr_s, r'$10^{-43}$ s'), (1e-32/yr_s, r'$10^{-32}$ s'),
           (1e-6/yr_s, '1 μs'),
           (1.0, '1 yr'), (3*60/yr_s, '3 min'),
           (1e9, '1 Gyr'), (5e9, '5 Gyr'),
           (10e9, '10 Gyr'), (t0_yr, '13.8 Gyr')]
t_ticks.sort()

for t, lbl in t_ticks:
    x = xf(tx(t))
    fig.add_artist(plt.Line2D([x, x], [y_t, y_t+0.015], color='#000', lw=1.0,
                              transform=fig.transFigure))
    fig.text(x, y_t-0.030, lbl, ha='center', va='top',
             fontsize=10.5)

fig.text(ax_L-0.008, y_t, 'time:', ha='right', va='center',
         fontsize=10, fontweight='bold')

z_ticks = [
    (1e9,  '10⁹',  '10⁹'),
    (1e6,  '10⁶',  '10⁶'),
    (1e4,  '10⁴',  '10⁴'),
    (1090, '1090', '1091'),
    (6,    '6',    '7'),
    (2,    '2',    '3'),
    (1,    '1',    '2'),
    (0.5,  '0.5',  '1.5'),
    (0,    '0',    '1'),
]

for z, lz, l1z in z_ticks:
    t = t_for_z(z)
    x_n = tx(t)
    if not (-0.01 <= x_n <= 1.01): continue
    x = xf(x_n)
    # 1090/1091 go ABOVE the tick to escape the dense cluster with 10⁴ and 6/7
    above = (z == 1090)
    # z line
    fig.add_artist(plt.Line2D([x, x], [y_z, y_z+0.010], color='#222', lw=0.8,
                              transform=fig.transFigure))
    if above:
        fig.text(x, y_z+0.014, lz, ha='center', va='bottom',
                 fontsize=12, color='#222')
    else:
        fig.text(x, y_z-0.022, lz, ha='center', va='top',
                 fontsize=12, color='#222')
    # 1+z line
    fig.add_artist(plt.Line2D([x, x], [y_1z, y_1z+0.010], color='#222', lw=0.8,
                              transform=fig.transFigure))
    if above:
        fig.text(x, y_1z+0.014, l1z, ha='center', va='bottom',
                 fontsize=12, color='#222')
    else:
        fig.text(x, y_1z-0.022, l1z, ha='center', va='top',
                 fontsize=12, color='#222')

fig.text(ax_L-0.008, y_z, 'z:', ha='right', va='center',
         fontsize=10, fontweight='bold', color='#222')
fig.text(ax_L-0.008, y_1z, '1+z:', ha='right', va='center',
         fontsize=10, fontweight='bold', color='#222')

# kT row: scale photon temperature kT = T0 × (1+z), where T0 = 2.348e-4 eV (kB × 2.7255 K)
kT0_eV = 2.348e-4
def z_for_kT(kT_eV):
    """Inverse: given kT in eV, return z such that kT = T0*(1+z)."""
    return kT_eV / kT0_eV - 1.0

# Pick ticks at "nice" temperatures spanning the figure
# Today: kT ≈ 0.235 meV ≈ 2.3e-4 eV; Planck era: kT ≈ M_Pl ≈ 10^28 eV
kT_ticks_spec = [
    (1e28, r'$10^{28}$ eV'),   # Planck scale (M_Pl c^2)
    (1e22, r'$10^{22}$ eV'),   # GUT-ish
    (1e16, r'$10^{16}$ eV'),
    (1e12, '1 TeV'),
    (1e9,  '1 GeV'),
    (1e6,  '1 MeV'),           # BBN-ish
    (1e3,  '1 keV'),
    (1.0,  '1 eV'),            # recombination region
    (1e-3, '1 meV'),           # cosmic noon era
    (5e-4, '0.5 meV'),
    (2.5e-4, '0.25 meV'),      # near today (T_CMB ≈ 0.235 meV)
]

for kT_eV, lbl in kT_ticks_spec:
    z_here = z_for_kT(kT_eV)
    if z_here < -0.5:
        continue  # below today
    t = t_for_z(max(z_here, 0))
    x_n = tx(t)
    if not (-0.01 <= x_n <= 1.01):
        continue
    x = xf(x_n)
    fig.add_artist(plt.Line2D([x, x], [y_kT, y_kT+0.010], color='#222', lw=0.8,
                              transform=fig.transFigure))
    fig.text(x, y_kT-0.022, lbl, ha='center', va='top',
             fontsize=11, color='#222')

fig.text(ax_L-0.008, y_kT, 'kT:', ha='right', va='center',
         fontsize=10, fontweight='bold', color='#222')

# Segment break line (with small label clarifying it's a scale change)
xf_break = xf(W_LOG)
fig.add_artist(plt.Line2D([xf_break, xf_break], [0.005, 0.78],
                          color='#999', lw=0.8, ls=':', alpha=0.55,
                          transform=fig.transFigure))
fig.text(xf_break, y_t+0.025, '↔', ha='center', va='bottom',
         fontsize=13, color='#555')
fig.text(xf_break-0.013, y_t+0.025, 'log', ha='right', va='bottom',
         fontsize=13, style='italic', color='#555')
fig.text(xf_break+0.013, y_t+0.025, 'linear', ha='left', va='bottom',
         fontsize=13, style='italic', color='#555')

plt.savefig('/home/claude/cosmic_timeline.pdf', bbox_inches='tight', facecolor='white')
plt.savefig('/home/claude/cosmic_timeline.png', dpi=130, bbox_inches='tight', facecolor='white')
print("saved")
