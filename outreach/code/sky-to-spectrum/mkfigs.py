#!/usr/bin/env python3
"""Emit the two static SVG figures for sky-to-spectrum.html.

Figure 1: the FIRAS monopole spectrum against a 2.725 K Planck curve, with the
          residuals beneath at 10000x the vertical scale.
Figure 2: the universal blackbody photon spectrum in x = E/kT, showing what
          FIRAS actually measured and where hydrogen ionisation sits.
"""
import numpy as np
from scipy.integrate import quad

h=6.62607015e-34; c=2.99792458e8; kB=1.380649e-23; keV=8.617333262e-5
def esc(s): return s
def txt(x,y,s,anchor='middle',size=12,fill='var(--ink-faint)',extra=''):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" {extra}>{s}</text>')

# ---------------------------------------------------------------- figure 1
d=np.loadtxt('firas.txt'); nu,spec,resid,sig,gal=d.T
def planck_MJy(nu_cm,T):
    f=nu_cm*100*c
    return 2*h*f**3/c**2/np.expm1(h*f/(kB*T))*1e20
X0,X1=70,740; A0,A1=22,300; B0,B1=340,432
nx0,nx1=0.0,22.5; iy0,iy1=0.0,420.0; ry=200.0     # kJy/sr half-scale for residuals
fx=lambda v: X0+(X1-X0)*(v-nx0)/(nx1-nx0)
fa=lambda v: A1-(A1-A0)*(v-iy0)/(iy1-iy0)
fb=lambda v: (B0+B1)/2-((B1-B0)/2)*(v/ry)
s=[f'<svg viewBox="0 0 800 492" role="img" aria-labelledby="f1t f1d">',
   '<title id="f1t">The COBE FIRAS spectrum of the microwave background</title>',
   '<desc id="f1d">Intensity against frequency, with a 2.725 kelvin blackbody curve through '
   'the measurements, and residuals beneath magnified ten thousand times that remain '
   'consistent with zero.</desc>']
for v in (0,100,200,300,400):
    s.append(f'<line x1="{X0}" x2="{X1}" y1="{fa(v):.1f}" y2="{fa(v):.1f}" stroke="var(--rule)" stroke-width="1"/>')
    s.append(txt(X0-8,fa(v)+4,v,'end'))
for v in (0,5,10,15,20):
    s.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{B1}" y2="{B1+5}" stroke="var(--ink-faint)"/>')
    s.append(txt(fx(v),B1+19,v))
s.append(f'<rect x="{X0}" y="{A0}" width="{X1-X0}" height="{A1-A0}" fill="none" stroke="var(--ink-faint)"/>')
# the Planck curve
g=np.linspace(0.6,22.4,400)
s.append('<path d="'+' '.join(('M' if i==0 else 'L')+f'{fx(v):.1f} {fa(planck_MJy(v,2.725)):.1f}'
         for i,v in enumerate(g))+'" fill="none" stroke="var(--scarlet)" stroke-width="2"/>')
for a,b in zip(nu,spec):
    s.append(f'<circle cx="{fx(a):.1f}" cy="{fa(b):.1f}" r="2.6" fill="none" '
             f'stroke="var(--ink)" stroke-width="1.3"/>')
s.append(txt(20,(A0+A1)/2,'intensity  [MJy/sr]','middle',13,'var(--ink-soft)',
             f'transform="rotate(-90 20 {(A0+A1)/2:.0f})"'))
s.append(txt((X0+X1)/2,A0+18,'COBE/FIRAS measurements, and a 2.725 K blackbody','middle',13,'var(--ink-soft)'))
# residual panel
s.append(f'<rect x="{X0}" y="{B0}" width="{X1-X0}" height="{B1-B0}" fill="none" stroke="var(--ink-faint)"/>')
s.append(f'<line x1="{X0}" x2="{X1}" y1="{(B0+B1)/2:.1f}" y2="{(B0+B1)/2:.1f}" stroke="var(--ink-faint)" stroke-dasharray="3 3"/>')
clip=lambda v: max(-ry, min(ry, v))
noff=0
for a,r,e in zip(nu,resid,sig):
    off = abs(r) > ry
    if off: noff += 1
    s.append(f'<line x1="{fx(a):.1f}" x2="{fx(a):.1f}" y1="{fb(clip(r-e)):.1f}" y2="{fb(clip(r+e)):.1f}" stroke="var(--ink-soft)" stroke-width="1"/>')
    col = 'var(--scarlet)' if off else 'var(--ink-soft)'
    s.append(f'<circle cx="{fx(a):.1f}" cy="{fb(clip(r)):.1f}" r="1.8" fill="{col}"/>')
s.append(txt(X0-8,fb(150)+4,'+150','end',11)); s.append(txt(X0-8,fb(-150)+4,'\u2212150','end',11))
s.append(txt(20,(B0+B1)/2,'residual [kJy/sr]','middle',11,'var(--ink-soft)',
             f'transform="rotate(-90 20 {(B0+B1)/2:.0f})"'))
s.append(txt((X0+X1)/2,B1+40,'frequency  [cm\u207b\u00b9]','middle',13,'var(--ink-soft)'))
mag = (iy1/(A1-A0)) / ((ry/1000.0)/((B1-B0)/2))
s.append(txt(X1-6,B0+16,f'residuals, vertical scale \u00d7 {mag:.0f}','end',11,'var(--ink-faint)'))
s.append(txt(X1-6,B1-6,f'{noff} point{"s" if noff!=1 else ""} beyond \u00b1{ry:.0f} clipped','end',10,'var(--scarlet)'))
s.append('</svg>')
open('fig_firas.svg','w').write('\n'.join(s))
print(f"  fig_firas.svg  {len('\\n'.join(s))/1024:.1f} KB, {len(nu)} points")

# ---------------------------------------------------------------- figure 2
tot,_=quad(lambda x: x*x/np.expm1(x),0,80)
X0,X1,Y0,Y1=70,760,24,330
xmax=60.0; lo,hi=-22.0,0.5
gx=lambda v: X0+(X1-X0)*v/xmax
gy=lambda v: Y1-(Y1-Y0)*(v-lo)/(hi-lo)
s=[f'<svg viewBox="0 0 800 400" role="img" aria-labelledby="f2t f2d">',
   '<title id="f2t">The Wien tail of a blackbody, and where hydrogen ionisation sits</title>',
   '<desc id="f2d">The blackbody photon number spectrum plotted against photon energy in units '
   'of kT on a logarithmic vertical axis, falling by twenty orders of magnitude. The band COBE '
   'measured is shaded near the left; the energy needed to ionise hydrogen at recombination lies '
   'far out to the right.</desc>']
for e in range(0,-23,-4):
    s.append(f'<line x1="{X0}" x2="{X1}" y1="{gy(e):.1f}" y2="{gy(e):.1f}" stroke="var(--rule)"/>')
    s.append(txt(X0-8,gy(e)+4,('1' if e==0 else f'10<tspan baseline-shift="super" font-size="8">{e}</tspan>'),'end',11))
for v in (0,10,20,30,40,50,60):
    s.append(f'<line x1="{gx(v):.1f}" x2="{gx(v):.1f}" y1="{Y1}" y2="{Y1+5}" stroke="var(--ink-faint)"/>')
    s.append(txt(gx(v),Y1+19,v))
# what FIRAS measured
s.append(f'<rect x="{gx(1.2):.1f}" y="{Y0}" width="{gx(11.2)-gx(1.2):.1f}" height="{Y1-Y0}" '
         f'fill="var(--scarlet)" opacity="0.10"/>')
s.append(txt((gx(1.2)+gx(11.2))/2,Y0+16,'measured by COBE','middle',11,'var(--scarlet)'))
xs=np.linspace(0.05,xmax,700)
ys=np.log10(np.maximum(xs**2/np.expm1(xs)/tot,1e-30))
s.append('<path d="'+' '.join(('M' if i==0 else 'L')+f'{gx(a):.1f} {gy(min(b,hi)):.1f}'
         for i,(a,b) in enumerate(zip(xs,ys)) if b>lo)+'" fill="none" stroke="var(--scarlet)" stroke-width="2"/>')
for xv,lab,col in ((27.0,'one ionising photon per baryon (5850 K)','var(--gold)'),
                   (53.1,'13.6 eV at recombination, 2973 K','var(--ink)')):
    s.append(f'<line x1="{gx(xv):.1f}" x2="{gx(xv):.1f}" y1="{Y0}" y2="{Y1}" stroke="{col}" '
             f'stroke-width="1.4" stroke-dasharray="5 4"/>')
s.append(txt(gx(27)-6,Y0+40,'one ionising photon','end',11,'var(--gold)'))
s.append(txt(gx(27)-6,Y0+54,'per baryon  (5850 K)','end',11,'var(--gold)'))
s.append(txt(gx(53.1)-6,Y0+80,'13.6 eV at 2973 K','end',11,'var(--ink)'))
s.append(txt(gx(53.1)-6,Y0+94,'(recombination)','end',11,'var(--ink)'))
s.append(f'<rect x="{X0}" y="{Y0}" width="{X1-X0}" height="{Y1-Y0}" fill="none" stroke="var(--ink-faint)"/>')
s.append(txt((X0+X1)/2,Y1+42,'photon energy in units of kT','middle',13,'var(--ink-soft)'))
s.append(txt(20,(Y0+Y1)/2,'fraction of photons per unit ln E','middle',12,'var(--ink-soft)',
             f'transform="rotate(-90 20 {(Y0+Y1)/2:.0f})"'))
s.append('</svg>')
open('fig_tail.svg','w').write('\n'.join(s))
print(f"  fig_tail.svg   {len('\\n'.join(s))/1024:.1f} KB")
