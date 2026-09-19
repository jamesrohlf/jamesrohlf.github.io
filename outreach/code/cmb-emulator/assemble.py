#!/usr/bin/env python3
import json, sys
pay = open('payload.json').read()
pl  = open('planck_points.json').read()
js  = (open('cmb.js.html').read()
       .replace('__PAYLOAD__', pay)
       .replace('__PLANCK__', pl))
html = open('cmb.head.html').read() + open('cmb.prose.html').read() + js
out = sys.argv[1] if len(sys.argv)>1 else 'cmb-spectrum.html'
open(out,'w').write(html)
print(f"  {out}: {len(html)/1024:.0f} KB")
