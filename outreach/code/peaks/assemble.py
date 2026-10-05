#!/usr/bin/env python3
import sys
mid = open('mid.html').read().replace('<!--BOX-->', open('box.html').read())
js  = (open('js.html').read()
       .replace('__BOXJS__', open('boxjs.html').read())
       .replace('__DATA__', open('peaks_data.json').read())
       .replace('__BOX__',  open('box_data.json').read()))
html = open('head.html').read() + mid + js
out = sys.argv[1] if len(sys.argv) > 1 else '../../cmb-peaks.html'
open(out, 'w').write(html)
print(f"  {out}: {len(html)/1024:.0f} KB")
