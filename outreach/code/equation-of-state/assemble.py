#!/usr/bin/env python3
import sys
html = open('head.html').read() + open('mid.html').read() + open('js.html').read()
out = sys.argv[1] if len(sys.argv) > 1 else '../../equation-of-state.html'
open(out, 'w').write(html)
print(f"  {out}: {len(html)/1024:.0f} KB")
