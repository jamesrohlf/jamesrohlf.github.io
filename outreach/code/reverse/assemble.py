#!/usr/bin/env python3
import sys
html = (open('head.html').read() + open('mid.html').read()
        + open('js.html').read().replace('__DATA__', open('reverse_data.json').read()))
out = sys.argv[1] if len(sys.argv) > 1 else '../../cmb-reverse.html'
open(out, 'w').write(html)
print(f"  {out}: {len(html)/1024:.0f} KB")
