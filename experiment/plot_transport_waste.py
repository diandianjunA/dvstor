#!/usr/bin/env python3
"""Plot average bytes transferred per committed dynamic parent."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files={
 'SIFT100M':ROOT/'sift100m/program2/results/program2_20260831_190527/summary.json',
 'SPACEV100M':ROOT/'spacev100m/program2/results/program2_20260831_193042/summary.json',
 'DEEP100M':ROOT/'deep100m/program2/results/program2_20260831_195403/summary.json'}
vals=[]
for name,f in files.items():
 d=json.loads(f.read_text())['cases']; vals.append((name,d['fixed']['dynamic_bytes_per_committed_parent'],d['header']['dynamic_bytes_per_committed_parent'],d['live']['dynamic_bytes_per_committed_parent']))
W,H=900,470; x0,x1,y0,y1=100,850,90,380; ymax=900
colors=['#3B5B92','#C27A29','#5A8F72']; labels=['Complete read','Header then read','Our method']
def y(v): return y1-(v/ymax)*(y1-y0)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><style>.t{{font-family:"Times New Roman",serif;fill:#222}}.title{{font-size:18px;font-weight:bold}}.tick{{font-size:13px}}.axis{{stroke:#333;stroke-width:1}}.grid{{stroke:#ddd;stroke-width:.7}}</style>']
svg += ['<text x="450" y="35" text-anchor="middle" class="t title">Average transmission bytes per committed parent</text>']
for t in range(0,901,100):
 yy=y(t); svg += [f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" class="grid"/><text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end" class="t tick">{t}</text>']
svg += [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>',f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>',f'<text transform="translate(35,235) rotate(-90)" text-anchor="middle" class="t tick">Bytes per committed parent</text>']
for i,(name,*v) in enumerate(vals):
 c=x0+(i+.5)*(x1-x0)/3; bw=35
 svg.append(f'<text x="{c}" y="{y1+28}" text-anchor="middle" class="t tick">{name}</text>')
 for j,z in enumerate(v):
  xx=c+(j-1)*bw; yy=y(z); svg.append(f'<rect x="{xx-bw/2:.1f}" y="{yy:.1f}" width="{bw-3}" height="{y1-yy:.1f}" fill="{colors[j]}"/><text x="{xx:.1f}" y="{yy-6:.1f}" text-anchor="middle" class="t tick">{z:.0f}</text>')
for j,l in enumerate(labels):
 xx=300+j*180; svg += [f'<rect x="{xx}" y="57" width="16" height="12" fill="{colors[j]}"/><text x="{xx+23}" y="68" class="t tick">{l}</text>']
svg.append('</svg>'); out=ROOT/'paper_figures/transport_waste_comparison.svg'; out.write_text(''.join(svg)); subprocess.run(['convert',str(out),str(out.with_suffix('.png'))],check=True)
print('\n'.join(f'{n}: '+', '.join(f'{x:.3f}' for x in v) for n,*v in vals)); print(out)
