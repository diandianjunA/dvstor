#!/usr/bin/env python3
"""Plot co-rank (pre-merge top-C selection) share of total GPU query time."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files={'SIFT100M':ROOT/'sift100m/program3/results/program3_20260831_191030/summary.json','SPACEV100M':ROOT/'spacev100m/program3/results/program3_20260831_193611/summary.json','DEEP100M':ROOT/'deep100m/program3/results/program3_20260831_200109/summary.json'}
rows=[]
for n,f in files.items():
 p=json.loads(f.read_text())['performance']['early']; t=p['candidate_maintenance_breakdown']['Candidate Select']; total=t+p['candidate_maintenance_breakdown']['Merge']; rows.append((n,t,total,100*t/total))
W,H=760,430; x0,x1,y0,y1=115,700,90,350; ymax=100
def y(v): return y1-v/ymax*(y1-y0)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><style>.t{{font-family:"Times New Roman",serif;fill:#222}}.title{{font-size:18px;font-weight:bold}}.tick{{font-size:13px}}.axis{{stroke:#333}}.grid{{stroke:#ddd;stroke-width:.7}}</style><text x="380" y="35" text-anchor="middle" class="t title">Merge-time composition</text><rect x="255" y="57" width="16" height="12" fill="#3B5B92"/><text x="278" y="68" class="t tick">Co-rank selection</text><rect x="430" y="57" width="16" height="12" fill="#D9DEE8"/><text x="453" y="68" class="t tick">Remaining Merge</text>']
for k in range(0,101,20):
 yy=y(k);svg += [f'<line x1="{x0}" y1="{yy}" x2="{x1}" y2="{yy}" class="grid"/><text x="{x0-10}" y="{yy+4}" text-anchor="end" class="t tick">{k}%</text>']
svg += [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/><line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/><text transform="translate(35,220) rotate(-90)" text-anchor="middle" class="t tick">Share of Merge time (%)</text>']
for i,(n,t,total,pct) in enumerate(rows):
 c=x0+(i+.5)*(x1-x0)/3; yy=y(pct); svg += [f'<rect x="{c-18}" y="{yy}" width="36" height="{y1-yy}" fill="#3B5B92"/><rect x="{c-18}" y="{y0}" width="36" height="{yy-y0}" fill="#D9DEE8"/><text x="{c}" y="{yy-7}" text-anchor="middle" class="t tick">{pct:.1f}%</text><text x="{c}" y="{y1+27}" text-anchor="middle" class="t tick">{n}</text>']
svg.append('</svg>');out=ROOT/'paper_figures/corank_merge_time_share.svg';out.write_text(''.join(svg));subprocess.run(['convert',str(out),str(out.with_suffix('.png'))],check=True)
for n,t,total,p in rows:print(f'{n}: {t:.3f} us / {total:.3f} us = {p:.2f}%')
