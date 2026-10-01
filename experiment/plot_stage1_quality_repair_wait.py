#!/usr/bin/env python3
"""Generate publication-style SVG/PNG plots for stage-1 recall and repair wait."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
runs={"SIFT100M":ROOT/'sift100m/program1/results/program1_20260831_190010',"SPACEV100M":ROOT/'spacev100m/program1/results/program1_20260831_192505',"DEEP100M":ROOT/'deep100m/program1/results/program1_20260831_194733'}
rows=[]
for name,run in runs.items():
 q=json.loads((run/'quality/quality.json').read_text()); r=json.loads(next((run/'solution').glob('**/*.json')).read_text()); p=r['stage2']['stage2_packing']; rows.append((name,100*q['stage1_only_self_recall']['hit_rate'],100*q['finalized_self_recall']['hit_rate'],p['wait_ns']/p['waited_batches']/1e6))
W,H=1000,430; left,right,top,bottom=95,980,88,355; gap=105; pw=(right-left-gap)//2; colors=['#3B5B92','#C27A29','#5A8F72']
def esc(s): return s.replace('&','&amp;')
def panel(x,y,w,h,vals,ymax,title,ylabel,group=False):
 out=[f'<text x="{x+w/2}" y="55" text-anchor="middle" class="title">{title}</text>']
 for t in range(0,int(ymax)+1,20 if ymax>50 else 5):
  yy=y+h-t/ymax*h; out += [f'<line x1="{x}" y1="{yy:.1f}" x2="{x+w}" y2="{yy:.1f}" class="grid"/><text x="{x-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{t}</text>']
 out += [f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" class="axis"/>',f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" class="axis"/>',f'<text transform="translate({x-62},{y+h/2}) rotate(-90)" text-anchor="middle" class="label">{ylabel}</text>']
 n=len(rows); bw=22; centers=[x+w*(i+.5)/n for i in range(n)]
 for i,(name,*_) in enumerate(rows):
  out.append(f'<text x="{centers[i]}" y="{y+h+25}" text-anchor="middle" class="tick">{name}</text>')
  if group:
   for j,v in enumerate(vals[i]):
    xx=centers[i]+(j-0.5)*bw; hh=v/ymax*h; out.append(f'<rect x="{xx:.1f}" y="{y+h-hh:.1f}" width="{bw-2}" height="{hh:.1f}" fill="{colors[j]}"/>')
  else:
   v=vals[i]; hh=v/ymax*h; out.append(f'<rect x="{centers[i]-bw/2:.1f}" y="{y+h-hh:.1f}" width="{bw}" height="{hh:.1f}" fill="{colors[2]}"/>')
 return ''.join(out)
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><style>.title,.label,.tick{{font-family:"Times New Roman",serif;fill:#222}}.title{{font-size:17px}}.label{{font-size:14px}}.tick{{font-size:13px}}.axis{{stroke:#333;stroke-width:1}}.grid{{stroke:#ddd;stroke-width:.7}}</style>'''
svg+='<text x="370" y="25" class="tick">Stage 1</text><rect x="425" y="14" width="16" height="12" fill="#3B5B92"/><text x="470" y="25" class="tick">After repair</text><rect x="550" y="14" width="16" height="12" fill="#C27A29"/>'
svg+=panel(left,88,pw,267,[[r[1],r[2]] for r in rows],100,'(a) Query accuracy','Self-recall@10 (%)',True)
svg+=panel(left+pw+gap,88,pw,267,[r[3] for r in rows],20,'(b) Average repair wait','Wait per repaired batch (ms)')+'</svg>'
out=ROOT/'paper_figures/stage1_query_accuracy_repair_wait.svg'; out.write_text(svg)
subprocess.run(['convert',str(out),str(out.with_suffix('.png'))],check=True)
print(f'Wrote {out} and {out.with_suffix(".png")}')
