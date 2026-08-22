#!/usr/bin/env python3
"""Render the two Program 3 story figures as dependency-free SVG."""

import html, json, sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: plot_story.py <run-root>")
root = Path(sys.argv[1]).resolve()
d = json.loads((root / "story_summary.json").read_text(encoding="utf-8"))

def svg(name, body, width=1100, height=450):
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><style>text{{font-family:Arial,"Noto Sans",sans-serif;fill:#222}}.t{{font-size:19px;font-weight:700}}.l{{font-size:14px}}.s{{font-size:12px;fill:#555}}.v{{font-size:13px;font-weight:700}}.a{{stroke:#555}}.g{{stroke:#ddd}}</style><rect width="100%" height="100%" fill="white"/>{body}</svg>'''
    (root / name).write_text(text, encoding="utf-8")
    print(root / name)

items = d["motivation"]
max_cand = max(x["candidate_slots_per_round"] for x in items) * 1.15
max_time = max(x["full_merge_pipeline_us_per_round"] for x in items) * 1.15
parts = ['<text class="t" x="25" y="34">(a) Batch expansion inflates the merge input</text>', '<text class="t" x="570" y="34">(b) Exact prefix is only a fraction of full merge</text>']
for i,x in enumerate(items):
    bx=65+i*115; h=245*x["candidate_slots_per_round"]/max_cand; y=330-h
    parts += [f'<rect x="{bx}" y="{y:.1f}" width="72" height="{h:.1f}" fill="#7570b3"/>',f'<text class="v" x="{bx+36}" y="{y-7:.1f}" text-anchor="middle">{x["candidate_slots_per_round"]:.0f}</text>',f'<text class="l" x="{bx+36}" y="355" text-anchor="middle">C={x["commit_width"]}</text>']
    bx2=610+i*112; total=245*x["full_merge_pipeline_us_per_round"]/max_time; pref=total*x["prefix_compute_share"]; rem=total-pref; y2=330-total
    parts += [f'<rect x="{bx2}" y="{y2:.1f}" width="70" height="{pref:.1f}" fill="#1b9e77"/>',f'<rect x="{bx2}" y="{y2+pref:.1f}" width="70" height="{rem:.1f}" fill="#d95f02"/>',f'<text class="v" x="{bx2+35}" y="{y2-7:.1f}" text-anchor="middle">{x["full_merge_pipeline_us_per_round"]:.1f} μs</text>',f'<text class="l" x="{bx2+35}" y="355" text-anchor="middle">C={x["commit_width"]}</text>',f'<text class="s" x="{bx2+35}" y="{y2+pref/2+4:.1f}" text-anchor="middle">{100*x["prefix_compute_share"]:.0f}%</text>']
parts += ['<line class="a" x1="45" y1="330" x2="525" y2="330"/><line class="a" x1="585" y1="330" x2="1060" y2="330"/>','<rect x="690" y="395" width="16" height="16" fill="#1b9e77"/><text class="s" x="714" y="408">Exact prefix computation</text><rect x="870" y="395" width="16" height="16" fill="#d95f02"/><text class="s" x="894" y="408">Remaining full-Beam merge</text>']
svg("program3_story_motivation.svg",''.join(parts))

p=d["performance"]; late,early=p["late"],p["early"]
qratio=early["qps"]/late["qps"]; values=[1,qratio]; labels=["Late-Issue","Exact-Early-Issue"]; colors=["#7570b3","#1b9e77"]
parts=['<text class="t" x="25" y="34">Exact-frontier issue/commit decoupling: one-shot strict A/B</text>']
for i,(value,label,color) in enumerate(zip(values,labels,colors)):
    x=250+i*300; h=245*value/max(values); y=330-h
    parts += [f'<rect x="{x}" y="{y:.1f}" width="150" height="{h:.1f}" fill="{color}"/>',f'<text class="v" x="{x+75}" y="{y-10:.1f}" text-anchor="middle">{value:.3f}× QPS</text>',f'<text class="l" x="{x+75}" y="360" text-anchor="middle">{html.escape(label)}</text>']
parts += [f'<text class="v" x="550" y="405" text-anchor="middle">P99 −{100*(1-early["p99_ms"]/late["p99_ms"]):.2f}% · ROB hit {100*early["critical_rob_hit_ratio"]:.2f}% · speculative reads 0 · Recall unchanged</text>','<line class="a" x1="150" y1="330" x2="850" y2="330"/>']
svg("program3_story_effectiveness.svg",''.join(parts))
