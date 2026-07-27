# Live-extent adjacency byte oracle

This oracle credits only live-degree packing. It does not use visited outcomes, Beam membership, or a query-dependent edge certificate.

## Sample

- Queries / score chunks: 152 / 2233
- Parents / live edges: 29793 / 1395491
- Average live degree: 46.84
- Trace overflow: 0

## Byte opportunity

| layout | sample bytes | bytes/parent | reduction vs fixed |
|---|---:|---:|---:|
| fixed record | 24787776 | 832.00 | 0.00% |
| ideal live prefix | 11640616 | 390.72 | 53.04% |
| 8-edge extent class | 12445200 | 417.72 | 49.79% |

The 8-edge class needs 6.277 extents/parent and adds 6.91% rounding bytes over the impossible byte-exact layout.

## Untagged-length continuation cost

| first prefix | parents needing tail | one contiguous tail WQE/parent | 8-edge chain WQE/parent lower–upper | contiguous payload/fixed | extent payload/fixed lower–upper |
|---:|---:|---:|---:|---:|---:|
| 8 | 29766 (99.91%) | 1.999 | 5.855–6.729 | 46.96% | 46.96%–53.69% |
| 16 | 29083 (97.62%) | 1.976 | 4.862–5.716 | 47.02% | 47.02%–53.59% |
| 32 | 21328 (71.59%) | 1.716 | 3.113–3.740 | 48.95% | 48.95%–53.77% |
| 48 | 11790 (39.57%) | 1.396 | 1.994–2.340 | 55.72% | 55.72%–58.39% |
| 64 | 5814 (19.51%) | 1.195 | 1.412–1.583 | 66.63% | 66.63%–67.94% |

## Projected total RDMA bytes

| layout | proposed graph bytes/query | proposed total bytes/query | total RDMA byte reduction |
|---|---:|---:|---:|
| ideal live prefix | 76263.3 | 95719.3 | 47.36% |
| 8-edge extent class | 81534.5 | 100990.5 | 44.47% |

A one-shot extent keeps one graph READ per parent only if the extent class is known before posting the RDMA READ. Otherwise the continuation table exposes the dependent-WQE cost. Version/checksum layout changes and their GPU cost are deliberately not modeled.
