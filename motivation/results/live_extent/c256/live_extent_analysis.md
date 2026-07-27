# Live-extent adjacency byte oracle

This oracle credits only live-degree packing. It does not use visited outcomes, Beam membership, or a query-dependent edge certificate.

## Sample

- Queries / score chunks: 462 / 6764
- Parents / live edges: 90493 / 4271919
- Average live degree: 47.21
- Trace overflow: 0

## Byte opportunity

| layout | sample bytes | bytes/parent | reduction vs fixed |
|---|---:|---:|---:|
| fixed record | 75290176 | 832.00 | 0.00% |
| ideal live prefix | 35623240 | 393.66 | 52.69% |
| 8-edge extent class | 38050896 | 420.48 | 49.46% |

The 8-edge class needs 6.320 extents/parent and adds 6.81% rounding bytes over the impossible byte-exact layout.

## Untagged-length continuation cost

| first prefix | parents needing tail | one contiguous tail WQE/parent | 8-edge chain WQE/parent lower–upper | contiguous payload/fixed | extent payload/fixed lower–upper |
|---:|---:|---:|---:|---:|---:|
| 8 | 90446 (99.95%) | 1.999 | 5.901–6.776 | 47.32% | 47.32%–54.04% |
| 16 | 88335 (97.62%) | 1.976 | 4.907–5.761 | 47.36% | 47.36%–53.93% |
| 32 | 64737 (71.54%) | 1.715 | 3.162–3.788 | 49.32% | 49.32%–54.14% |
| 48 | 36356 (40.18%) | 1.402 | 2.037–2.388 | 56.05% | 56.05%–58.75% |
| 64 | 18439 (20.38%) | 1.204 | 1.438–1.616 | 66.83% | 66.83%–68.20% |

## Projected total RDMA bytes

| layout | proposed graph bytes/query | proposed total bytes/query | total RDMA byte reduction |
|---|---:|---:|---:|
| ideal live prefix | 76837.4 | 96293.4 | 47.05% |
| 8-edge extent class | 82073.7 | 101529.7 | 44.17% |

A one-shot extent keeps one graph READ per parent only if the extent class is known before posting the RDMA READ. Otherwise the continuation table exposes the dependent-WQE cost. Version/checksum layout changes and their GPU cost are deliberately not modeled.
