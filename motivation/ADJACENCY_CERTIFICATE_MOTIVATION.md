# Certified query-selective adjacency transfer: motivation result

## Verdict

**Stop this design as the main query innovation.** On the current SIFT100M
C16 baseline, neither the implementable group certificate nor the suffix
certificate exposes enough query-selective remote payload to offset the extra
one-sided RDMA stage. The result misses the phase gate by orders of magnitude,
not by a tunable constant.

The only large byte opportunity observed is unused capacity in the fixed-size
832-byte graph record. That is a remote record-layout/packing opportunity, not
evidence for query-dependent certified edge transfer.

## Experiment

Configuration:

- dataset/index: SIFT100M, schema-16 OPQ/PQ32;
- search: fixed batch 16, stable-run Beam merge, Beam 128;
- concurrency: 256;
- storage updates: disabled;
- sample: 152 queries, 2,233 score chunks, 29,793 expanded parents and
  1,395,491 decoded edges;
- graph record: 832 bytes, 102 RemotePtr capacity;
- trace overflow: zero;
- Recall@10 before and after the trace run: 0.935.

The probe is read-only and never changes Beam, visited, expansion count, graph
reads or results. It runs only on sampled queries and is deliberately excluded
from performance claims.

### Three distinct oracles

1. **Perfect pre-transfer oracle.** It uses exact device ADC for every
   structurally valid edge, including edges that the later visited check
   rejects. This is an unattainable upper bound, but it does not cheat by using
   neighbor-ID information unavailable before transfer.
2. **Implementable annulus certificate.** For each contiguous group it derives
   `[a,b]` from parent/neighbor PQ reconstruction distances and evaluates
   `LB=max(0,a-r,r-b)^2` against the full Beam's pre-merge cutoff.
3. **Implementable suffix certificate.** It reads a fixed prefix and uses the
   minimum parent/neighbor PQ radius of the remaining suffix. A failed
   certificate causes one additional contiguous tail READ.

Beam-not-full chunks, dynamic edges and invalid/unknown records are never
credited as skippable. The comparison is strict and uses the cutoff before the
current merge.

## Observations

### The post-read usefulness funnel is sparse, but that does not imply
pre-transfer certifiability

| stage | edges | fraction of decoded |
|---|---:|---:|
| decoded | 1,395,491 | 100% |
| visited survivors / finite PQ scores | 1,228,537 | 88.04% |
| new candidates entering Beam | 112,361 | 8.05% |

Only 8.05% of decoded edges enter Beam, but the GPU cannot know which edges
those are before receiving their IDs. Using final Beam membership as a byte
oracle would therefore overstate the opportunity.

### The feasible geometric certificate is effectively powerless

| group size | groups | perfect oracle can skip | annulus certificate can skip | coalesced WQE/parent |
|---:|---:|---:|---:|---:|
| 4 | 359,606 | 48.18% | 3 groups (0.00083%) | 2.000 |
| 8 | 187,008 | 38.13% | 2 groups (0.00107%) | 2.000 |
| 16 | 100,612 | 29.66% | 1 group (0.00099%) | 2.000 |
| 32 | 56,935 | 21.82% | 0 | 2.000 |

There were zero lower-bound violations. The failure is not numerical
incorrectness: the lower bound is simply too loose. For almost every group,
the query-parent radius lies inside the group's parent-centered radial
interval, so the certified lower bound collapses to zero.

Even the perfect group oracle leaves at least one required run for essentially
every parent. Commodity one-sided RDMA would therefore turn one graph READ per
parent into one synopsis READ plus at least one group READ per parent.

### A contiguous prefix/suffix layout does not rescue the design

| prefix edges | perfect live-edge skip | suffix-certificate live-edge skip | tail-free parents | WQE/parent | chunks with second stage |
|---:|---:|---:|---:|---:|---:|
| 8 | 7.71% | 0.00% | 0.09% | 1.999 | 100.00% |
| 16 | 10.59% | ~0.00% | 2.39% | 1.976 | 99.78% |
| 32 | 8.71% | 0.00% | 28.41% | 1.716 | 96.51% |
| 48 | 4.50% | 0.00% | 60.43% | 1.396 | 82.94% |
| 64 | 1.72% | 0.00% | 80.49% | 1.195 | 62.92% |

The apparent increase in `tail-free parents` at large prefixes is caused by
low-degree records having no tail, not by a successful certificate. The
suffix certificate saved one edge in the entire prefix-16 sample and zero at
the other prefix sizes.

### The apparent 53% byte reduction is fixed-record padding

The sampled records contain 46.84 live edges on average despite capacity 102.
Reading a packed header plus exactly the live RemotePtrs would reduce sampled
graph payload by 53.04% before applying any query-dependent certificate.

This must not be reported as certificate benefit. Once padding is removed, the
implementable certificate contributes approximately zero additional byte
savings. Even the impossible perfect suffix oracle skips at most 10.59% of all
live edges (prefix 16).

### Extra WQEs make the systems result worse

The trace-off C16/c256 reference processes approximately:

- 195.19 graph READs/query;
- 128 exact-vector READs/query;
- 323.19 total READ WQEs/query;
- 58.68K QPS, or about 18.96M READ WQEs/s.

The least harmful measured suffix choice, prefix 64, needs 1.195 graph
WQEs/parent and approximately 361.28 total WQEs/query. Reaching 20% higher QPS
would require about 25.44M WQEs/s, 1.34 times the already observed rate, before
charging the second dependent RTT, certificate compute or validation.

The current checksum also covers all 832 bytes. A real partial-read layout
would require new version/per-group validation or immutable publication, which
can only make this result worse.

## Phase-gate result

| gate | required | observed | result |
|---|---:|---:|---|
| perfect oracle skips live graph payload | >=70% | <=10.59% | fail |
| feasible certificate retains perfect benefit | >=80% | ~0% | fail |
| parents avoiding a tail READ | >=92% | <=80.49%, almost entirely no-tail records | fail |
| geometric bound violations | 0 | 0 | pass |
| certificate-only modeled query gain | >=20% | ~0% | fail |
| WQE rate for +20% QPS | close to observed | 1.34x observed at best | fail |

Therefore a size-dependent RDMA microbenchmark or a partial-record prototype
cannot reverse the conclusion: the implementable predicate certifies almost
nothing, while even the perfect oracle is too weak at the parent/round level.

## Artifacts

- Trace/result directory:
  `motivation/results/adjacency_certificate_smoke/concurrency_256/repeat_1/`
- Machine-readable analysis:
  `adjacency_oracle_analysis.json`
- Rendered analysis:
  `adjacency_oracle_analysis.md`
- Analyzer:
  `motivation/analyze_adjacency_oracle.py`
- Reproduction driver:
  `motivation/run_adjacency_certificate_motivation.sh`

