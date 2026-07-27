# Feedback-Priced Expansion Motivation Test

## Question

The test evaluates the proposed intuition:

> A large amount of newly inserted mass near the front of the Beam means that
> feedback is changing the search rapidly, so the next expansion batch should
> become narrower; a stable Beam should use a wider batch.

It does not implement another controller. It first asks whether Beam turnover
predicts parent value, and whether an unrealistically strong hindsight oracle
has enough headroom to beat the fixed-C16 stable-run baseline.

## Instrumentation

The sampled diagnostic kernel records, for every traced round:

- the 32 highest-ranked unexpanded post-merge candidates;
- whether each candidate came from the old Beam or the current round;
- which of those candidates is selected in a later round;
- the current round's selected parent handles;
- whether each selected parent produces at least one post-visited finite child
  that survives in the authoritative Beam;
- graph, score, and Beam cycles excluding the deliberate oracle work.

The probe is enabled only by the existing sampled RDMA trace mode. The default
kernel and search state are unchanged. Trace throughput and tail latency are
not performance results: the diagnostic kernel intentionally performs costly
membership scans and uses 170 registers/thread versus 140 in the default
kernel.

Analysis uses a **productive-suffix oracle**. For each fixed-C16 round, it
chooses the smallest width in `{1,4,8,12,16}` that includes the last parent
known in hindsight to contribute a child to the immediate Beam. It linearly
removes graph/score work for the suffix, but charges one observed Beam merge for
every extra feedback round. This remains optimistic because it does not:

- replay the changed visited set or future traversal;
- charge the extra RDMA latency of smaller batches;
- account for a skipped parent's possible delayed contribution.

If this oracle cannot win, a causal online signal cannot reasonably do better.

## Runs

Dataset/index: SIFT100M schema-16, OPQ/PQ32, traversal Beam 128, fixed C16,
stable-run Beam merge, 32 QPs per storage node.

| concurrency | sampled queries | sampled rounds | turnover mean | P50 | P90 |
|---:|---:|---:|---:|---:|---:|
| 64 | 101 | 1,478 | 0.742 | 1.000 | 1.000 |
| 256 | 462 | 6,764 | 0.735 | 1.000 | 1.000 |

`turnover` is the fraction of new candidates among the first 16 unexpanded
post-merge candidates.

## Observation 1: the one-point feedback interpretation is incomplete

For high-turnover rounds (`turnover >= 0.75`), candidates just outside the
fixed-C16 boundary behave very differently:

| concurrency | rank 16–31 origin | eventually expanded |
|---:|---|---:|
| 64 | new | 23.00% |
| 64 | old | 85.33% |
| 256 | new | 23.13% |
| 256 | old | 86.78% |

Thus a new candidate near the front does **not** imply that the old reservoir
behind it has become worthless. The exact opposite is observed outside the
immediate batch: old candidates remain durable, while the fresh tail is often
evicted. A scalar such as the earliest-new position loses this distinction.

At the same time, the first 16 fresh candidates are not noise. Under high
turnover, 87.7%–88.2% of selected fresh parents immediately contribute at
least one child that remains in the authoritative Beam. A controller that
narrows precisely in these rounds delays highly productive work.

## Observation 2: there is almost no removable prefix suffix inside C16

Selected-parent immediate productivity decreases only mildly with rank:

| selected rank | productivity at concurrency 256 |
|---:|---:|
| 0 | 73.60% |
| 4 | 69.74% |
| 8 | 67.52% |
| 12 | 66.78% |
| 15 | 66.74% |

Because expansion must preserve Beam order, isolated unproductive parents in
the middle cannot be removed by choosing a shorter prefix. The hindsight
oracle can remove only:

| concurrency | removable selected-parent suffix | round work after merge toll |
|---:|---:|---:|
| 64 | 5.24% | **+6.37%** |
| 256 | 5.32% | **+6.85%** |

The extra feedback/merge cost is larger than all graph and scoring work saved
by perfect suffix knowledge. Additional small-batch RDMA waiting, omitted by
the oracle, would make the result worse.

Projecting the concurrency-256 result onto the uninstrumented stable-run C16
baseline (58.68 KQPS, 4.279 ms GPU query time) predicts a 5.86% regression, or
about 55.43 KQPS. This is an upper-bound projection, not a measured dynamic
policy result.

## Conclusion

The experiment rejects the current dynamic-batch premise for this system:

1. earliest-new position is not a sufficient summary of Beam turnover;
2. high turnover does not mean the next C16 prefix is low-value;
3. fixed C16 already prevents the low-value fresh tail at ranks 16–31 from
   being expanded immediately;
4. even a hindsight productive-suffix oracle cannot pay for the additional
   feedback rounds.

Therefore another Feedback Horizon / Hardware Hunger controller should **not**
be implemented on this evidence. Hardware pressure cannot manufacture
algorithmic headroom when the legal C16 prefix is already productive, and
widening beyond it reintroduces the known stale-expansion problem.

This result narrows the search for a query-side contribution: the next design
must remove or overlap a cost that exists *within the useful C16 work*, rather
than trying to choose a better prefix width.

## Reproduction

```bash
CONCURRENCIES="64 256" \
  ./motivation/run_feedback_pricing_motivation.sh
```

Outputs:

- `motivation/results/feedback_pricing/<run>/beam_turnover.jsonl`
- `motivation/results/feedback_pricing/<run>/analysis.json`
- `motivation/results/feedback_pricing/<run>/analysis.md`

The concurrency-256 smoke dataset used in this report is under
`motivation/results/feedback_pricing_smoke/concurrency_256/repeat_1`.
