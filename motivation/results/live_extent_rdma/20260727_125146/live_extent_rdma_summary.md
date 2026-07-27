# Live-extent RDMA transport probe

**Scope: transport-only.** This is a dedicated GPU-initiated, one-sided RDMA READ microbenchmark. It does not measure query QPS, query latency, Beam/decode/PQ/visited work, storage-side format cost, or NIC wire bandwidth. `application_payload_GB_per_s` is requested application bytes divided by probe wall time.

All aggregate cells below are `median / IQR / CV` over repeats. IQR uses inclusive quartiles; CV uses sample standard deviation. Comparison ratios are paired by repeat before aggregation.

## Repeated transport cases

| active QPs | transfer | reps | READ WQE/s | requested-payload GB/s | batch P50 us | batch P99 us |
|---:|---|---:|---:|---:|---:|---:|
| 1 | one-shot 16B | 3 | 1489203 / 34368 / 2.5% | 0.024 / 0.001 / 2.4% | 10.240 / 1.024 / 10.0% | 13.312 / 0.512 / 4.6% |
| 1 | one-shot 80B | 3 | 1469508 / 35945 / 2.5% | 0.118 / 0.003 / 2.5% | 10.240 / 0.512 / 6.0% | 12.288 / 0.000 / 0.0% |
| 1 | one-shot 144B | 3 | 1433949 / 52080 / 3.7% | 0.206 / 0.007 / 3.8% | 10.240 / 0.512 / 5.6% | 12.288 / 0.000 / 0.0% |
| 1 | one-shot 272B | 3 | 1410686 / 34094 / 2.6% | 0.384 / 0.010 / 2.6% | 10.240 / 0.512 / 5.6% | 12.288 / 0.000 / 0.0% |
| 1 | one-shot 400B | 3 | 1408451 / 31775 / 2.5% | 0.563 / 0.013 / 2.5% | 10.240 / 0.512 / 5.6% | 12.288 / 0.512 / 4.7% |
| 1 | one-shot 448B | 3 | 1405728 / 25440 / 1.9% | 0.630 / 0.011 / 1.9% | 10.240 / 0.512 / 5.6% | 12.288 / 0.512 / 4.7% |
| 1 | one-shot 528B | 3 | 1373155 / 110884 / 9.3% | 0.725 / 0.058 / 9.3% | 11.264 / 0.512 / 5.4% | 20.480 / 5.632 / 29.3% |
| 1 | one-shot 832B | 3 | 1351351 / 126074 / 10.7% | 1.124 / 0.105 / 10.7% | 10.240 / 0.512 / 5.6% | 27.648 / 8.192 / 37.9% |
| 1 | dependent 16+400B | 3 | 1457726 / 3659 / 0.3% | 0.303 / 0.001 / 0.2% | 20.480 / 0.512 / 2.8% | 23.552 / 13.824 / 48.7% |
| 1 | dependent 16+448B | 3 | 1460654 / 28186 / 2.1% | 0.339 / 0.007 / 2.0% | 21.504 / 0.512 / 2.8% | 23.552 / 0.512 / 2.5% |
| 8 | one-shot 16B | 3 | 10361017 / 1781575 / 20.2% | 0.166 / 0.028 / 20.3% | 11.264 / 0.000 / 0.0% | 14.336 / 15.872 / 74.4% |
| 8 | one-shot 80B | 3 | 10606563 / 184331 / 1.8% | 0.849 / 0.015 / 1.8% | 11.264 / 0.000 / 0.0% | 13.312 / 0.512 / 4.3% |
| 8 | one-shot 144B | 3 | 10450686 / 826602 / 8.4% | 1.505 / 0.119 / 8.4% | 11.264 / 0.512 / 5.1% | 14.336 / 3.584 / 24.2% |
| 8 | one-shot 272B | 3 | 10261343 / 780149 / 8.2% | 2.791 / 0.212 / 8.2% | 12.288 / 0.512 / 4.9% | 14.336 / 3.584 / 24.7% |
| 8 | one-shot 400B | 3 | 10181356 / 602751 / 6.5% | 4.073 / 0.241 / 6.5% | 12.288 / 0.512 / 4.9% | 14.336 / 5.632 / 35.9% |
| 8 | one-shot 448B | 3 | 10240000 / 1555782 / 18.8% | 4.588 / 0.697 / 18.8% | 12.288 / 0.512 / 4.9% | 14.336 / 15.872 / 73.6% |
| 8 | one-shot 528B | 3 | 10121778 / 566460 / 6.4% | 5.344 / 0.299 / 6.4% | 12.288 / 0.000 / 0.0% | 14.336 / 8.192 / 47.8% |
| 8 | one-shot 832B | 3 | 9850701 / 1030703 / 12.5% | 8.196 / 0.857 / 12.5% | 12.288 / 0.000 / 0.0% | 15.360 / 20.992 / 82.6% |
| 8 | dependent 16+400B | 3 | 10453246 / 336257 / 3.2% | 2.174 / 0.070 / 3.2% | 23.552 / 0.512 / 2.5% | 26.624 / 12.288 / 40.7% |
| 8 | dependent 16+448B | 3 | 10377817 / 363892 / 3.5% | 2.408 / 0.085 / 3.5% | 23.552 / 0.512 / 2.5% | 27.648 / 12.800 / 40.9% |
| 32 | one-shot 16B | 3 | 38300418 / 295608 / 0.8% | 0.613 / 0.004 / 0.7% | 12.288 / 0.512 / 4.7% | 15.360 / 0.000 / 0.0% |
| 32 | one-shot 80B | 3 | 36534893 / 215744 / 0.6% | 2.923 / 0.018 / 0.6% | 13.312 / 0.000 / 0.0% | 16.384 / 0.000 / 0.0% |
| 32 | one-shot 144B | 3 | 35899593 / 1602309 / 5.2% | 5.170 / 0.231 / 5.2% | 13.312 / 0.512 / 4.3% | 16.384 / 2.560 / 16.3% |
| 32 | one-shot 272B | 3 | 34561902 / 3627914 / 13.0% | 9.401 / 0.987 / 13.0% | 14.336 / 0.000 / 0.0% | 17.408 / 10.240 / 48.8% |
| 32 | one-shot 400B | 3 | 27978141 / 45836 / 0.2% | 11.191 / 0.018 / 0.2% | 17.408 / 0.000 / 0.0% | 20.480 / 0.000 / 0.0% |
| 32 | one-shot 448B | 3 | 25224159 / 12422 / 0.1% | 11.300 / 0.005 / 0.0% | 19.456 / 0.000 / 0.0% | 22.528 / 0.000 / 0.0% |
| 32 | one-shot 528B | 3 | 21638069 / 9149 / 0.0% | 11.425 / 0.005 / 0.0% | 23.552 / 0.000 / 0.0% | 26.624 / 0.000 / 0.0% |
| 32 | one-shot 832B | 3 | 13958560 / 592155 / 4.8% | 11.614 / 0.492 / 4.8% | 37.888 / 1.024 / 3.2% | 47.104 / 11.776 / 24.2% |
| 32 | dependent 16+400B | 3 | 35116598 / 30130 / 0.1% | 7.304 / 0.006 / 0.1% | 28.672 / 0.000 / 0.0% | 32.768 / 0.000 / 0.0% |
| 32 | dependent 16+448B | 3 | 35046890 / 213485 / 0.7% | 8.131 / 0.049 / 0.7% | 28.672 / 0.000 / 0.0% | 32.768 / 0.512 / 1.8% |
| 160 | one-shot 16B | 3 | 8107834 / 150606 / 2.0% | 0.130 / 0.002 / 1.8% | 320.512 / 5.120 / 1.8% | 442.368 / 4.608 / 1.1% |
| 160 | one-shot 80B | 3 | 7978359 / 327129 / 4.1% | 0.638 / 0.026 / 4.2% | 317.440 / 11.264 / 3.5% | 419.840 / 19.456 / 5.0% |
| 160 | one-shot 144B | 3 | 18838489 / 253029 / 1.5% | 2.713 / 0.037 / 1.5% | 134.144 / 2.048 / 1.6% | 184.320 / 4.608 / 2.6% |
| 160 | one-shot 272B | 3 | 23085941 / 151598 / 0.7% | 6.279 / 0.042 / 0.7% | 107.520 / 1.024 / 1.0% | 145.408 / 2.048 / 1.5% |
| 160 | one-shot 400B | 3 | 22986442 / 277219 / 1.2% | 9.195 / 0.110 / 1.2% | 109.568 / 1.024 / 0.9% | 144.384 / 2.560 / 1.8% |
| 160 | one-shot 448B | 3 | 22577345 / 104236 / 0.5% | 10.115 / 0.046 / 0.5% | 111.616 / 0.512 / 0.5% | 147.456 / 2.048 / 1.5% |
| 160 | one-shot 528B | 3 | 21097046 / 305778 / 1.5% | 11.139 / 0.161 / 1.5% | 118.784 / 1.024 / 0.9% | 159.744 / 12.800 / 8.1% |
| 160 | one-shot 832B | 3 | 13914859 / 143360 / 1.1% | 11.577 / 0.120 / 1.1% | 181.248 / 0.512 / 0.3% | 230.400 / 38.912 / 17.3% |
| 160 | dependent 16+400B | 3 | 20900519 / 133027 / 0.7% | 4.347 / 0.027 / 0.7% | 238.592 / 1.536 / 0.7% | 328.704 / 0.512 / 0.2% |
| 160 | dependent 16+448B | 3 | 22435870 / 499691 / 2.3% | 5.205 / 0.116 / 2.3% | 221.184 / 4.096 / 1.9% | 335.872 / 25.600 / 8.8% |

## One-shot 400/448B versus fixed 832B

Each paired case posts the same number of READ WQEs. A WQE/s ratio above 1 means the shorter payload completed more READs per second; latency ratios below 1 are better. Application GB/s is shown separately because its numerator changes with payload size.

| active QPs | payload | payload reduction | READ WQE count | READ WQE/s ratio | batch P50 ratio | batch P99 ratio | application GB/s ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400B | 51.9% | 1.000x (IQR 0.000, CV 0.0%) | 1.042x (IQR 0.142, CV 14.0%) | 1.000x (IQR 0.095, CV 9.5%) | 0.444x (IQR 0.293, CV 53.3%) | 0.501x (IQR 0.068, CV 14.0%) |
| 1 | 448B | 46.2% | 1.000x (IQR 0.000, CV 0.0%) | 1.029x (IQR 0.130, CV 13.2%) | 1.000x (IQR 0.000, CV 0.0%) | 0.481x (IQR 0.255, CV 45.6%) | 0.554x (IQR 0.069, CV 13.2%) |
| 8 | 400B | 51.9% | 1.000x (IQR 0.000, CV 0.0%) | 1.043x (IQR 0.065, CV 6.7%) | 1.000x (IQR 0.042, CV 4.9%) | 0.933x (IQR 0.243, CV 36.5%) | 0.502x (IQR 0.031, CV 6.7%) |
| 8 | 448B | 46.2% | 1.000x (IQR 0.000, CV 0.0%) | 1.040x (IQR 0.061, CV 6.8%) | 1.000x (IQR 0.042, CV 4.9%) | 0.933x (IQR 0.065, CV 8.4%) | 0.560x (IQR 0.033, CV 6.8%) |
| 32 | 400B | 51.9% | 1.000x (IQR 0.000, CV 0.0%) | 2.007x (IQR 0.093, CV 5.0%) | 0.459x (IQR 0.013, CV 3.2%) | 0.435x (IQR 0.081, CV 21.5%) | 0.965x (IQR 0.045, CV 5.0%) |
| 32 | 448B | 46.2% | 1.000x (IQR 0.000, CV 0.0%) | 1.807x (IQR 0.083, CV 5.0%) | 0.514x (IQR 0.015, CV 3.2%) | 0.478x (IQR 0.089, CV 21.5%) | 0.973x (IQR 0.045, CV 5.0%) |
| 160 | 400B | 51.9% | 1.000x (IQR 0.000, CV 0.0%) | 1.669x (IQR 0.011, CV 0.7%) | 0.605x (IQR 0.004, CV 0.7%) | 0.613x (IQR 0.078, CV 14.6%) | 0.802x (IQR 0.005, CV 0.7%) |
| 160 | 448B | 46.2% | 1.000x (IQR 0.000, CV 0.0%) | 1.617x (IQR 0.022, CV 1.4%) | 0.616x (IQR 0.005, CV 0.7%) | 0.635x (IQR 0.077, CV 14.9%) | 0.871x (IQR 0.012, CV 1.4%) |

## Dependent 16B header + body versus corresponding one-shot body

The dependent case waits for the 16B header stage before issuing the body stage. It carries 16 additional application bytes versus the listed one-shot reference, so this is deliberately reported as a measured transport penalty rather than a byte-identical comparison.
All two-stage cases execute after the one-shot list even in reverse repeats; this comparison is paired but not fully order-counterbalanced and can retain temporal drift.

| active QPs | dependent/reference | READ WQE count penalty | READ WQE/s ratio | logical batch/s ratio | batch P50 penalty | batch P99 penalty | application GB/s ratio |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 16+400B / 400B | 2.000x (IQR 0.000, CV 0.0%) | 1.035x (IQR 0.025, CV 2.6%) | 0.517x (IQR 0.013, CV 2.6%) | 10.240 us; 2.000x (IQR 0.045, CV 2.7%) | 11.264 us; 1.917x (IQR 1.199, CV 51.3%) | 0.538x (IQR 0.013, CV 2.6%) |
| 1 | 16+448B / 448B | 2.000x (IQR 0.000, CV 0.0%) | 1.047x (IQR 0.008, CV 0.8%) | 0.524x (IQR 0.004, CV 0.8%) | 10.240 us; 2.000x (IQR 0.095, CV 4.8%) | 10.240 us; 1.833x (IQR 0.074, CV 4.0%) | 0.542x (IQR 0.005, CV 0.9%) |
| 8 | 16+400B / 400B | 2.000x (IQR 0.000, CV 0.0%) | 1.039x (IQR 0.072, CV 7.0%) | 0.519x (IQR 0.036, CV 7.0%) | 11.264 us; 1.917x (IQR 0.042, CV 2.5%) | 11.264 us; 1.786x (IQR 1.230, CV 59.8%) | 0.540x (IQR 0.037, CV 7.0%) |
| 8 | 16+448B / 448B | 2.000x (IQR 0.000, CV 0.0%) | 1.033x (IQR 0.220, CV 21.0%) | 0.516x (IQR 0.110, CV 21.0%) | 11.264 us; 1.917x (IQR 0.042, CV 2.5%) | 11.264 us; 1.786x (IQR 1.486, CV 75.3%) | 0.535x (IQR 0.114, CV 21.0%) |
| 32 | 16+400B / 400B | 2.000x (IQR 0.000, CV 0.0%) | 1.255x (IQR 0.003, CV 0.3%) | 0.628x (IQR 0.002, CV 0.3%) | 11.264 us; 1.647x (IQR 0.000, CV 0.0%) | 12.288 us; 1.600x (IQR 0.000, CV 0.0%) | 0.653x (IQR 0.002, CV 0.3%) |
| 32 | 16+448B / 448B | 2.000x (IQR 0.000, CV 0.0%) | 1.389x (IQR 0.008, CV 0.6%) | 0.695x (IQR 0.004, CV 0.6%) | 9.216 us; 1.474x (IQR 0.000, CV 0.0%) | 10.240 us; 1.455x (IQR 0.023, CV 1.8%) | 0.720x (IQR 0.004, CV 0.6%) |
| 160 | 16+400B / 400B | 2.000x (IQR 0.000, CV 0.0%) | 0.908x (IQR 0.016, CV 1.8%) | 0.454x (IQR 0.008, CV 1.8%) | 129.024 us; 2.178x (IQR 0.034, CV 1.6%) | 184.320 us; 2.277x (IQR 0.037, CV 1.7%) | 0.472x (IQR 0.008, CV 1.8%) |
| 160 | 16+448B / 448B | 2.000x (IQR 0.000, CV 0.0%) | 0.994x (IQR 0.018, CV 1.9%) | 0.497x (IQR 0.009, CV 1.9%) | 109.568 us; 1.982x (IQR 0.028, CV 1.6%) | 188.416 us; 2.278x (IQR 0.205, CV 9.9%) | 0.515x (IQR 0.009, CV 1.9%) |

## Interpretation boundary

A shorter one-shot READ improving this table establishes only that fixed 832B records leave transport headroom in this isolated access pattern. A dependent two-stage penalty shows how much of that headroom is lost when the live length is unavailable before the first READ. Neither result proves that changing the index layout improves dvstor end to end; that requires a query-path A/B with identical Recall and graph-read semantics.
