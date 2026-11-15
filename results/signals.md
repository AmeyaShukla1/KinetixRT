318 samples from 9 episodes (5 successful).

Spearman rank correlation between runtime signals (rows) and measured error proxies (columns):

| signal | sens3 (3 vs 10 steps) | sens5 (5 vs 10 steps) | drift_5 (old plan vs fresh) | drift_10 (old plan vs fresh) | drift_15 (old plan vs fresh) | drift_20 (old plan vs fresh) |
|---|---|---|---|---|---|---|
| event proximity (1 - ticks_to_event/25) | +0.44 | +0.44 | +0.40 | +0.39 | +0.30 | +0.36 |
| ticks since last event (neg.) | +0.31 | +0.33 | +0.35 | +0.38 | +0.43 | +0.39 |
| planned speed | -0.06 | -0.10 | -0.09 | -0.08 | -0.11 | -0.09 |
| state step (proprio speed) | +0.02 | +0.03 | +0.00 | -0.08 | -0.06 | -0.11 |
| image step: scene cam | -0.06 | -0.08 | -0.16 | -0.25 | -0.28 | -0.29 |
| image step: wrist cam | -0.11 | -0.13 | -0.24 | -0.17 | -0.18 | -0.13 |
| sample disagreement (2 noise draws) | +0.73 | +0.72 | +0.62 | +0.58 | +0.58 | +0.54 |

Error proxies by plan phase (mean translational action error, action units; n in brackets):

| phase | n | sens3 | sens5 | sample_dis | drift_5 | drift_10 | drift_15 | drift_20 | grip flip mismatch 3-vs-10 |
|---|---|---|---|---|---|---|---|---|---|
| event within 12 ticks | 138 | 0.069 | 0.031 | 0.348 | 0.420 | 0.501 | 0.516 | 0.573 | 0.030 |
| event 13-25 ticks away | 47 | 0.050 | 0.023 | 0.263 | 0.256 | 0.315 | 0.324 | 0.376 | 0.000 |
| no event within 25 ticks | 133 | 0.047 | 0.022 | 0.251 | 0.268 | 0.312 | 0.368 | 0.392 | 0.000 |
| within 8 ticks after an event | 103 | 0.065 | 0.030 | 0.337 | 0.391 | 0.482 | 0.520 | 0.575 | 0.024 |
