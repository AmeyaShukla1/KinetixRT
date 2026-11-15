| controller | episodes | episodes with a plan invalidation | invalidations/ep | hold ticks/ep | stale results rejected/ep | startup wait ticks | starved+expired ticks/ep | chunk-sourced ticks | fast path p50 / max-of-p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `fixed_h2` | 10 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.7 / 3.5 |
| `fixed_h3` | 10 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.6 / 5.5 |
| `fixed_h5` | 10 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.5 / 2.7 |
| `fixed_high` | 10 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.5 / 5.2 |
| `physical_h1` | 10 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.7 / 2.5 |
| `physical_h1_inverted` | 10 | 1 | 0.10 | 0.1 | 0.00 | 0.0 | 0.0 | 99.9% | 1.8 / 3.8 |
| `physical_h2` | 10 | 1 | 0.10 | 0.1 | 0.00 | 0.0 | 0.0 | 99.9% | 1.5 / 4.2 |
| `reference_replan` | 10 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.8 / 5.6 |

Phase tick fractions (controllers that classify phases):

| controller | calm | default | critical |
|---|---|---|---|
| `physical_h1` | 55% | 0% | 45% |
| `physical_h1_inverted` | 60% | 0% | 40% |
| `physical_h2` | 60% | 0% | 40% |

Successes per task (successes / episodes):

| controller | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_h2` | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `fixed_h3` | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `fixed_h5` | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 |
| `fixed_high` | 1/1 | 1/1 | 0/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| `physical_h1` | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 |
| `physical_h1_inverted` | 1/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `physical_h2` | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `reference_replan` | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
