| controller | episodes | episodes with a plan invalidation | invalidations/ep | hold ticks/ep | stale results rejected/ep | startup wait ticks | starved+expired ticks/ep | chunk-sourced ticks | fast path p50 / max-of-p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 23 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.6 / 16.7 |
| `fixed_long_horizon` | 23 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.5 / 16.2 |
| `fixed_matched` | 20 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.5 / 5.9 |
| `fixed_reduced_steps` | 23 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.5 / 17.4 |
| `physical_aware` | 23 | 1 | 0.04 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.5 / 18.7 |
| `physical_precise` | 10 | 1 | 0.40 | 0.4 | 0.00 | 0.0 | 0.0 | 99.8% | 6.9 / 16.5 |

Phase tick fractions (controllers that classify phases):

| controller | calm | default | critical |
|---|---|---|---|
| `physical_aware` | 56% | 0% | 44% |
| `physical_precise` | 46% | 0% | 54% |

Successes per task (successes / episodes):

| controller | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 1/3 | 3/3 | 0/3 | 1/2 | 0/2 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 |
| `fixed_long_horizon` | 1/3 | 0/3 | 0/3 | 0/2 | 1/2 | 0/2 | 2/2 | 0/2 | 0/2 | 0/2 |
| `fixed_matched` | 1/2 | 2/2 | 1/2 | 1/2 | 2/2 | 0/2 | 2/2 | 1/2 | 2/2 | 1/2 |
| `fixed_reduced_steps` | 1/3 | 3/3 | 0/3 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 |
| `physical_aware` | 2/3 | 3/3 | 1/3 | 1/2 | 2/2 | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 |
| `physical_precise` | 1/3 | 1/3 | 1/3 | 1/1 | - | - | - | - | - | - |
