| controller | episodes | episodes with a plan invalidation | invalidations/ep | hold ticks/ep | stale results rejected/ep | startup wait ticks | starved+expired ticks/ep | chunk-sourced ticks | fast path p50 / max-of-p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 20 | 0 | 0.00 | 0.0 | 1.00 | 27.0 | 88.0 | 55.2% | 1.4 / 5.5 |
| `fixed_reduced_steps` | 20 | 0 | 0.00 | 0.0 | 0.00 | 17.0 | 40.8 | 75.7% | 1.5 / 5.6 |
| `physical_aware` | 20 | 2 | 0.10 | 3.7 | 0.10 | 20.0 | 45.9 | 73.0% | 1.5 / 5.6 |
| `runtime_only` | 20 | 0 | 0.00 | 0.0 | 0.00 | 27.0 | 40.9 | 72.2% | 1.4 / 5.3 |

Phase tick fractions (controllers that classify phases):

| controller | calm | default | critical |
|---|---|---|---|
| `physical_aware` | 57% | 0% | 43% |

Successes per task (successes / episodes):

| controller | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 1/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 |
| `fixed_reduced_steps` | 1/2 | 1/2 | 0/2 | 0/2 | 1/2 | 0/2 | 2/2 | 1/2 | 0/2 | 1/2 |
| `physical_aware` | 1/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 1/2 |
| `runtime_only` | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 | 0/2 | 2/2 | 1/2 | 0/2 | 0/2 |
