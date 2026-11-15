| controller | episodes | episodes with a plan invalidation | invalidations/ep | hold ticks/ep | stale results rejected/ep | startup wait ticks | starved+expired ticks/ep | chunk-sourced ticks | fast path p50 / max-of-p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 3 | 0 | 0.00 | 0.0 | 1.00 | 27.0 | 69.3 | 60.2% | 1.5 / 5.6 |
| `physical_aware` | 3 | 0 | 0.00 | 0.0 | 0.00 | 20.0 | 46.0 | 71.7% | 1.5 / 5.9 |

Phase tick fractions (controllers that classify phases):

| controller | calm | default | critical |
|---|---|---|---|
| `physical_aware` | 55% | 0% | 45% |

Successes per task (successes / episodes):

| controller | t0 | t1 | t2 |
|---|---|---|---|
| `fixed_high` | 1/1 | 0/1 | 0/1 |
| `physical_aware` | 1/1 | 0/1 | 0/1 |
