| controller | episodes | episodes with a plan invalidation | invalidations/ep | hold ticks/ep | stale results rejected/ep | startup wait ticks | starved+expired ticks/ep | chunk-sourced ticks | fast path p50 / max-of-p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 6 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.6 / 5.8 |
| `physical_aware` | 5 | 5 | 1.00 | 1.0 | 0.00 | 0.0 | 0.0 | 99.6% | 1.5 / 5.6 |
| `physical_no_scene` | 4 | 0 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 100.0% | 1.5 / 5.6 |

Phase tick fractions (controllers that classify phases):

| controller | calm | default | critical |
|---|---|---|---|
| `physical_aware` | 51% | 0% | 48% |
| `physical_no_scene` | 51% | 0% | 49% |

Successes per task (successes / episodes):

| controller | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|---|---|
| `fixed_high` | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 |
| `physical_aware` | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | - |
| `physical_no_scene` | 0/1 | 0/1 | 0/1 | 0/1 | - | - |

Perturbed episodes: scene-surprise response

| controller | perturbed episodes | max surprise median | episodes with invalidation | median ticks from perturbation to first invalidation |
|---|---|---|---|---|
| `fixed_high` | 6 | 0.128 | 0 | nan |
| `physical_aware` | 5 | 0.140 | 5 | 2 |
| `physical_no_scene` | 4 | 0.128 | 0 | nan |
