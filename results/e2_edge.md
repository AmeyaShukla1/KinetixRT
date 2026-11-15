| controller | episodes | success (95% CI) | inferences/ep | denoise steps/ep | mean steps | model s/ep (wall) | steps to success | deadline-miss ticks/ep | mean action age (ticks) | smoothness |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 20 | 30.0% (15-52) | 6.1 | 61 | 10.0 | 19.3 | 202 | 88.0 | 51.4 | 0.122 |
| `fixed_reduced_steps` | 20 | 35.0% (18-57) | 8.6 | 26 | 3.0 | 18.9 | 158 | 40.8 | 41.2 | 0.115 |
| `physical_aware` | 20 | 20.0% (8-42) | 8.9 | 30 | 3.5 | 19.9 | 168 | 45.9 | 41.3 | 0.116 |
| `runtime_only` | 20 | 30.0% (15-52) | 8.9 | 34 | 4.0 | 20.1 | 162 | 40.9 | 42.5 | 0.117 |

Paired comparison against `fixed_high` (same task, initial state and noise seed):

| controller | pairs | success diff (pp) | wins / losses | McNemar p | inference ratio (95% CI) | denoise-step ratio (95% CI) | model-time ratio (95% CI) |
|---|---|---|---|---|---|---|---|
| `fixed_reduced_steps` | 20 | +5.0 | 3 / 2 | 1.00 | 1.41 (1.17-1.64) | 0.42 (0.35-0.49) | 0.98 (0.80-1.16) |
| `physical_aware` | 20 | -10.0 | 2 / 4 | 0.69 | 1.46 (1.25-1.69) | 0.50 (0.43-0.57) | 1.03 (0.88-1.20) |
| `runtime_only` | 20 | +0.0 | 3 / 3 | 1.00 | 1.46 (1.22-1.70) | 0.55 (0.48-0.63) | 1.04 (0.87-1.22) |
