| controller | episodes | success (95% CI) | inferences/ep | denoise steps/ep | mean steps | model s/ep (wall) | steps to success | deadline-miss ticks/ep | mean action age (ticks) | smoothness |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 3 | 33.3% (6-79) | 6.3 | 63 | 10.0 | 22.0 | 166 | 69.3 | 44.1 | 0.121 |
| `physical_aware` | 3 | 33.3% (6-79) | 8.3 | 29 | 3.5 | 19.1 | 139 | 46.0 | 37.4 | 0.125 |

Paired comparison against `fixed_high` (same task, initial state and noise seed):

| controller | pairs | success diff (pp) | wins / losses | McNemar p | inference ratio (95% CI) | denoise-step ratio (95% CI) | model-time ratio (95% CI) |
|---|---|---|---|---|---|---|---|
| `physical_aware` | 3 | +0.0 | 0 / 0 | 1.00 | 1.32 (1.00-1.38) | 0.46 (0.37-0.49) | 0.87 (0.72-0.94) |
