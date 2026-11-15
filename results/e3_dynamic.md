| controller | episodes | success (95% CI) | inferences/ep | denoise steps/ep | mean steps | model s/ep (wall) | steps to success | deadline-miss ticks/ep | mean action age (ticks) | smoothness |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 6 | 16.7% (3-56) | 25.0 | 250 | 10.0 | 91.9 | 96 | 0.0 | 4.5 | 0.171 |
| `physical_aware` | 5 | 0.0% (0-43) | 20.6 | 188 | 9.1 | 73.4 | nan | 0.0 | 7.9 | 0.157 |
| `physical_no_scene` | 4 | 0.0% (0-49) | 20.0 | 179 | 8.9 | 73.6 | nan | 0.0 | 8.0 | 0.157 |

Paired comparison against `fixed_high` (same task, initial state and noise seed):

| controller | pairs | success diff (pp) | wins / losses | McNemar p | inference ratio (95% CI) | denoise-step ratio (95% CI) | model-time ratio (95% CI) |
|---|---|---|---|---|---|---|---|
| `physical_aware` | 5 | +0.0 | 0 / 0 | 1.00 | 0.74 (0.72-0.75) | 0.67 (0.66-0.69) | 0.71 (0.68-0.74) |
| `physical_no_scene` | 4 | +0.0 | 0 / 0 | 1.00 | 0.71 (0.64-0.77) | 0.64 (0.56-0.70) | 0.70 (0.64-0.76) |
