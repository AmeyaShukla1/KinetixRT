| controller | episodes | success (95% CI) | inferences/ep | denoise steps/ep | mean steps | model s/ep (wall) | steps to success | deadline-miss ticks/ep | mean action age (ticks) | smoothness |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 10 | 50.0% (24-76) | 19.1 | 191 | 10.0 | 60.0 | 97 | 0.0 | 4.5 | 0.166 |
| `fixed_matched` | 10 | 60.0% (31-83) | 11.3 | 113 | 10.0 | 36.2 | 119 | 0.0 | 7.8 | 0.140 |

Paired comparison against `fixed_high` (same task, initial state and noise seed):

| controller | pairs | success diff (pp) | wins / losses | McNemar p | inference ratio (95% CI) | denoise-step ratio (95% CI) | model-time ratio (95% CI) |
|---|---|---|---|---|---|---|---|
| `fixed_matched` | 10 | +10.0 | 2 / 1 | 1.00 | 0.59 (0.47-0.76) | 0.59 (0.47-0.76) | 0.60 (0.48-0.77) |
