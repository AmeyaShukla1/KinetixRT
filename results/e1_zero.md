| controller | episodes | success (95% CI) | inferences/ep | denoise steps/ep | mean steps | model s/ep (wall) | steps to success | deadline-miss ticks/ep | mean action age (ticks) | smoothness |
|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_high` | 23 | 47.8% (29-67) | 19.5 | 195 | 10.0 | 71.6 | 99 | 0.0 | 4.5 | 0.160 |
| `fixed_long_horizon` | 23 | 17.4% (7-37) | 5.4 | 54 | 10.0 | 20.9 | 100 | 0.0 | 22.9 | 0.123 |
| `fixed_matched` | 20 | 65.0% (43-82) | 10.9 | 109 | 10.0 | 42.7 | 124 | 0.0 | 7.8 | 0.152 |
| `fixed_reduced_steps` | 23 | 52.2% (33-71) | 18.8 | 56 | 3.0 | 49.4 | 99 | 0.0 | 4.5 | 0.134 |
| `physical_aware` | 23 | 52.2% (33-71) | 13.3 | 117 | 8.7 | 46.4 | 100 | 0.0 | 8.4 | 0.148 |
| `physical_precise` | 10 | 40.0% (17-69) | 27.1 | 260 | 9.5 | 96.3 | 88 | 0.0 | 6.4 | 0.185 |

Paired comparison against `fixed_high` (same task, initial state and noise seed):

| controller | pairs | success diff (pp) | wins / losses | McNemar p | inference ratio (95% CI) | denoise-step ratio (95% CI) | model-time ratio (95% CI) |
|---|---|---|---|---|---|---|---|
| `fixed_long_horizon` | 23 | -30.4 | 2 / 9 | 0.07 | 0.28 (0.23-0.35) | 0.28 (0.23-0.35) | 0.29 (0.24-0.37) |
| `fixed_matched` | 20 | +15.0 | 6 / 3 | 0.51 | 0.57 (0.44-0.76) | 0.57 (0.44-0.76) | 0.59 (0.45-0.81) |
| `fixed_reduced_steps` | 23 | +4.3 | 1 / 0 | 1.00 | 0.96 (0.90-1.00) | 0.29 (0.27-0.30) | 0.69 (0.63-0.75) |
| `physical_aware` | 23 | +4.3 | 5 / 4 | 1.00 | 0.68 (0.52-0.88) | 0.60 (0.45-0.78) | 0.65 (0.50-0.84) |
| `physical_precise` | 10 | -10.0 | 1 / 2 | 1.00 | 1.44 (0.99-1.99) | 1.39 (0.94-1.93) | 1.54 (1.02-2.14) |
