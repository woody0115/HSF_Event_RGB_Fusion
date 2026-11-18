| tag | n | all | day | dim | night | night ped | val-mPC | train kinds | held-out | noise | blur | weather | digital | light |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_single (single_evt) | 1 | 29.9 | 29.6 | 39.1 | 22.9 | 8.9 | - | - | - | - | - | - | - | - |
| test_single (single_rgb) | 1 | 50.1 | 50.2 | 60.4 | 36.9 | 40.6 | 17.2 | 21.6 | 11.6 | 4.6 | 14.1 | 17.8 | 29.6 | 35.4 |
| test_base_sw2 | 3 | 51.4 ± 0.3 | 51.6 ± 0.4 | 60.6 ± 0.3 | 38.1 ± 1.2 | 34.9 ± 1.4 | 20.2 ± 0.7 | 24.6 ± 0.5 | 14.9 ± 1.0 | 6.5 ± 1.1 | 17.9 ± 0.8 | 22.8 ± 0.7 | 33.0 ± 0.8 | 38.2 ± 0.7 |
| test_ft5c_sw2 | 3 | 51.0 ± 0.2 | 51.0 ± 0.3 | 60.6 ± 0.1 | 38.8 ± 0.4 | 36.6 ± 1.4 | 36.3 ± 0.3 | 38.1 ± 0.2 | 34.0 ± 0.5 | 31.8 ± 0.7 | 33.7 ± 0.4 | 36.9 ± 0.2 | 41.7 ± 0.2 | 43.6 ± 0.2 |

| tag | r_e clean P3/P4/P5 | r_e corrupted P3/P4/P5 | gate clean | gate corrupted | gate night | P3 g_r clean / noise |
|---|---|---|---|---|---|---|
| test_base_sw2 | 0.92 / 0.82 / 0.87 | 0.93 / 0.86 / 0.81 | 0.202 | 0.199 | 0.201 | - |
| test_ft5c_sw2 | 0.92 / 0.82 / 0.88 | 1.23 / 1.47 / 1.23 | 0.224 | 0.599 | 0.260 | - |

Paired deltas vs test_base_sw2 (mean ± std over checkpoints):

| tag | n | Δall | Δnight | Δnight ped | Δval-mPC | Δheld-out | Δnoise |
|---|---|---|---|---|---|---|---|
| test_ft5c_sw2 | 3 | -0.46 ± 0.28 | 0.69 ± 0.74 | 1.66 ± 1.52 | 16.04 ± 0.50 | 19.17 ± 0.50 | 25.33 ± 0.40 |

| tag | RGB input = 0 (val, every 4th frame) |
|---|---|
| test_single (single_rgb) | 0.0 |
| test_base_sw2 | 0.0 ± 0.0 |
| test_ft5c_sw2 | 29.8 ± 0.3 |
