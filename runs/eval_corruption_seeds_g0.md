# RGB corruption robustness (DSEC-Det test, every n-th frame; event input clean)

mAP50-95 (%). mPC = mean over corruptions x severities, rPC = mPC / clean. Event share = mean gate of gated fusions.

| model | clean | mPC | rPC | noise | blur | weather | digital | light |
|---|---|---|---|---|---|---|---|---|
| add_seed1 | 49.32 | 22.62 | 0.4586 | 8.89 | 20.43 | 27.02 | 34.88 | 39.22 |
| add_seed2 | 51.23 | 21.84 | 0.4263 | 7.3 | 19.78 | 26.17 | 34.59 | 39.78 |
| relp2_add_seed1 | 50.75 | 23.09 | 0.455 | 9.28 | 21.04 | 27.52 | 34.97 | 40.4 |
| cafr_seed1 | 50.94 | 23.51 | 0.4615 | 8.68 | 22.22 | 28.07 | 36.3 | 40.44 |

| model | corruption | s1 | s2 | s3 | s4 | s5 | event share s1..s5 |
|---|---|---|---|---|---|---|---|
| add_seed1 | gaussian_noise | 14.11 | 9.55 | 3.47 | 2.5 | 2.17 | - |
| add_seed1 | shot_noise | 29.43 | 14.49 | 9.94 | 8.37 | 3.38 | - |
| add_seed1 | impulse_noise | 14.61 | 8.32 | 7.24 | 2.7 | 3.14 | - |
| add_seed1 | defocus_blur | 30.59 | 25.3 | 17.6 | 14.21 | 12.45 | - |
| add_seed1 | motion_blur | 36.39 | 26.84 | 17.76 | 12.79 | 10.42 | - |
| add_seed1 | fog | 33.46 | 30.92 | 26.88 | 24.1 | 19.74 | - |
| add_seed1 | brightness | 48.31 | 44.28 | 40.58 | 36.87 | 33.68 | - |
| add_seed1 | contrast | 39.12 | 35.79 | 31.21 | 22.4 | 16.56 | - |
| add_seed1 | dark | 48.03 | 46.04 | 43.36 | 35.06 | 23.6 | - |
| add_seed2 | gaussian_noise | 12.62 | 7.58 | 2.73 | 1.85 | 1.38 | - |
| add_seed2 | shot_noise | 27.22 | 12.89 | 8.17 | 3.82 | 1.92 | - |
| add_seed2 | impulse_noise | 12.84 | 7.77 | 5.34 | 1.79 | 1.53 | - |
| add_seed2 | defocus_blur | 30.46 | 24.41 | 16.17 | 13.31 | 12.11 | - |
| add_seed2 | motion_blur | 36.17 | 25.97 | 17.12 | 12.32 | 9.79 | - |
| add_seed2 | fog | 32.71 | 29.98 | 25.8 | 23.12 | 19.26 | - |
| add_seed2 | brightness | 49.61 | 44.85 | 40.27 | 36.24 | 33.12 | - |
| add_seed2 | contrast | 38.84 | 34.94 | 30.19 | 21.55 | 16.32 | - |
| add_seed2 | dark | 49.34 | 46.94 | 43.74 | 35.33 | 23.54 | - |
| relp2_add_seed1 | gaussian_noise | 15.02 | 10.22 | 3.57 | 2.64 | 1.5 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | shot_noise | 29.26 | 15.19 | 10.58 | 3.71 | 2.18 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | impulse_noise | 16.42 | 11.7 | 12.02 | 3.16 | 2.05 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | defocus_blur | 31.92 | 26.33 | 18.19 | 14.43 | 12.32 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | motion_blur | 37.24 | 27.5 | 18.31 | 13.73 | 10.41 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | fog | 33.33 | 30.97 | 27.29 | 24.95 | 21.08 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | brightness | 48.84 | 44.05 | 39.88 | 36.26 | 33.43 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | contrast | 39.79 | 36.32 | 31.55 | 22.92 | 16.7 | 0.89 0.89 0.89 0.89 0.89 |
| relp2_add_seed1 | dark | 48.96 | 46.77 | 44.06 | 36.5 | 25.71 | 0.89 0.89 0.89 0.89 0.89 |
| cafr_seed1 | gaussian_noise | 15.0 | 9.36 | 3.38 | 2.12 | 1.51 | - |
| cafr_seed1 | shot_noise | 28.34 | 16.06 | 12.22 | 3.29 | 2.6 | - |
| cafr_seed1 | impulse_noise | 14.84 | 9.45 | 7.19 | 2.64 | 2.16 | - |
| cafr_seed1 | defocus_blur | 32.59 | 27.62 | 19.92 | 16.32 | 14.34 | - |
| cafr_seed1 | motion_blur | 37.56 | 27.9 | 19.37 | 14.75 | 11.84 | - |
| cafr_seed1 | fog | 34.73 | 31.93 | 27.69 | 25.12 | 20.86 | - |
| cafr_seed1 | brightness | 49.68 | 45.51 | 41.46 | 38.05 | 35.58 | - |
| cafr_seed1 | contrast | 40.8 | 37.64 | 33.04 | 23.31 | 17.9 | - |
| cafr_seed1 | dark | 49.29 | 47.04 | 44.0 | 36.32 | 25.57 | - |
