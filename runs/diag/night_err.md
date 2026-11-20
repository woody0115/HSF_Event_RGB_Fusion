# Error breakdown by lighting and class (val, score ≥ 0.3)

Share of GT boxes (%), mean over checkpoints. found = right class, IoU ≥ 0.5; loc = right class, 0.1 ≤ IoU < 0.5; cls = other class, IoU ≥ 0.5; low = right class and box only below the score threshold; miss = nothing. bg FP = detections ≥ threshold with IoU < 0.1 to every GT, per frame.

## day (2831 frames)

| model | found | loc | cls | low | miss | bg FP / frame |
|---|---|---|---|---|---|---|
| rgb | 79.4 | 3.2 | 4.6 | 10.6 | 2.1 | 0.53 |
| add | 77.5 | 3.4 | 4.5 | 11.7 | 2.8 | 0.46 |
| add_H1 | 77.2 | 3.3 | 4.5 | 12.5 | 2.5 | 0.48 |
| sw2 | 78.1 | 3.6 | 4.7 | 10.9 | 2.8 | 0.47 |
| sw2_C | 76.6 | 3.5 | 4.7 | 12.0 | 3.2 | 0.46 |
| sw2_G3 | 78.0 | 3.3 | 4.6 | 11.6 | 2.4 | 0.48 |

| class | GT | rgb found / cls / low / miss | add found / cls / low / miss | add_H1 found / cls / low / miss | sw2 found / cls / low / miss | sw2_C found / cls / low / miss | sw2_G3 found / cls / low / miss |
|---|---|---|---|---|---|---|---|
| pedestrian | 1179 | 63.0 / 0.3 / 29.6 / 6.0 | 57.2 / 0.7 / 32.2 / 9.0 | 54.7 / 0.9 / 35.5 / 7.9 | 57.8 / 0.8 / 30.4 / 9.7 | 55.5 / 1.0 / 30.8 / 11.6 | 57.9 / 1.0 / 32.7 / 7.4 |
| rider | 78 | 92.3 / 0.0 / 3.8 / 1.3 | 93.6 / 0.4 / 4.3 / 0.0 | 97.0 / 0.0 / 0.9 / 0.0 | 89.3 / 0.0 / 9.8 / 0.0 | 93.6 / 0.0 / 5.6 / 0.0 | 91.5 / 0.0 / 6.8 / 0.0 |
| car | 9535 | 87.9 / 0.2 / 7.4 / 0.9 | 85.9 / 0.4 / 8.3 / 1.4 | 86.0 / 0.4 / 8.7 / 1.1 | 86.7 / 0.3 / 7.6 / 1.2 | 85.3 / 0.4 / 8.7 / 1.4 | 86.7 / 0.2 / 8.0 / 1.1 |
| bus | 491 | 27.3 / 57.4 / 11.2 / 4.1 | 34.3 / 52.5 / 8.6 / 4.7 | 34.4 / 53.5 / 8.2 / 3.9 | 31.5 / 55.7 / 8.7 / 4.1 | 29.7 / 57.1 / 8.5 / 4.7 | 29.3 / 57.0 / 9.6 / 4.1 |
| truck | 522 | 35.6 / 46.0 / 8.2 / 9.8 | 33.3 / 45.1 / 10.7 / 10.5 | 31.5 / 44.4 / 12.3 / 11.6 | 34.1 / 46.4 / 9.1 / 9.8 | 34.4 / 43.9 / 9.5 / 11.5 | 33.0 / 45.4 / 9.9 / 11.2 |
| bicycle | 356 | 54.2 / 4.5 / 33.4 / 7.6 | 50.5 / 3.5 / 37.3 / 8.7 | 49.8 / 4.7 / 38.1 / 7.4 | 52.8 / 4.2 / 32.5 / 10.4 | 49.5 / 5.1 / 35.6 / 9.7 | 51.7 / 4.7 / 36.0 / 7.6 |
| motorcycle | 413 | 67.8 / 5.1 / 16.5 / 2.7 | 67.5 / 3.9 / 18.6 / 3.3 | 68.1 / 2.7 / 19.8 / 3.2 | 69.7 / 4.0 / 16.1 / 2.3 | 65.6 / 3.2 / 21.3 / 3.8 | 68.5 / 3.2 / 18.1 / 2.6 |

Found by size (%; GT small / medium / large in brackets):

| class | GT S / M / L | rgb | add | add_H1 | sw2 | sw2_C | sw2_G3 |
|---|---|---|---|---|---|---|---|
| pedestrian | 727 / 427 / 25 | 52.0 / 80.3 / 88.0 | 45.2 / 75.6 / 89.3 | 40.8 / 76.3 / 88.0 | 45.4 / 77.3 / 88.0 | 42.2 / 76.3 / 88.0 | 44.8 / 78.3 / 89.3 |
| rider | 64 / 14 / 0 | 90.6 / 100.0 / – | 92.2 / 100.0 / – | 96.4 / 100.0 / – | 87.0 / 100.0 / – | 92.2 / 100.0 / – | 89.6 / 100.0 / – |
| car | 6106 / 2707 / 722 | 84.0 / 93.5 / 99.0 | 82.1 / 91.3 / 98.2 | 82.2 / 91.3 / 97.9 | 82.7 / 92.4 / 98.8 | 81.2 / 91.2 / 98.0 | 82.8 / 92.5 / 98.6 |
| bus | 346 / 87 / 58 | 17.3 / 60.9 / 36.2 | 25.6 / 60.9 / 46.0 | 24.8 / 60.9 / 51.7 | 21.8 / 60.9 / 45.4 | 17.9 / 60.9 / 53.4 | 18.0 / 60.9 / 48.9 |
| truck | 133 / 317 / 72 | 9.0 / 36.6 / 80.6 | 8.3 / 33.2 / 80.1 | 7.3 / 30.9 / 79.2 | 8.5 / 34.8 / 78.2 | 9.3 / 34.8 / 79.2 | 8.5 / 32.8 / 79.2 |
| bicycle | 183 / 173 / 0 | 37.2 / 72.3 / – | 34.1 / 67.8 / – | 31.7 / 69.0 / – | 38.6 / 67.8 / – | 36.4 / 63.4 / – | 35.3 / 69.0 / – |
| motorcycle | 144 / 249 / 20 | 45.1 / 78.3 / 100.0 | 42.8 / 79.1 / 100.0 | 42.6 / 80.3 / 100.0 | 46.3 / 80.9 / 100.0 | 38.4 / 78.6 / 100.0 | 47.0 / 78.4 / 100.0 |

Wrong class: GT → predicted (boxes, mean over checkpoints; ≥ 1 shown):

- rgb: bus → car 258, truck → car 239, bus → truck 24, car → truck 15, motorcycle → car 12, bicycle → motorcycle 9, motorcycle → pedestrian 8, bicycle → pedestrian 6
- add: bus → car 237, truck → car 236, car → truck 32, bus → truck 20, motorcycle → pedestrian 10, car → bus 7, motorcycle → car 6, bicycle → pedestrian 6
- add_H1: bus → car 253, truck → car 232, car → truck 25, bus → truck 10, bicycle → motorcycle 9, car → bus 6, bicycle → pedestrian 6, motorcycle → pedestrian 6
- sw2: bus → car 253, truck → car 242, car → truck 24, bus → truck 21, motorcycle → car 10, bicycle → pedestrian 7, motorcycle → pedestrian 6, bicycle → motorcycle 6
- sw2_C: bus → car 261, truck → car 228, car → truck 30, bus → truck 19, bicycle → motorcycle 10, pedestrian → rider 7, motorcycle → car 7, bicycle → pedestrian 6
- sw2_G3: bus → car 264, truck → car 234, car → truck 18, bus → truck 16, motorcycle → car 8, bicycle → motorcycle 8, bicycle → pedestrian 6, motorcycle → pedestrian 5

Background FP per frame by predicted class:

| model | pedestrian | rider | car | bus | truck | bicycle | motorcycle | train |
|---|---|---|---|---|---|---|---|---|
| rgb | 0.032 | 0.009 | 0.430 | 0.001 | 0.018 | 0.007 | 0.034 | 0.000 |
| add | 0.026 | 0.006 | 0.385 | 0.006 | 0.014 | 0.006 | 0.020 | 0.000 |
| add_H1 | 0.021 | 0.006 | 0.404 | 0.002 | 0.013 | 0.004 | 0.029 | 0.000 |
| sw2 | 0.013 | 0.007 | 0.400 | 0.003 | 0.017 | 0.007 | 0.022 | 0.000 |
| sw2_C | 0.014 | 0.006 | 0.384 | 0.001 | 0.021 | 0.005 | 0.024 | 0.000 |
| sw2_G3 | 0.020 | 0.008 | 0.400 | 0.001 | 0.014 | 0.006 | 0.033 | 0.000 |

## dim (1075 frames)

| model | found | loc | cls | low | miss | bg FP / frame |
|---|---|---|---|---|---|---|
| rgb | 81.1 | 2.3 | 1.8 | 13.1 | 1.7 | 0.38 |
| add | 81.0 | 2.2 | 2.8 | 12.1 | 1.8 | 0.35 |
| add_H1 | 82.7 | 2.3 | 2.4 | 11.5 | 1.2 | 0.43 |
| sw2 | 82.6 | 2.4 | 2.2 | 11.1 | 1.8 | 0.35 |
| sw2_C | 82.5 | 2.5 | 2.5 | 11.1 | 1.5 | 0.41 |
| sw2_G3 | 82.5 | 2.5 | 1.9 | 11.7 | 1.5 | 0.40 |

| class | GT | rgb found / cls / low / miss | add found / cls / low / miss | add_H1 found / cls / low / miss | sw2 found / cls / low / miss | sw2_C found / cls / low / miss | sw2_G3 found / cls / low / miss |
|---|---|---|---|---|---|---|---|
| pedestrian | 206 | 39.3 / 2.4 / 51.0 / 6.8 | 31.4 / 3.6 / 59.1 / 6.0 | 35.3 / 5.0 / 56.5 / 3.2 | 31.1 / 4.9 / 57.1 / 6.8 | 37.5 / 4.7 / 52.8 / 4.9 | 40.1 / 5.2 / 50.3 / 4.4 |
| rider | 6 | 0.0 / 0.0 / 66.7 / 33.3 | 0.0 / 16.7 / 50.0 / 33.3 | 0.0 / 16.7 / 50.0 / 33.3 | 0.0 / 0.0 / 77.8 / 22.2 | 0.0 / 11.1 / 72.2 / 16.7 | 0.0 / 5.6 / 72.2 / 22.2 |
| car | 2929 | 85.9 / 1.1 / 9.6 / 0.8 | 85.6 / 2.4 / 8.5 / 1.0 | 87.2 / 1.8 / 7.7 / 0.8 | 87.3 / 1.6 / 7.5 / 1.0 | 86.6 / 2.0 / 7.7 / 1.0 | 86.7 / 1.3 / 8.3 / 0.9 |
| bus | 53 | 32.1 / 22.6 / 43.4 / 1.9 | 72.3 / 10.1 / 17.6 / 0.0 | 79.2 / 10.1 / 10.7 / 0.0 | 75.5 / 11.3 / 13.2 / 0.0 | 76.7 / 11.3 / 11.9 / 0.0 | 65.4 / 13.2 / 21.4 / 0.0 |
| truck | 22 | 54.5 / 31.8 / 13.6 / 0.0 | 59.1 / 21.2 / 18.2 / 1.5 | 45.5 / 22.7 / 31.8 / 0.0 | 65.2 / 25.8 / 9.1 / 0.0 | 72.7 / 24.2 / 3.0 / 0.0 | 62.1 / 24.2 / 13.6 / 0.0 |
| bicycle | 12 | 0.0 / 0.0 / 16.7 / 83.3 | 0.0 / 8.3 / 11.1 / 80.6 | 0.0 / 8.3 / 36.1 / 55.6 | 0.0 / 0.0 / 13.9 / 86.1 | 0.0 / 5.6 / 36.1 / 58.3 | 0.0 / 2.8 / 27.8 / 69.4 |
| motorcycle | 10 | 0.0 / 10.0 / 50.0 / 40.0 | 0.0 / 6.7 / 43.3 / 50.0 | 0.0 / 6.7 / 83.3 / 10.0 | 0.0 / 3.3 / 70.0 / 26.7 | 0.0 / 6.7 / 83.3 / 10.0 | 0.0 / 3.3 / 80.0 / 16.7 |

Found by size (%; GT small / medium / large in brackets):

| class | GT S / M / L | rgb | add | add_H1 | sw2 | sw2_C | sw2_G3 |
|---|---|---|---|---|---|---|---|
| pedestrian | 154 / 51 / 1 | 33.1 / 58.8 / 0.0 | 23.4 / 56.2 / 0.0 | 23.2 / 72.5 / 0.0 | 23.8 / 53.6 / 0.0 | 30.5 / 59.5 / 0.0 | 31.2 / 68.0 / 0.0 |
| rider | 0 / 6 / 0 | – / 0.0 / – | – / 0.0 / – | – / 0.0 / – | – / 0.0 / – | – / 0.0 / – | – / 0.0 / – |
| car | 1567 / 973 / 389 | 78.9 / 91.6 / 99.7 | 77.4 / 93.0 / 99.7 | 79.2 / 94.9 / 99.7 | 79.8 / 94.2 / 99.7 | 78.8 / 93.9 / 99.7 | 79.4 / 93.3 / 99.7 |
| bus | 42 / 6 / 5 | 40.5 / 0.0 / 0.0 | 88.1 / 22.2 / 0.0 | 95.2 / 33.3 / 0.0 | 89.7 / 38.9 / 0.0 | 92.9 / 27.8 / 0.0 | 80.2 / 16.7 / 0.0 |
| truck | 22 / 0 / 0 | 54.5 / – / – | 59.1 / – / – | 45.5 / – / – | 65.2 / – / – | 72.7 / – / – | 62.1 / – / – |
| bicycle | 1 / 11 / 0 | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – |
| motorcycle | 8 / 2 / 0 | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – | 0.0 / 0.0 / – |

Wrong class: GT → predicted (boxes, mean over checkpoints; ≥ 1 shown):

- rgb: car → bus 25, bus → car 8, car → truck 6, truck → bus 4, bus → truck 4, truck → car 3, pedestrian → bicycle 3, pedestrian → rider 2
- add: car → bus 69, bus → car 5, pedestrian → bicycle 3, pedestrian → rider 3, truck → bus 2, truck → car 2, car → pedestrian 1, bicycle → pedestrian 1
- add_H1: car → bus 53, pedestrian → rider 7, bus → car 5, truck → bus 3, truck → car 2, pedestrian → car 2, pedestrian → bicycle 2, bicycle → pedestrian 1
- sw2: car → bus 46, bus → car 6, pedestrian → bicycle 6, truck → car 3, pedestrian → rider 3, truck → bus 2, car → truck 1, car → pedestrian 1
- sw2_C: car → bus 54, bus → car 6, pedestrian → bicycle 5, pedestrian → rider 3, truck → bus 3, truck → car 3, car → pedestrian 2, car → truck 1
- sw2_G3: car → bus 35, bus → car 7, pedestrian → bicycle 7, truck → car 4, pedestrian → rider 2, truck → bus 2, car → pedestrian 2, pedestrian → car 1

Background FP per frame by predicted class:

| model | pedestrian | rider | car | bus | truck | bicycle | motorcycle | train |
|---|---|---|---|---|---|---|---|---|
| rgb | 0.072 | 0.000 | 0.312 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| add | 0.053 | 0.000 | 0.292 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 |
| add_H1 | 0.060 | 0.000 | 0.367 | 0.004 | 0.000 | 0.001 | 0.000 | 0.000 |
| sw2 | 0.055 | 0.000 | 0.294 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 |
| sw2_C | 0.072 | 0.000 | 0.332 | 0.004 | 0.000 | 0.001 | 0.000 | 0.000 |
| sw2_G3 | 0.074 | 0.000 | 0.324 | 0.002 | 0.000 | 0.001 | 0.000 | 0.000 |

## night (2718 frames)

| model | found | loc | cls | low | miss | bg FP / frame |
|---|---|---|---|---|---|---|
| rgb | 80.5 | 5.2 | 0.9 | 10.8 | 2.5 | 0.47 |
| add | 77.1 | 4.9 | 0.8 | 13.8 | 3.4 | 0.41 |
| add_H1 | 80.4 | 4.3 | 0.9 | 12.3 | 2.1 | 0.48 |
| sw2 | 79.2 | 5.2 | 0.8 | 12.0 | 2.8 | 0.46 |
| sw2_C | 80.1 | 4.9 | 0.8 | 11.6 | 2.5 | 0.48 |
| sw2_G3 | 80.4 | 4.9 | 0.7 | 11.7 | 2.2 | 0.47 |

| class | GT | rgb found / cls / low / miss | add found / cls / low / miss | add_H1 found / cls / low / miss | sw2 found / cls / low / miss | sw2_C found / cls / low / miss | sw2_G3 found / cls / low / miss |
|---|---|---|---|---|---|---|---|
| pedestrian | 1068 | 50.7 / 0.2 / 36.3 / 12.1 | 42.0 / 0.3 / 43.4 / 13.6 | 48.4 / 1.2 / 42.4 / 7.3 | 44.5 / 0.2 / 42.0 / 12.6 | 48.9 / 0.2 / 39.0 / 11.2 | 53.7 / 0.3 / 36.4 / 9.1 |
| rider | 203 | 58.1 / 24.1 / 16.7 / 1.0 | 55.8 / 16.9 / 24.1 / 3.1 | 69.0 / 14.0 / 15.8 / 1.1 | 55.0 / 19.7 / 21.8 / 3.4 | 62.1 / 14.4 / 20.9 / 2.5 | 62.7 / 15.4 / 20.2 / 1.6 |
| car | 11445 | 85.5 / 0.2 / 7.4 / 1.2 | 82.5 / 0.2 / 9.9 / 1.9 | 85.3 / 0.1 / 8.5 / 1.2 | 84.7 / 0.2 / 7.9 / 1.3 | 85.0 / 0.2 / 8.0 / 1.3 | 85.0 / 0.1 / 8.2 / 1.2 |
| bus | 30 | 0.0 / 43.3 / 36.7 / 20.0 | 11.1 / 33.3 / 44.4 / 11.1 | 10.0 / 34.4 / 42.2 / 13.3 | 11.1 / 34.4 / 42.2 / 12.2 | 13.3 / 32.2 / 44.4 / 10.0 | 3.3 / 38.9 / 45.6 / 12.2 |
| truck | 4 | 0.0 / 75.0 / 25.0 / 0.0 | 0.0 / 75.0 / 25.0 / 0.0 | 0.0 / 91.7 / 8.3 / 0.0 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 100.0 / 0.0 / 0.0 |
| bicycle | 441 | 44.9 / 5.2 / 31.5 / 14.7 | 38.9 / 6.1 / 34.0 / 17.5 | 43.8 / 8.5 / 32.7 / 11.8 | 38.2 / 6.0 / 35.4 / 16.9 | 43.3 / 6.7 / 32.0 / 14.6 | 42.5 / 5.9 / 34.8 / 13.5 |
| motorcycle | 27 | 29.6 / 33.3 / 37.0 / 0.0 | 23.5 / 23.5 / 53.1 / 0.0 | 58.0 / 23.5 / 18.5 / 0.0 | 24.7 / 29.6 / 45.7 / 0.0 | 45.7 / 37.0 / 17.3 / 0.0 | 35.8 / 29.6 / 34.6 / 0.0 |

Found by size (%; GT small / medium / large in brackets):

| class | GT S / M / L | rgb | add | add_H1 | sw2 | sw2_C | sw2_G3 |
|---|---|---|---|---|---|---|---|
| pedestrian | 667 / 391 / 10 | 41.5 / 66.2 / 50.0 | 34.3 / 55.6 / 26.7 | 37.8 / 66.7 / 43.3 | 39.1 / 54.0 / 30.0 | 42.0 / 61.0 / 33.3 | 44.3 / 69.8 / 46.7 |
| rider | 92 / 87 / 24 | 42.4 / 67.8 / 83.3 | 48.6 / 55.9 / 83.3 | 55.4 / 77.4 / 90.3 | 42.0 / 60.9 / 83.3 | 51.8 / 68.6 / 77.8 | 50.0 / 69.7 / 86.1 |
| car | 6053 / 3869 / 1523 | 76.6 / 93.9 / 99.3 | 71.6 / 93.1 / 99.1 | 75.5 / 95.1 / 99.4 | 75.2 / 94.0 / 99.3 | 75.4 / 94.4 / 99.4 | 75.4 / 94.3 / 99.4 |
| bus | 18 / 0 / 12 | 0.0 / – / 0.0 | 16.7 / – / 2.8 | 16.7 / – / 0.0 | 16.7 / – / 2.8 | 18.5 / – / 5.6 | 5.6 / – / 0.0 |
| truck | 4 / 0 / 0 | 0.0 / – / – | 0.0 / – / – | 0.0 / – / – | 0.0 / – / – | 0.0 / – / – | 0.0 / – / – |
| bicycle | 199 / 208 / 34 | 26.6 / 57.2 / 76.5 | 23.6 / 47.4 / 76.5 | 27.1 / 53.4 / 83.3 | 21.8 / 46.3 / 85.3 | 28.8 / 50.8 / 82.4 | 25.6 / 52.2 / 81.4 |
| motorcycle | 12 / 15 / 0 | 8.3 / 46.7 / – | 5.6 / 37.8 / – | 38.9 / 73.3 / – | 13.9 / 33.3 / – | 27.8 / 60.0 / – | 19.4 / 48.9 / – |

Wrong class: GT → predicted (boxes, mean over checkpoints; ≥ 1 shown):

- rgb: rider → pedestrian 49, bicycle → pedestrian 15, bus → car 13, car → bus 10, bicycle → motorcycle 8, motorcycle → bicycle 7, car → truck 6, truck → car 3
- add: rider → pedestrian 34, car → bus 21, bicycle → motorcycle 12, bicycle → pedestrian 11, bus → car 10, motorcycle → bicycle 5, bicycle → rider 4, car → truck 3
- add_H1: rider → pedestrian 28, bicycle → motorcycle 23, car → bus 17, bus → car 10, pedestrian → rider 10, bicycle → pedestrian 10, motorcycle → bicycle 5, bicycle → rider 5
- sw2: rider → pedestrian 39, car → bus 16, bicycle → motorcycle 13, bicycle → pedestrian 12, bus → car 10, motorcycle → bicycle 7, truck → car 4, bicycle → rider 2
- sw2_C: rider → pedestrian 29, bicycle → motorcycle 19, car → bus 18, bus → car 10, bicycle → pedestrian 9, motorcycle → bicycle 9, truck → car 3, pedestrian → rider 2
- sw2_G3: rider → pedestrian 31, bicycle → motorcycle 14, bus → car 12, bicycle → pedestrian 10, car → bus 9, motorcycle → bicycle 6, truck → car 4, car → truck 3

Background FP per frame by predicted class:

| model | pedestrian | rider | car | bus | truck | bicycle | motorcycle | train |
|---|---|---|---|---|---|---|---|---|
| rgb | 0.043 | 0.000 | 0.424 | 0.002 | 0.000 | 0.003 | 0.001 | 0.000 |
| add | 0.034 | 0.000 | 0.376 | 0.001 | 0.000 | 0.003 | 0.001 | 0.000 |
| add_H1 | 0.041 | 0.001 | 0.427 | 0.002 | 0.000 | 0.006 | 0.004 | 0.000 |
| sw2 | 0.039 | 0.001 | 0.413 | 0.002 | 0.000 | 0.003 | 0.000 | 0.000 |
| sw2_C | 0.049 | 0.001 | 0.425 | 0.003 | 0.000 | 0.005 | 0.000 | 0.000 |
| sw2_G3 | 0.047 | 0.001 | 0.416 | 0.002 | 0.000 | 0.005 | 0.000 | 0.000 |
