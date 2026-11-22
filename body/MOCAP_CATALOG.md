# Motion capture catalog (MANN dog, 51 clips)

Every 50 Hz frame named in the footage's regime vocabulary from kinematics alone (hip height, trunk pitch and roll, speed, paw contacts, footfall timing). Written by `catalog_mocap.py`. Capture licence CC BY-NC 4.0: prototyping only.

## Time per regime

| regime | seconds |
|---|---|
| stand | 910 |
| walk | 650 |
| sit | 267 |
| down | 235 |
| crouch | 194 |
| jump | 75 |
| side | 52 |
| run | 52 |
| trot | 20 |
| rear_up | 3 |

## Transitions found

| from → to | count | examples |
|---|---|---|
| walk → stand | 193 | D1_001_KAN01_001 12.2s, D1_004_KAN01_001 12.3s, D1_005_KAN01_001 13.7s |
| stand → walk | 188 | D1_001_KAN01_001 11.2s, D1_001_KAN01_001 25.1s, D1_009_KAN01_001 2.7s |
| crouch → stand | 62 | D1_047z_KAN01_004 9.8s, D1_047z_KAN01_005 0.6s, D1_047z_KAN01_005 3.3s |
| stand → crouch | 45 | D1_025_KAN01_001 9.6s, D1_047z_KAN01_005 4.3s, D1_047z_KAN01_005 10.4s |
| walk → crouch | 37 | D1_047z_KAN01_004 5.0s, D1_047z_KAN01_004 9.2s, D1_047z_KAN01_005 2.7s |
| crouch → walk | 30 | D1_047z_KAN01_004 5.3s, D1_047z_KAN01_005 6.6s, D1_047z_KAN01_005 19.0s |
| sit → walk | 26 | D1_004_KAN01_001 10.9s, D1_006_KAN01_002 15.3s, D1_007_KAN01_001 1.1s |
| walk → run | 23 | D1_010_KAN01_001 6.6s, D1_045_KAN01_001 2.9s, D1_047z_KAN01_002 11.8s |
| sit → crouch | 19 | D1_059_KAN01_001 6.9s, D1_059_KAN01_001 10.3s, D1_059_KAN01_001 15.6s |
| stand → sit | 16 | D1_013_KAN01_001 8.9s, D1_031_KAN01_001 12.4s, D1_047_KAN01_001 12.9s |
| crouch → sit | 16 | D1_013_KAN01_001 22.4s, D1_025_KAN01_001 9.9s, D1_057_KAN01_001 0.8s |
| run → jump | 11 | D1_010_KAN01_004 1.7s, D1_057_KAN01_001 16.6s, D1_058_KAN01_001 26.3s |
| run → walk | 11 | D1_053_KAN01_002 2.2s, D1_053_KAN01_002 6.7s, D1_053_KAN01_002 10.5s |
| walk → sit | 9 | D1_008_KAN01_002 1.2s, D1_047z_KAN01_003 59.4s, D1_047z_KAN01_003 72.7s |
| trot → walk | 9 | D1_031_KAN01_001 11.1s, D1_045_KAN01_001 6.3s, D1_047z_KAN01_003 3.9s |
| walk → trot | 8 | D1_009_KAN01_001 6.6s, D1_009_KAN01_002 3.9s, D1_031_KAN01_001 8.3s |
| jump → crouch | 8 | D1_ex03_KAN02_006 35.3s, D1_ex03_KAN02_006 62.2s, D1_ex03_KAN02_006 76.8s |
| run → crouch | 7 | D1_061_KAN01_001 29.9s, D1_061_KAN01_001 108.9s, D1_ex03_KAN02_001 22.9s |
| down → sit | 6 | D1_006_KAN01_001 10.2s, D1_006_KAN01_002 14.9s, D1_061_KAN01_001 69.2s |
| sit → stand | 6 | D1_006_KAN01_001 12.6s, D1_013_KAN01_001 2.2s, D1_013_KAN01_001 27.4s |
| down → walk | 5 | D1_005_KAN01_001 13.3s, D1_013_KAN01_001 37.8s, D1_045_KAN01_001 2.3s |
| side → down | 5 | D1_006_KAN01_002 14.5s, D1_061z_KAN01_003 58.7s, D1_061z_KAN01_003 66.3s |
| crouch → down | 5 | D1_061_KAN01_001 42.8s, D1_061_KAN01_001 54.3s, D1_061_KAN01_001 112.4s |
| jump → run | 4 | D1_010_KAN01_004 2.1s, D1_ex04_KAN02_002 7.4s, D1_ex04_KAN02_004 9.0s |
| sit → down | 4 | D1_013_KAN01_001 14.1s, D1_061z_KAN01_002 16.9s, D1_061z_KAN01_002 29.8s |
| down → crouch | 4 | D1_013_KAN01_001 21.8s, D1_061_KAN01_001 2.2s, D1_061_KAN01_001 126.1s |
| stand → down | 4 | D1_013_KAN01_001 32.7s, D1_045_KAN01_001 7.5s, D1_071_KAN02_002 8.7s |
| run → trot | 4 | D1_045_KAN01_001 3.3s, D1_061_KAN01_001 34.3s, D1_ex03_KAN02_012 0.7s |
| rear_up → walk | 4 | D1_055_KAN01_001 104.3s, D1_059_KAN01_001 24.4s, D1_059_KAN01_001 28.8s |
| down → side | 4 | D1_061z_KAN01_003 42.5s, D1_061z_KAN01_003 65.6s, D1_061z_KAN01_003 78.4s |
| crouch → run | 4 | D1_ex03_KAN02_006 77.1s, D1_ex04_KAN02_001 9.6s, D1_ex04_KAN02_002 6.5s |
| crouch → rear_up | 3 | D1_059_KAN01_001 19.8s, D1_059_KAN01_001 23.9s, D1_059_KAN01_001 28.3s |
| stand → run | 3 | D1_061_KAN01_001 108.0s, D1_ex03_KAN02_001 22.6s, D1_ex03_KAN02_001 43.8s |
| walk → jump | 3 | D1_ex03_KAN02_006 69.2s, D1_ex04_KAN02_001 7.7s, D1_ex04_KAN02_002 5.1s |
| sit → rear_up | 2 | D1_055_KAN01_001 216.9s, D1_059_KAN01_001 45.1s |
| jump → walk | 2 | D1_057_KAN01_001 17.5s, D1_058_KAN01_001 27.0s |
| trot → run | 1 | D1_009_KAN01_001 8.8s |
| stand → rear_up | 1 | D1_055_KAN01_001 103.8s |
| rear_up → stand | 1 | D1_055_KAN01_001 217.7s |
| rear_up → sit | 1 | D1_059_KAN01_001 20.2s |
| down → stand | 1 | D1_061_KAN01_001 50.4s |
| down → run | 1 | D1_ex05_KAN02_001 25.1s |

## Clips

| clip | seconds | mean speed m/s | regimes (seconds each) |
|---|---|---|---|
| D1_001_KAN01_001 | 25.5 | 0.05 | stand 11.2 → walk 1.0 → stand 12.9 → walk 0.4 |
| D1_003_KAN01_001 | 15.9 | 0.01 | sit 15.9 |
| D1_004_KAN01_001 | 15.9 | 0.06 | sit 10.9 → walk 1.4 → stand 3.6 |
| D1_005_KAN01_001 | 16.6 | 0.03 | down 13.3 → walk 0.4 → stand 2.9 |
| D1_006_KAN01_001 | 16.3 | 0.04 | down 10.2 → sit 2.4 → stand 3.7 |
| D1_006_KAN01_002 | 22.7 | 0.06 | side 14.5 → down 0.5 → sit 0.3 → walk 1.5 → stand 6.0 |
| D1_007_KAN01_001 | 9.9 | 0.96 | sit 1.1 → walk 8.8 |
| D1_008_KAN01_001 | 17.9 | 0.69 | sit 4.4 → walk 13.5 |
| D1_008_KAN01_002 | 15.2 | 0.70 | walk 1.2 → sit 5.0 → walk 9.0 |
| D1_009_KAN01_001 | 9.1 | 1.02 | stand 2.7 → walk 3.9 → trot 2.2 → run 0.3 |
| D1_009_KAN01_002 | 9.1 | 1.14 | sit 3.5 → walk 0.4 → trot 5.2 |
| D1_010_KAN01_001 | 9.4 | 1.07 | sit 6.0 → walk 0.6 → run 2.8 |
| D1_010_KAN01_002 | 2.6 | 4.08 | run 2.6 |
| D1_010_KAN01_003 | 3.4 | 3.01 | run 3.4 |
| D1_010_KAN01_004 | 6.0 | 2.94 | run 1.7 → jump 0.4 → run 3.9 |
| D1_013_KAN01_001 | 47.3 | 0.05 | sit 2.2 → stand 6.6 → sit 5.3 → down 7.7 → crouch 0.6 → sit 5.0 → stand 5.3 → down 5.1 → walk 0.9 → stand 8.6 |
| D1_025_KAN01_001 | 17.9 | 0.36 | sit 2.9 → walk 6.3 → stand 0.4 → crouch 0.3 → sit 8.0 |
| D1_031_KAN01_001 | 21.0 | 0.40 | sit 7.6 → walk 0.8 → trot 2.7 → walk 1.0 → stand 0.4 → sit 8.6 |
| D1_045_KAN01_001 | 22.5 | 0.35 | down 2.3 → walk 0.6 → run 0.4 → trot 2.9 → walk 0.6 → stand 0.7 → down 14.9 |
| D1_047_KAN01_001 | 18.0 | 0.26 | sit 5.0 → walk 4.7 → stand 0.5 → walk 1.6 → stand 1.2 → sit 4.5 → walk 0.7 |
| D1_047z_KAN01_002 | 13.6 | 0.61 | stand 0.7 → walk 2.4 → stand 7.6 → walk 1.0 → run 1.9 |
| D1_047z_KAN01_003 | 78.1 | 0.26 | walk 1.5 → stand 0.8 → walk 1.3 → trot 0.4 → walk 2.1 → stand 0.8 → walk 3.1 → stand 0.9 → walk 0.6 → stand 0.7 → walk 1.0 → stand 1.7 → walk 0.5 → stand 1.8 → walk 1.3 → stand 0.6 → walk 1.0 → trot 0.3 → walk 1.2 → stand 4.4 → walk 1.1 → stand 8.3 → walk 2.0 → stand 7.0 → walk 0.5 → stand 7.8 → walk 1.9 → stand 2.2 → walk 0.3 → trot 0.9 → walk 1.6 → sit 2.5 → stand 0.8 → walk 0.6 → stand 3.9 → walk 2.4 → stand 1.2 → sit 1.1 → walk 0.8 → sit 4.6 → stand 0.8 |
| D1_047z_KAN01_004 | 17.6 | 0.28 | stand 2.8 → walk 2.2 → crouch 0.3 → walk 3.9 → crouch 0.7 → stand 7.7 |
| D1_047z_KAN01_005 | 45.8 | 0.42 | crouch 0.6 → stand 0.5 → walk 1.6 → crouch 0.6 → stand 1.0 → crouch 0.5 → stand 0.5 → walk 0.9 → crouch 0.5 → walk 2.0 → crouch 0.6 → stand 1.3 → crouch 0.6 → stand 0.5 → walk 4.0 → stand 0.5 → walk 1.6 → crouch 1.3 → walk 2.4 → crouch 0.4 → walk 0.4 → crouch 0.7 → walk 2.6 → crouch 3.3 → stand 0.6 → crouch 2.2 → stand 0.4 → walk 0.3 → crouch 3.2 → walk 0.8 → crouch 3.3 → walk 0.7 → stand 2.1 → walk 2.0 → crouch 0.5 → stand 0.9 |
| D1_049_KAN01_001 | 14.8 | 0.42 | sit 3.2 → walk 2.0 → trot 1.0 → walk 4.6 → sit 4.1 |
| D1_053_KAN01_001 | 72.3 | 0.49 | walk 3.4 → trot 0.4 → walk 5.2 → stand 1.2 → walk 17.7 → stand 0.4 → walk 2.7 → stand 1.6 → walk 2.0 → stand 1.4 → walk 4.6 → stand 1.6 → walk 2.8 → sit 0.6 → walk 4.0 → stand 0.8 → walk 1.6 → stand 0.8 → walk 1.2 → stand 3.3 → walk 0.9 → stand 6.3 → walk 1.2 → stand 1.6 → walk 3.0 → stand 0.4 → walk 1.3 |
| D1_053_KAN01_002 | 64.1 | 0.67 | run 2.2 → walk 1.4 → run 3.1 → walk 0.6 → run 3.2 → walk 5.5 → stand 0.8 → walk 2.5 → stand 0.6 → walk 1.1 → stand 0.6 → crouch 1.1 → stand 0.3 → walk 1.7 → stand 1.3 → crouch 1.0 → stand 1.4 → walk 1.1 → stand 7.9 → walk 1.1 → stand 1.2 → walk 2.3 → stand 8.2 → walk 1.0 → stand 0.8 → walk 2.0 → stand 1.5 → walk 0.7 → crouch 0.8 → stand 0.8 → crouch 0.4 → stand 0.8 → walk 0.6 → stand 3.7 → crouch 1.1 |
| D1_055_KAN01_001 | 223.3 | 0.20 | run 1.7 → walk 2.7 → stand 1.2 → walk 1.0 → stand 0.4 → walk 10.8 → stand 2.4 → crouch 1.6 → stand 2.1 → crouch 2.0 → stand 0.6 → crouch 1.8 → stand 3.9 → crouch 0.7 → stand 1.0 → crouch 1.1 → stand 12.8 → walk 1.4 → stand 6.4 → walk 2.8 → stand 5.9 → walk 1.2 → stand 1.0 → sit 1.6 → stand 1.1 → walk 0.8 → crouch 0.6 → stand 0.6 → walk 2.3 → stand 21.2 → walk 1.1 → crouch 0.8 → stand 0.7 → walk 2.2 → stand 4.4 → rear_up 0.5 → walk 0.4 → stand 6.2 → walk 6.7 → stand 1.0 → walk 1.0 → stand 4.1 → crouch 2.8 → stand 1.0 → crouch 3.1 → stand 4.3 → crouch 1.1 → stand 1.6 → crouch 4.6 → stand 3.8 → crouch 1.3 → stand 3.4 → crouch 4.2 → stand 11.6 → crouch 1.1 → stand 3.7 → crouch 2.1 → stand 0.4 → crouch 17.4 → stand 1.0 → crouch 4.3 → stand 1.9 → crouch 1.8 → stand 5.6 → walk 4.0 → crouch 2.0 → walk 2.7 → stand 2.0 → sit 0.6 → rear_up 0.7 → stand 2.6 → walk 3.1 |
| D1_057_KAN01_001 | 31.8 | 0.33 | walk 0.5 → crouch 0.3 → sit 14.3 → walk 0.5 → run 1.0 → jump 0.9 → walk 1.6 → stand 9.6 → walk 3.1 |
| D1_058_KAN01_001 | 36.3 | 0.47 | sit 3.9 → walk 9.6 → sit 2.4 → walk 1.4 → stand 6.5 → walk 1.3 → run 1.2 → jump 0.6 → walk 2.1 → stand 7.2 |
| D1_059_KAN01_001 | 50.4 | 0.28 | stand 6.5 → sit 0.3 → crouch 1.8 → sit 1.6 → crouch 1.7 → sit 0.7 → walk 0.6 → stand 1.9 → sit 0.4 → crouch 0.6 → sit 0.5 → crouch 3.1 → rear_up 0.4 → sit 1.6 → crouch 2.1 → rear_up 0.5 → walk 0.3 → stand 0.4 → sit 1.2 → crouch 1.9 → rear_up 0.6 → walk 0.6 → stand 0.9 → walk 2.7 → sit 1.0 → walk 1.4 → run 0.6 → walk 5.1 → stand 0.8 → walk 0.4 → sit 2.7 → rear_up 0.5 → walk 2.0 → stand 0.9 → walk 1.9 |
| D1_061_KAN01_001 | 194.8 | 0.34 | down 2.2 → crouch 0.3 → walk 1.6 → stand 0.4 → walk 0.6 → stand 0.5 → walk 0.6 → stand 14.9 → walk 1.1 → run 0.8 → walk 1.6 → stand 2.2 → walk 1.0 → stand 0.4 → walk 0.8 → run 1.0 → crouch 1.3 → walk 0.5 → run 2.6 → trot 0.6 → walk 6.7 → crouch 1.3 → down 7.6 → stand 1.0 → crouch 1.0 → stand 0.9 → crouch 0.9 → down 14.9 → sit 6.6 → walk 4.7 → stand 3.5 → walk 0.8 → stand 0.6 → walk 1.7 → stand 1.4 → walk 4.9 → stand 0.5 → walk 1.5 → stand 0.4 → walk 0.4 → stand 2.0 → walk 9.5 → stand 0.4 → run 0.9 → crouch 3.5 → down 13.7 → crouch 0.4 → walk 0.5 → stand 0.6 → walk 4.7 → stand 1.2 → walk 1.2 → stand 0.5 → walk 4.0 → stand 1.5 → walk 8.9 → run 1.2 → walk 2.3 → crouch 1.7 → stand 0.5 → walk 1.6 → stand 5.0 → crouch 0.5 → stand 0.9 → crouch 2.4 → down 26.5 → walk 2.6 |
| D1_061z_KAN01_002 | 51.4 | 0.27 | stand 1.6 → walk 1.9 → stand 7.7 → walk 0.9 → stand 3.6 → sit 1.3 → down 5.4 → walk 2.6 → stand 3.7 → sit 1.1 → down 9.2 → crouch 0.3 → walk 1.6 → stand 1.5 → walk 2.5 → stand 2.5 → walk 0.6 → stand 0.3 → walk 1.8 → run 1.4 |
| D1_061z_KAN01_003 | 145.8 | 0.12 | walk 4.8 → stand 2.3 → walk 1.0 → stand 0.9 → walk 0.7 → crouch 0.9 → stand 1.0 → walk 2.5 → stand 2.0 → crouch 1.6 → stand 0.6 → walk 1.7 → stand 13.7 → walk 1.2 → stand 1.8 → sit 1.4 → down 4.3 → side 16.2 → down 6.9 → side 0.7 → down 12.1 → side 8.0 → down 0.4 → sit 0.8 → walk 0.7 → stand 5.6 → crouch 10.4 → stand 2.0 → crouch 11.4 → stand 6.8 → walk 1.3 → stand 4.9 → walk 0.7 → crouch 1.5 → stand 1.0 → walk 1.5 → crouch 3.5 → stand 7.0 |
| D1_071_KAN02_002 | 47.6 | 0.19 | walk 4.2 → stand 4.5 → down 5.9 → side 12.7 → down 8.2 → sit 5.9 → walk 5.0 → stand 1.2 |
| D1_073_KAN02_002 | 54.0 | 0.26 | walk 0.6 → stand 1.3 → walk 0.4 → stand 2.0 → walk 4.2 → stand 4.4 → walk 1.6 → stand 1.9 → walk 0.8 → stand 1.4 → walk 4.2 → stand 0.6 → walk 0.6 → stand 1.1 → walk 0.4 → stand 1.0 → walk 0.7 → stand 3.4 → crouch 0.4 → stand 2.4 → walk 0.5 → stand 1.7 → walk 0.4 → stand 0.6 → walk 0.5 → stand 4.0 → walk 0.3 → stand 3.4 → walk 0.4 → stand 1.0 → walk 0.4 → stand 0.6 → walk 0.4 → stand 0.8 → walk 0.3 → stand 0.5 → walk 1.4 → stand 2.1 → walk 1.0 |
| D1_086_KAN02_001 | 85.0 | 0.07 | sit 2.7 → crouch 0.7 → sit 1.7 → crouch 0.5 → sit 4.1 → crouch 2.4 → sit 2.2 → crouch 5.1 → sit 15.1 → crouch 3.0 → sit 1.0 → crouch 1.0 → sit 2.8 → crouch 1.8 → walk 0.6 → sit 1.7 → crouch 1.1 → walk 0.5 → stand 2.6 → crouch 0.7 → stand 4.8 → walk 0.8 → stand 0.9 → walk 1.2 → stand 2.5 → sit 1.6 → crouch 1.0 → sit 9.1 → crouch 1.3 → sit 0.9 → crouch 2.7 → sit 7.0 |
| D1_ex01_KAN01_001 | 14.3 | 0.52 | stand 2.6 → walk 6.7 → stand 3.5 → walk 1.5 |
| D1_ex02_KAN02_001 | 81.2 | 0.12 | down 44.7 → sit 3.8 → crouch 0.3 → stand 7.6 → walk 0.4 → run 1.3 → walk 0.4 → crouch 1.4 → walk 5.5 → crouch 3.8 → down 12.1 |
| D1_ex03_KAN02_001 | 84.2 | 0.30 | stand 0.9 → sit 13.4 → walk 0.8 → crouch 2.4 → stand 5.0 → run 0.3 → crouch 0.8 → stand 2.8 → walk 11.7 → stand 5.6 → run 0.5 → crouch 2.0 → walk 2.9 → stand 2.4 → walk 2.5 → stand 1.4 → walk 0.4 → run 0.4 → crouch 1.2 → stand 4.4 → crouch 1.1 → walk 2.0 → stand 2.4 → walk 1.2 → stand 8.3 → walk 5.1 → stand 2.2 |
| D1_ex03_KAN02_003 | 136.7 | 0.06 | stand 1.9 → crouch 0.3 → stand 17.0 → walk 1.4 → crouch 0.7 → stand 4.5 → walk 1.7 → crouch 1.2 → stand 2.4 → walk 1.1 → stand 6.1 → walk 1.2 → stand 17.8 → walk 0.3 → stand 41.6 → crouch 1.2 → stand 0.6 → walk 0.7 → stand 32.5 → crouch 0.8 → stand 1.5 |
| D1_ex03_KAN02_006 | 113.3 | 0.33 | jump 35.3 → crouch 0.7 → walk 1.0 → stand 1.3 → walk 2.5 → stand 0.6 → walk 0.5 → stand 2.9 → walk 3.5 → run 0.3 → jump 13.6 → crouch 0.4 → walk 0.5 → stand 5.0 → walk 1.1 → jump 7.6 → crouch 0.3 → run 0.9 → walk 1.8 → stand 2.3 → crouch 0.7 → stand 2.3 → walk 2.5 → stand 1.5 → walk 4.1 → crouch 0.4 → stand 0.5 → walk 2.1 → stand 1.2 → walk 2.5 → stand 1.6 → walk 0.6 → stand 0.8 → walk 1.1 → run 0.3 → jump 9.1 |
| D1_ex03_KAN02_012 | 59.9 | 0.57 | run 0.7 → trot 2.0 → walk 0.5 → stand 14.0 → walk 0.4 → crouch 0.9 → walk 0.8 → stand 3.2 → walk 4.7 → stand 8.1 → walk 24.5 |
| D1_ex03_KAN02_013 | 76.3 | 0.18 | walk 7.9 → stand 16.0 → crouch 0.3 → stand 9.7 → sit 28.1 → walk 0.8 → crouch 0.6 → stand 6.2 → walk 3.1 → stand 3.4 |
| D1_ex03_KAN02_014 | 58.1 | 0.34 | walk 5.7 → stand 1.5 → walk 2.0 → stand 1.4 → walk 1.5 → stand 0.9 → sit 2.1 → crouch 0.4 → walk 3.7 → stand 0.4 → walk 1.4 → stand 1.6 → walk 3.6 → stand 0.5 → walk 1.5 → stand 0.4 → walk 5.0 → stand 9.1 → walk 0.6 → run 0.8 → walk 1.1 → stand 9.5 → walk 0.8 → stand 2.6 |
| D1_ex04_KAN02_001 | 13.0 | 0.68 | walk 0.4 → stand 5.0 → walk 2.3 → jump 1.5 → crouch 0.4 → run 0.5 → jump 0.3 → crouch 0.3 → walk 2.2 |
| D1_ex04_KAN02_002 | 8.4 | 1.16 | sit 2.5 → walk 2.6 → jump 1.1 → crouch 0.3 → run 0.4 → jump 0.5 → run 1.0 |
| D1_ex04_KAN02_003 | 8.3 | 1.44 | walk 0.4 → run 0.3 → jump 0.3 → crouch 0.3 → run 0.5 → jump 1.9 → crouch 0.4 → walk 2.6 → run 1.6 |
| D1_ex04_KAN02_004 | 12.5 | 0.91 | walk 1.2 → stand 2.3 → walk 0.9 → crouch 0.3 → sit 2.8 → walk 0.5 → run 0.4 → jump 0.6 → run 0.4 → jump 1.2 → run 0.4 → trot 1.4 |
| D1_ex05_KAN02_001 | 154.7 | 0.42 | walk 6.6 → stand 1.5 → walk 0.5 → run 1.0 → crouch 0.6 → walk 1.6 → stand 4.1 → crouch 1.1 → stand 0.8 → down 7.3 → run 1.1 → crouch 0.8 → walk 0.5 → stand 0.4 → walk 1.0 → stand 2.3 → walk 2.0 → stand 1.1 → walk 1.5 → stand 0.7 → walk 1.5 → stand 0.9 → walk 1.2 → stand 2.1 → walk 9.8 → stand 1.1 → walk 0.9 → stand 2.3 → walk 0.7 → stand 1.7 → walk 0.8 → stand 0.8 → walk 3.9 → stand 4.2 → walk 8.2 → stand 4.5 → walk 0.7 → stand 4.9 → walk 0.3 → stand 0.8 → walk 3.8 → crouch 0.6 → walk 1.7 → crouch 1.2 → stand 0.5 → walk 0.6 → stand 1.5 → walk 2.6 → crouch 0.9 → walk 9.3 → stand 1.4 → walk 9.0 → run 0.9 → walk 2.5 → stand 0.5 → walk 1.5 → stand 1.2 → walk 0.3 → stand 0.3 → walk 0.5 → stand 0.4 → walk 0.4 → stand 25.3 |
| D1_ex06_KAN02_001 | 89.3 | 0.25 | stand 4.3 → walk 0.9 → stand 2.5 → walk 0.7 → crouch 3.2 → walk 1.5 → stand 1.0 → crouch 1.2 → walk 0.6 → stand 6.9 → walk 1.1 → crouch 2.2 → stand 0.5 → walk 0.7 → stand 4.2 → walk 3.6 → stand 3.6 → walk 1.0 → stand 1.2 → walk 0.9 → stand 0.5 → crouch 1.1 → stand 1.2 → crouch 0.4 → stand 4.4 → walk 1.2 → stand 3.6 → walk 1.0 → stand 2.9 → crouch 0.6 → stand 3.6 → crouch 0.6 → stand 1.7 → walk 2.2 → stand 3.1 → walk 0.8 → stand 0.8 → walk 4.3 → crouch 0.9 → walk 2.3 → stand 5.2 → walk 5.1 |
