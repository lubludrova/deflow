# Anderson stopping iterations in the E06 table audit

Native n_iter includes two initialization evaluations. T = stopped by batch-mean tolerance, C = exhausted the iteration cap. All nine audit batches are included, independently of FP64 reference validity. These are not full-training statistics.

| Target step | Seed | 1e-5/10 | 1e-5/25 | 1e-5/100 | 1e-5/500 | 1e-6/100 |
|---|---|---|---|---|---|---|
| 25000 | 1 | 10 C | 25 C | 54 T | 54 T | 77 T |
| 25000 | 2 | 10 C | 25 C | 55 T | 55 T | 80 T |
| 25000 | 3 | 10 C | 25 C | 52 T | 52 T | 75 T |
| 100000 | 1 | 10 C | 25 C | 56 T | 56 T | 74 T |
| 100000 | 2 | 10 C | 25 C | 100 C | 163 T | 100 C |
| 100000 | 3 | 10 C | 25 C | 66 T | 66 T | 88 T |
| 200000 | 1 | 10 C | 25 C | 61 T | 61 T | 83 T |
| 200000 | 2 | 10 C | 25 C | 100 C | 186 T | 100 C |
| 200000 | 3 | 10 C | 25 C | 100 C | 500 C | 100 C |
