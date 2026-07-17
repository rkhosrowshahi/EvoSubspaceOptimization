# Intrinsic-dimension random projection CMAES vs full-space DE

Baseline: `Full space (F=0.5, CR=0.9)` from past W&B runs (3M NFE, D=1000).
Treatment: random projection at the per-function intrinsic dimension (90% fitness-variance capture), absolute assignment, CMAES via pymoo (pop=100).

| Function | Intrinsic d | RP mean best | Full-space mean | Ratio (RP/FS) | Seed wins |
|----------|-------------|--------------|-----------------|---------------|-----------|
| F3 | 8 | 2.103900e+01 | 2.931587e+00 | 7.177 | 0/4 |
| F6 | 13 | 1.012677e+06 | 1.851298e+03 | 547.009 | 0/3 |
| F10 | 17 | 9.015731e+07 | 8.640240e+07 | 1.043 | -- |

Ratio < 1 means intrinsic-dimension RP beat the full-space mean on average.
Seed wins counts matched-seed runs where RP best fitness < full-space best fitness.
