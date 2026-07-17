# Intrinsic-dimension random projection DE vs full-space DE

Baseline: `Full space (F=0.5, CR=0.9)` from past W&B runs (3M NFE, D=1000).
Treatment: random projection at the per-function intrinsic dimension (90% fitness-variance capture), absolute assignment, same DE settings.

| Function | Intrinsic d | RP mean best | Full-space mean | Ratio (RP/FS) | Seed wins |
|----------|-------------|--------------|-----------------|---------------|-----------|
| F3 | 8 | 2.103441e+01 | 2.931587e+00 | 7.175 | 0/1 |

Ratio < 1 means intrinsic-dimension RP beat the full-space mean on average.
Seed wins counts matched-seed runs where RP best fitness < full-space best fitness.
