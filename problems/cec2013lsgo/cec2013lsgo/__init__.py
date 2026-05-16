"""CEC-2013 Large-Scale Global Optimisation benchmark (seed-based Python implementation).

This package provides F1-F15 from the CEC-2013 LSGO competition.
All problem data is generated from a user-supplied seed, making the benchmark
fully scalable to any dimensionality D.

F16-F25 from the original repository are intentionally omitted; they were
hard-wired to specific dimensions (D=10 000 and D=100 000) and have been
superseded by this generalisable seed-based approach.
"""

from .benchmarks import LSGO2013, VALID_FUNC_IDS

__all__ = ["LSGO2013", "VALID_FUNC_IDS"]
