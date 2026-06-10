"""CEC-2013 Large-Scale Global Optimization benchmark (seed-based Python implementation).

Provides F1-F15 from the CEC-2013 LSGO competition (the set in dmolina/cec2013lsgo).
GPLv3 — see LICENSE, NOTICE, and CHANGELOG.md in the repository root.
"""

from .benchmarks import LSGO2013, VALID_FUNC_IDS

__all__ = ["LSGO2013", "VALID_FUNC_IDS"]
