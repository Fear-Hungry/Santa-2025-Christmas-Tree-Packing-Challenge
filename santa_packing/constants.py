"""Project-wide constants.

These values centralize numeric tolerances, submission formatting rules, and
competition constraints used throughout the codebase.
"""

from __future__ import annotations

# Numerical tolerance used across collision/formatting/validation.
# Numerical epsilon used by collision/quantization helpers. Keep this large
# enough to avoid false positives on strict "touching" edge cases, but small
# enough to not mask real overlaps.
EPS: float = 1e-9

# Submission formatting.
SUBMISSION_DECIMALS: int = 17
SUBMISSION_PREFIX: str = "s"

# Competition constraints.
XY_LIMIT: float = 100.0
