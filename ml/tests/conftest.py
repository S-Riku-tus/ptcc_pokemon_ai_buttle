from __future__ import annotations

import sys
from pathlib import Path

# Make `import ml` work whether pytest is invoked from the repository root or
# directly against ml/tests.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT,):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
