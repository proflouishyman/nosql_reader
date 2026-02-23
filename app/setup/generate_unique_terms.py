#!/usr/bin/env python3
"""
Compatibility wrapper for the legacy setup path.

The canonical generator now lives at /app/generate_unique_terms.py.
"""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    # Keep legacy callers working by forwarding execution to the canonical script.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        # Ensure canonical imports like `from database_setup import ...` resolve.
        sys.path.insert(0, str(project_root))

    canonical_script = project_root / "generate_unique_terms.py"
    runpy.run_path(str(canonical_script), run_name="__main__")
