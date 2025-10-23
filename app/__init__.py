"""app package initializer.

This file allows modules such as ``app.util.mongo_env`` to be imported
reliably when the codebase runs inside Docker or via direct script
execution. Without it, Python treats ``app`` as a plain directory and
``from app.util import ...`` raises ``ModuleNotFoundError``.
"""

# No runtime logic is required here; the presence of this file alone
# registers the directory as a Python package so entrypoint helpers and
# maintenance scripts can resolve the shared MongoDB utilities.
