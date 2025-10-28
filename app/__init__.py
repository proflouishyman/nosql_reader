"""Initialize the app package so submodules like utils are importable."""  # enables package imports after renaming entrypoint

from .main import app, cache  # Re-export to keep existing from app import app, cache statements working
