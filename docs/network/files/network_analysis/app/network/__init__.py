# app/network/__init__.py
"""
Network Analysis module for the Historical Document Reader.

Provides co-occurrence network construction, querying, and visualization
for entities extracted from document collections.

Domain-agnostic: works with any entity types present in linked_entities.
"""

from flask import Blueprint

network_bp = Blueprint(
    'network',
    __name__,
    template_folder='../templates',
    url_prefix='/api/network',
)

# Import routes to register them on the blueprint.
# This import MUST come after the Blueprint is created to avoid circular imports.
from app.network import network_routes  # noqa: F401, E402


def init_app(app):
    """
    Register the network blueprint with the Flask application.

    Call this from your app factory or directly in app.py/routes.py:

        from app.network import init_app as init_network
        init_network(app)

    The blueprint is only registered when NETWORK_ANALYSIS_ENABLED is truthy.
    """
    from app.network.config import NetworkConfig

    config = NetworkConfig.from_env()
    if not config.enabled:
        app.logger.info("Network analysis is disabled (NETWORK_ANALYSIS_ENABLED != true).")
        return

    app.register_blueprint(network_bp)
    app.logger.info("Network analysis blueprint registered at /api/network.")
