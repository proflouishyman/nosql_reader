# app/network/__init__.py
"""
Network Analysis module for the Historical Document Reader.

Provides co-occurrence network construction, querying, and visualization
for entities extracted from document collections.

Domain-agnostic: works with any entity types present in linked_entities.
"""

from flask import Blueprint

network_bp = Blueprint(
    "network",
    __name__,
    template_folder="../templates",
    url_prefix="/api/network",
)


def init_app(app):
    """
    Register the network blueprint with the Flask application.

    Call this from your app factory or directly in app.py/routes.py:

        from network import init_app as init_network
        init_network(app)

    The blueprint is only registered when NETWORK_ANALYSIS_ENABLED is truthy.
    """
    from network.config import NetworkConfig
    from network import network_routes  # noqa: F401
    from network import statistics_routes  # noqa: F401

    if "network" not in app.blueprints:
        app.register_blueprint(network_bp)
        config = NetworkConfig.from_env()
        if config.enabled:
            app.logger.info("Network analysis blueprint registered at /api/network.")
        else:
            app.logger.info("Network analysis blueprint registered in disabled mode.")
