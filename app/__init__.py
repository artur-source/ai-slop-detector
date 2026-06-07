# Flask application factory and package initialization.

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured Flask application instance.
    """

    load_dotenv()

    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-key-change-in-prod")

    from app.routes import main

    app.register_blueprint(main)

    return app

