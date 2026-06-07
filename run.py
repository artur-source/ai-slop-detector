# Local Flask entrypoint for the AI slop detector web app.

from __future__ import annotations

import os

from app import create_app
from config import AppConfig


config = AppConfig()
app = create_app()


if __name__ == "__main__":
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(host=config.flask_host, port=config.flask_port, debug=debug)

