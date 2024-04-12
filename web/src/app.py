"""
Main homepage for ABC-web
"""

import os
from flask import Flask
from flask_cors import CORS
import index as index
import logging

INPUT_DIR = '/data/inputs/'
OUTPUT_DIR = '/data/outputs/web/'
DB_NAME = 'db.sqlite'

logger = logging.getLogger(__name__)

logging.basicConfig(filename=f'/var/log/web.log',
level=logging.INFO,
format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
force=True,
)


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    path_to_db = os.path.join(OUTPUT_DIR, DB_NAME)

    app.config.from_mapping(
        SECRET_KEY='secret',
        DATABASE=path_to_db
    )
    app.config['INPUT_DIR'] = INPUT_DIR
    app.config['OUTPUT_DIR'] = OUTPUT_DIR

    # Add blueprints
    app.register_blueprint(index.bp)
    app.add_url_rule('/', endpoint='index')

    return app

App=create_app()
CORS(App) # Allow cross-origin requests
logger.info("App started")