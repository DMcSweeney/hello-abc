"""
Flask application
"""

import os
from flask import Flask
from flask_cors import CORS
import main, api

INPUT_DIR = '/data/inputs/'
OUTPUT_DIR = '/data/outputs/abc-spine/'
DB_NAME = 'db.sqlite'

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
    app.register_blueprint(main.bp)
    app.register_blueprint(api.bp)

    app.add_url_rule('/', endpoint='main')

    return app


App = create_app()
CORS(App) # Allow Cross-origin requests