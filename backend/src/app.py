"""
Flask application
"""

import os
import logging
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

import main
from config import BaseConfig
from api import spine, segment

INPUT_DIR = '/data/inputs/'
OUTPUT_DIR = '/data/outputs/'
DB_NAME = 'db.sqlite'

logger = logging.getLogger(__name__)

logging.basicConfig(filename=f'/var/log/backend.log',
level=logging.INFO,
format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
force=True,
)

def init_models(db):
    class Images(db.Model):
        __tablename__ = 'images'
        id = db.Column(db.Integer, primary_key=True)
        patientID = db.Column(db.String, nullable=False)
        project = db.Column(db.String, nullable=False)
        path = db.Column(db.String, nullable=False)
        series_uuid = db.Column(db.String, nullable=False)
    class Results(db.Model):
        __tablename__ = 'results'
        id = db.Column(db.Integer, ForeignKey(Images.id), primary_key=True)
        
        relationship('Images', foreign_keys='Results.id')
    
    db.create_all()

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(BaseConfig)

    db = SQLAlchemy()

    app.config['INPUT_DIR'] = INPUT_DIR
    app.config['OUTPUT_DIR'] = OUTPUT_DIR

    # Add blueprints
    app.register_blueprint(main.bp)
    app.register_blueprint(spine.bp)
    app.register_blueprint(segment.bp)

    app.add_url_rule('/', endpoint='main')

    db.init_app(app)
    with app.app_context():
        init_models(db)
        
    return app


App = create_app()
CORS(App) # Allow Cross-origin requests~
logger.info("App started")



