"""
Main flask application
"""
import logging
from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from redis import Redis

import main
from config import BaseConfig


INPUT_DIR = '/data/inputs/'
OUTPUT_DIR = '/data/outputs/'

logger = logging.getLogger(__name__)

logging.basicConfig(#filename=f'/var/log/backend.log',
level=logging.INFO,
format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
force=True,
)

# Init flask app
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(BaseConfig)
app.config['INPUT_DIR'] = INPUT_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR

#Connect to MongoDB
logger.info(f"Starting connection to: {app.config['MONGO_URI']}")
mongo = PyMongo(app)

# Connect to Redis
logger.info("Connecting to Redis")
redis = Redis(host='redis', port=6379)

#import here to bypass circular imports
from api import spine, segment, sanity, post_process, conquest 

# Add blueprints
app.register_blueprint(main.bp)
app.register_blueprint(spine.bp)
app.register_blueprint(segment.bp)
app.register_blueprint(sanity.bp)
app.register_blueprint(post_process.bp)
app.register_blueprint(conquest.bp)

app.add_url_rule('/', endpoint='main')

# with App.app_context():
#     init_models(db)

CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow Cross-origin requests~
logger.info("App started")



