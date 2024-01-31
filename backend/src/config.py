import os


class BaseConfig():
    DEBUG = os.environ['FLASK_DEBUG']
    DB_NAME = os.environ["POSTGRES_DB"]
    DB_USER = "postgres"
    DB_PASS = "postgres"
    DB_PORT = os.environ["POSTGRES_PORT"]
    SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASS}@db:{DB_PORT}/{DB_NAME}'
    SECRET_KEY = 'secret'
    ...