from flask import Flask
import os
from flask_cors import CORS
from app.models.saveForm import Dataset

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('config.py')
    CORS(app)
    app.secret_key = os.urandom(24)

    from app.urls import bind_urls
    bind_urls(app)

    from app.extensions import config_extensions
    config_extensions(app)
    return app
