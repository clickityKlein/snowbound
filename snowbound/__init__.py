#snowbound/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from flask_login import LoginManager


app = Flask(__name__)


# app.config['SECRET_KEY'] = 'mysecret'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')


### database setup ###
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)
Migrate(app, db)
### database setup ###


### login manager ###
#login_manager = LoginManager()
#login_manager.init_app(app)
#login_manager.login_view = 'users.login'
### login manager ###


### blueprint registrations ###
from snowbound.core.views import core
from snowbound.error_pages.handlers import error_pages
from snowbound.data_pages_controls.views import data_pages_controls
from snowbound.model_pages_controls.views import model_pages_controls
app.register_blueprint(core)
app.register_blueprint(error_pages)
app.register_blueprint(data_pages_controls)
app.register_blueprint(model_pages_controls)
### blueprint registrations ###
