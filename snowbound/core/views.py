#core/views.py
from flask import render_template, redirect, url_for, request, Blueprint, flash
core = Blueprint('core', __name__)

# render introduction page
@core.route('/')
def index():
    return render_template('index.html')

# render conclusion page
@core.route('/conclusion.html', methods=['GET', 'POST'])
def conclusion():
    return render_template('conclusion.html')
