"""
Web app
"""
import os
import requests
from flask import Blueprint, render_template, request, jsonify, make_response, current_app
import logging

bp = Blueprint('index', __name__)

logger = logging.getLogger(__name__)


@bp.route('/')
def home():
    return render_template('index/index.html')


@bp.route('/hello', methods=['GET'])
def hello():
    res = requests.get(f"http://backend:5001/hello")
    print(f"Backend replied with {res}", flush=True)
    if res.status_code == 200:
        res = res.json()
        return make_response(res, 200)
    
@bp.route('/show_input_dir', methods=["GET"])
def show_input_dir():
    files= os.listdir(current_app.config["INPUT_DIR"])
    res = make_response(jsonify({
        "folders": files
    }
    ), 200)
    return res