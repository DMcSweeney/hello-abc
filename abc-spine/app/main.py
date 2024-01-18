"""
Abc-Spine main endpoints
"""
import os
from flask import Blueprint, request, make_response, jsonify

bp = Blueprint('main', __name__)


@bp.route('/hello', methods=["GET", "POST"])
def hello():
    if request.method=="POST":
        user = request.args.get("user")
        print(f'Request received', flush=True)
        if user is not None:
            message = f"Hello {user}"
        else:
            message = "Hello"
        # Reply when another service says hello    
        res = make_response(jsonify({
            "message": message,
            "from": "abc-spine"
        }), 200)

    elif request.method=="GET":
        message = "Hello"
        res = make_response(jsonify({
            "message": message,
            "from": "abc-spine"
        }), 200)

    return res