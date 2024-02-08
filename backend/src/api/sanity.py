"""
Script with sanity checking endpoints
"""

import logging
from flask import Blueprint, request, make_response, jsonify
from app import mongo

bp = Blueprint('api/sanity', __name__)
logger = logging.getLogger(__name__)


@bp.route('/api/sanity/fetch_image', methods=["GET"])
def fetchImage():
    ## Figure out what image to retrieve
    vertebra = 'L3'
    database = mongo.db 
    response = database.quality_control.find_one({
        {f"quality_control.{vertebra}": 2}
    })
    
    print(response, flush=True)    

    res = make_response(jsonify({
        "message": "Here's your image"
    }), 200)

    return res