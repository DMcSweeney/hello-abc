"""
Script with sanity checking endpoints
"""
import base64
import logging
from flask import Blueprint, request, make_response, jsonify
from app import mongo

bp = Blueprint('api/sanity', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/sanity/fetch_first_image', methods=["GET"])
def fetchFirstImage():
    ## Figure out what image to retrieve
    vertebra = 'L3'
    database = mongo.db 
    response = database.quality_control.find_one({
        f"quality_control.{vertebra}": 2
    })

    #TODO Faster if image stored in db? Instead of loading and converting
    path_to_sanity_ = response['paths_to_sanity_images']['ALL'] 
    with open(path_to_sanity_, 'rb') as f:
        image = bytearray(f.read())
    encoded_im = base64.b64encode(image).decode('utf8').replace("'",'"')

    res = make_response(jsonify({
        "message": "Here's your image",
        "image": encoded_im,
        "patient_id": response["patientID"],
    }), 200)

    return res

@bp.route('/api/sanity/fetch_image_list', methods=["GET"])
def fetchImageList():
    # Figure out what image to retrieve
    vertebra = 'L3'
    database = mongo.db 
    cursor = database.quality_control.find({
        f"quality_control.{vertebra}": 2
    })
    for doc in cursor:
        print(doc, flush=True)

    res = make_response(jsonify({
        "message": "Here's your list",
    }), 200)
    
    #TODO return series IDs for faster retrieve?

    return res