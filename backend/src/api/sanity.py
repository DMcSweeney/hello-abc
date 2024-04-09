"""
Script with sanity checking endpoints
"""
import base64
import logging
from flask import Blueprint, request, make_response, jsonify, abort
from app import mongo

import abcTK.database.collections as cl


bp = Blueprint('api/sanity', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/sanity/fetch_first_image', methods=["POST"])
def fetchFirstImage():#
    project = request.args.get("project")

    ## Figure out what image to retrieve
    vertebra = 'L3'
    database = mongo.db 
    response = database.quality_control.find_one({
        f"quality_control.{vertebra}": 2, 'project': project
    })
    if response is None:
        ## None to label just show failures
        response = database.quality_control.find_one({
        f"quality_control.{vertebra}": 0, 'project': project
        })

    im_info = database.images.find_one({"_id": response["_id"]})

    #TODO Faster if image stored in db? Instead of loading and converting
    path_to_sanity_ = response['paths_to_sanity_images']['ALL'] 
    with open(path_to_sanity_, 'rb') as f:
        image = bytearray(f.read())
    encoded_im = base64.b64encode(image).decode('utf8').replace("'",'"')

    res = make_response(jsonify({
        "message": "Here's your image",
        "image": encoded_im,
        "patient_id": response["patient_id"],
        "status": response["quality_control"][vertebra],
        "series_uuid": im_info["series_uuid"],
        "acquisition_date": im_info["acquisition_date"],
        "input_path": im_info["input_path"],
        "vertebra": vertebra
    }), 200)

    return res


@bp.route('/api/sanity/fetch_image_by_id', methods=["POST"])
def fetchImageByID():
    _id = request.args.get("_id")
    project = request.args.get("project")
    vertebra = 'L3'
    database = mongo.db 
    response = database.quality_control.find_one({f"_id": _id,'project': project})
    im_info = database.images.find_one({"_id": response["_id"]})
    print(response, im_info, flush=True)
    #TODO Faster if image stored in db? Instead of loading and converting
    path_to_sanity_ = response['paths_to_sanity_images']['ALL'] 
    with open(path_to_sanity_, 'rb') as f:
        image = bytearray(f.read())
    encoded_im = base64.b64encode(image).decode('utf8').replace("'",'"')

    res = make_response(jsonify({
        "message": "Here's your image",
        "image": encoded_im,
        "patient_id": response["patient_id"],
        "status": response["quality_control"][vertebra],
        "series_uuid": im_info["series_uuid"],
        "acquisition_date": im_info["acquisition_date"],
        "input_path": im_info["input_path"],
        "vertebra": vertebra
    }), 200)

    return res


@bp.route('/api/sanity/fetch_image_list', methods=["POST"])
def fetchImageList():
    project = request.args.get("project")
    # Figure out what image to retrieve
    vertebra = 'L3'
    database = mongo.db

    response = database.quality_control.find_one({f"quality_control.{vertebra}": 2, 'project': project})
    cursor = database.quality_control.find(
        {'project': project, f"quality_control.{vertebra}": {"$exists": True}},
        {'_id': 1, f'quality_control.{vertebra}': 1}
        )
    
    if response is None:
        ## If none to label, show errors first
        cursor = cursor.sort(f"quality_control.{vertebra}")
    else:
        cursor = cursor.sort(f"quality_control.{vertebra}", -1)
    #TODO There must be a better way to do this?
    ids = []
    for doc in cursor:
        ids.append(doc['_id'])
    
    res = make_response(jsonify({
        "message": "Here's your list of IDs",
        "id_list": ids
    }), 200)
    return res



@bp.route('/api/sanity/fetch_spine_by_id', methods=["POST"])
def fetchSpineByID():
    _id = request.args.get("_id")
    project = request.args.get("project")
    database = mongo.db 
    response = database.quality_control.find_one({f"_id": _id, 'project': project})
    #TODO Faster if image stored in db? Instead of loading and converting
    path_to_sanity_ = response['paths_to_sanity_images']['SPINE'] 
    with open(path_to_sanity_, 'rb') as f:
        image = bytearray(f.read())
    encoded_im = base64.b64encode(image).decode('utf8').replace("'",'"')

    res = make_response(jsonify({
        "message": "Here's your image",
        "image": encoded_im,
        "patient_id": response["patient_id"],
    }), 200)

    return res

@bp.route('/api/sanity/pass_qa', methods=["POST"])
def pass_qa():
    project = request.args.get("project")
    _id = request.args.get("_id")
    vertebra = 'L3'
    database = mongo.db 
    database.quality_control.update_one({f"_id": _id, 'project': project}, {'$set': {f'quality_control.{vertebra}': 1, 'quality_control.SPINE': 1} })

    res = make_response(jsonify({
        "message": "QA pass recorded"
    }), 200)

    return res


@bp.route('/api/sanity/fail_qa', methods=["POST"])
def fail_qa():
    project = request.args.get("project")
    _id = request.args.get("_id")
    mode = request.args.get("mode")
    vertebra = 'L3'
    database = mongo.db 

    if mode == 'segmentation':
        database.quality_control.update_one({f"_id": _id, 'project': project},  {'$set': {f'quality_control.{vertebra}': 0, f'quality_control.SPINE': 1} })
    elif mode == 'labelling':
        database.quality_control.update_one({f"_id": _id,'project': project},  {'$set': {f'quality_control.SPINE': 0, f'quality_control.{vertebra}': 1} })
    elif mode == 'both':
        database.quality_control.update_one({f"_id": _id,'project': project},  {'$set': {f'quality_control.SPINE': 0, f'quality_control.{vertebra}': 0} })
    else:
        abort(400)

    res = make_response(jsonify({
        "message": "QA fail recorded"
    }), 200)

    return res


@bp.route('/api/sanity/get_summary', methods=['GET'])
def get_summary():
    #* Get recep of pass/fail/todo for current project
    project = request.args.get("project")
    vertebra = 'L3'

    database = mongo.db

    # Total documents 
    total = database.quality_control.count_documents({'project': project, f"quality_control.{vertebra}": {"$exists": True} })
    # Pass labelling + segmentation
    num_pass = database.quality_control.count_documents({'project': project, f"quality_control.{vertebra}": {"$exists": True},
                                                          '$and': [ {"quality_control.SPINE": 1}, {f"quality_control.{vertebra}": 1}]})
    # Fail either labelling or segmentation
    num_fail = database.quality_control.count_documents({'project': project, f"quality_control.{vertebra}": {"$exists": True},
                                                          '$or': [ {"quality_control.SPINE": 0}, {f"quality_control.{vertebra}": 0}]})
    # Unreviewed
    num_todo =database.quality_control.count_documents({'project': project, f"quality_control.{vertebra}": {"$exists": True},
                                                          '$or': [ {"quality_control.SPINE": 2}, {f"quality_control.{vertebra}": 2}]})

    logger.info(f"Found {num_pass} successes and {num_fail} failures with {num_todo} still to review.")
    res =make_response(jsonify({
        "message": "Succesfully retrieved summary for current project",
        "pass": num_pass,
        "fail": num_fail,
        "todo": num_todo,
        "total": total
    }), 200)

    return res


@bp.route('/api/sanity/fail_qa_report', methods=['POST'])
def fail_qa_report():
    ## Get query string args
    project = request.args.get("project")
    _id = request.args.get("_id")
    vertebra = 'L3'

    ## Get request body
    req = request.get_json()    
    payload = {k: v for k, v in req.items() if v != ''} ## If not empty
    print("Received payload: ", payload, flush=True)
    # Update db
    database = mongo.db

    database.quality_control.update_one({"_id": _id}, {"$set": {"qc_report": payload}}, upsert=True)
    logger.info(f"Updated quality control collection with: {payload}")

    ##TODO Update qc control flag
    if payload["failMode"] == 'badSegmentation':
        database.quality_control.update_one({f"_id": _id, 'project': project},  {'$set': {f'quality_control.{vertebra}': 0, f'quality_control.SPINE': 1} })
    elif payload["failMode"] == 'wrongLevel':
        database.quality_control.update_one({f"_id": _id,'project': project},  {'$set': {f'quality_control.SPINE': 0, f'quality_control.{vertebra}': 1} })
    else:
        ## FailMode = Other or scan Issue fail both 
        ## Fail both
        database.quality_control.update_one({f"_id": _id,'project': project},  {'$set': {f'quality_control.SPINE': 0, f'quality_control.{vertebra}': 0} })

    res = make_response(jsonify({
        'message': 'Succesfully submitted report'
    }), 200)

    return res