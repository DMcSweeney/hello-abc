"""
Endpoints for interacting with database

"""


from flask import Blueprint, request, make_response, jsonify
import logging

bp = Blueprint('api/database', __name__)
logger = logging.getLogger(__name__)


@bp.route('/api/database/delete_entry', methods=["POST"])
def delete_entry():
    req = request.get_json()
    _id = req['_id']
    coll = req['collection']
    
    from app import mongo

    database = mongo.db
    database[coll].delete_one({"_id": _id})

    res = make_response(jsonify({
        'message': f'Succesfully deleted entry {_id} in {coll}'
    }), 200)

    return res


@bp.route('/api/database/get_qc_report', methods=['GET'])
def get_gc_report():
    from app import mongo
    database = mongo.db
    project = request.args.get("project")
    _id = request.args.get("_id") if "_id" in request.args else None
    print(project, _id, flush=True)
    if _id is not None:
        cursor = database.quality_control.find({"project": project, "_id": _id, "qc_report": {"$ne": {}} }, {"series_uuid": 1, "qc_report": 1})
        message = f'Found report for _id ({_id})'
    else:
        cursor = database.quality_control.find({"project": project, "qc_report": {"$ne": {}} }, {"series_uuid": 1, "qc_report": 1})
        message = f'Found {len(reports)} non-empty qc reports for project ({project})'
    reports = [(x['series_uuid'], x['qc_report']) for x in cursor]

    if len(reports) == 0:
        res = make_response(jsonify({
            'message': "Could not find any reports",
            "reports": None,
            }), 200)
    else:
        res = make_response(jsonify({
            'message': message,
            "reports": reports
        }), 200)

    return res