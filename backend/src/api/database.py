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
    print("I AM HERE", _id, coll, flush=True)
    from app import mongo

    database = mongo.db
    database[coll].delete_one({"_id": _id})

    res = make_response(jsonify({
        'message': f'Succesfully deleted entry {_id} in {coll}'
    }), 200)

    return res
