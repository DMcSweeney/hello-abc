"""
Endpoints for submitting jobs

"""
import dill
import requests
import logging
from flask import Blueprint, request, make_response, jsonify, current_app
from rq import Queue
from rq.job import Job


bp = Blueprint('api/jobs', __name__)
logger = logging.getLogger(__name__)


@bp.route('/api/jobs/infer/spine', methods=["POST"])
def queue_infer_spine():
    req = request.get_json()

    req['APP_OUTPUT_DIR'] = current_app.config['OUTPUT_DIR']
    from app import redis
    from abcTK.inference.spine import infer_spine

    q = Queue(connection=redis, serializer=dill)
    job = q.enqueue(infer_spine, req)

    res = make_response(jsonify({
            "message": "Spine inference submitted",
            "request": req,
            "job-ID": job.id})
            , 200)
    return res


@bp.route('/api/jobs/infer/segment', methods=["POST"])
def queue_infer_segment():
    req = request.get_json()

    req['APP_OUTPUT_DIR'] = current_app.config['OUTPUT_DIR']
    from app import redis
    from abcTK.inference.segment import infer_segment

    q = Queue(connection=redis, serializer=dill)
    job = q.enqueue(infer_segment, req)

    res = make_response(jsonify({
            "message": "Segmentation inference submitted",
            "request": req,
            "job-ID": job.id})
            , 200)
    return res


@bp.route('/api/jobs/query_job', methods=["GET"])
def query_job():
    from app import redis
    #TODO Return job status + outputs given a job id
    job_id = request.args.get("id")

    job = Job.fetch(id=job_id, connection=redis)
    result = job.latest_result()
    print(result, result.type, flush=True)
    res = make_response(jsonify({
        'job-ID': job_id,
        'status': str(result.type),
        "result": str(result.return_value) if result.type == "Type.SUCCESSFUL" else result.exc_string
    }), 200)

    return res