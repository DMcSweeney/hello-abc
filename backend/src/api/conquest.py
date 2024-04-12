"""
Conquest endpoints

"""

import logging
from flask import Blueprint, request

bp = Blueprint('/api/conquest', __name__)
logger = logging.getLogger(__name__)



@bp.route('/api/conquest/handle_trigger', methods=["GET", "POST"])
def handle_trigger():
    if request.method == 'GET':
        series_uid = request.args.get("series_uid")

        logger.info(f"Trigger received for series: {series_uid}")
        return '<h1> Hello </h1>'