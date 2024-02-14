"""
Script containing post-processing endpoints
"""
import os
import logging
from flask import Blueprint, make_response, jsonify, request
from app import mongo
import SimpleITK as sitk
import numpy as np

from rt_utils import RTStructBuilder


bp = Blueprint('api/post_process', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/post_process/get_rt_struct', methods=["POST"])
def getRTStruct():
    _id = request.args.get("_id")
    
    database = mongo.db 
    response = database.segmentation.find_one({f"_id": _id}, {"output_dir": 1, "path": 1})
    print(response, flush=True)
    path_to_preds = os.path.join(response['output_dir'], 'masks')

    ## Go through output dir and read all masks into one RT-Struct
    Masks = {filename.rstrip('.nii.gz'): sitk.ReadImage(os.path.join(path_to_preds, filename)) \
             for filename in os.listdir(path_to_preds) if filename.endswith('.nii.gz')}

    # Point RTSTruct Builder to input dicom
    rtstruct = RTStructBuilder.create_new(dicom_series_path=response['path'])
    for name, Mask in Masks.items():
        mask = sitk.GetArrayFromImage(Mask).astype(bool) ## Need to re-order axes to match 
        mask = np.moveaxis(np.transpose(mask), 0, 1)
        logger.info(f"Adding {name} to RT-Struct. Shape: {mask.shape}")
        rtstruct.add_roi(mask=mask, name=name)

    ## Save RT struct and reply
    output_path = os.path.join(path_to_preds, 'rt-struct')
    rtstruct.save(output_path)

    res = make_response(jsonify({
        "message": "RT-Struct successfully generated.",
        "output_path": output_path
    }), 200)

    return res
    
@bp.route('/api/post_process/get_stats_metric', methods=["POST"])
def get_stats_metric():
    
    
    
    ...
