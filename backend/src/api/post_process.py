"""
Script containing post-processing endpoints
"""
import os
import logging
from flask import Blueprint, make_response, jsonify, request
from app import mongo
import SimpleITK as sitk
import numpy as np
import json

from rt_utils import RTStructBuilder


bp = Blueprint('api/post_process', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/post_process/get_rt_struct', methods=["POST"])
def getRTStruct():
    _id = request.args.get("_id")
    project = request.args.get("project")

    database = mongo.db 
    response = database.segmentation.find_one({f"_id": _id, "project": project}, {"output_dir": 1, "path": 1})
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
    
@bp.route('/api/post_process/get_stats', methods=["POST"])
def get_stats():
    req = request.get_json()
    _id = req['_id']
    project = req['project']

    if 'format' in req:
        ###
        ...
    else:
        req['format'] = 'voxels'

    vertebra = 'L3' ##TODO This should be a variable
    
    database = mongo.db
    response = database.segmentation.find_one({"_id": _id, "project": project})
    data = {}
    for type_, dict_ in response["statistics"].items():
        if req['format'] == 'metric':
            logger.info("Converting areas to mm2")

            spacing = database.images.find_one({"_id": _id, "project": project}, {'pixel_spacing': 1})
            print('SPACING', spacing, flush=True)
            
        areas = [x['area (voxels)'] for x in dict_.values()]
        densities = [x['density (HU)'] for x in dict_.values()]
        data[type_] = {'mean-area': np.mean(areas), 'stdev-area': np.std(areas),
                        'mean-density': np.mean(densities), 'stdev-density': np.std(densities),
                         **dict_}

    res = make_response(jsonify({
        "message": "Stats returned successfully",
        "data": data
    }), 200)
    return res
