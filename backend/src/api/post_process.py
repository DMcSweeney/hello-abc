"""
Script containing post-processing endpoints
"""
import os
import logging
from flask import Blueprint, make_response, jsonify, request, current_app
from app import mongo
import SimpleITK as sitk
import numpy as np
import json
import polars as pl

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
    
@bp.route('/api/post_process/get_stats_for_series', methods=["POST"])
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

            spacing = database.images.find_one({"_id": _id, "project": project}, {'X_spacing': 1, 'Y_spacing': 1, 'slice_thickness': 1})
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

@bp.route('/api/post_process/get_stats_for_project', methods=["POST"])
def get_stats_for_project():
    req = request.get_json()
    project = req['project']
    vertebra = 'L3'

    database = mongo.db
    response = database.segmentation.find({"project": project})

    df = pl.DataFrame()
    for doc in response:
        spacing = database.images.find_one({"project": project, "_id": doc["_id"]}, {'X_spacing': 1, 'Y_spacing': 1, 'slice_thickness': 1})
        qc = database.quality_control.find_one({"project": project, "_id": doc["_id"]}, {'quality_control': 1})
        if not qc:
            logger.warn(f"{doc['_id']} did not pass quality control, skipping")
            continue

        for type_, dict_ in doc["statistics"].items():     
            for slice_, value in dict_.items():
                slice_num = int(slice_.lstrip('Slice'))

                row = {"patient_id": doc["patient_id"], "series_uuid": doc["series_uuid"], #"acquisition_date": doc["acquisition_date"],
                       "compartment": type_, "area": value['area (voxels)'], "density": value['density (HU)'],
                       "slice_number": slice_num, "X_spacing": float(spacing['X_spacing']), "Y_spacing": float(spacing['Y_spacing']), "slice_thickness": float(spacing['slice_thickness']),
                        "spine_qc": qc["quality_control"]["SPINE"], "segmentation_qc": qc["quality_control"][vertebra]}
                tmp = pl.DataFrame(row)
                df = pl.concat([df, tmp])

    if df.is_empty():
        res = make_response(jsonify({
            "message": "No examples passed quality control.",
        }), 400)
        return res

    output_filename = os.path.join(current_app.config['OUTPUT_DIR'], project, 'statistics.csv') 
    df.write_csv(output_filename)

    res = make_response(jsonify({
        "message": "Stats returned successfully",
        "output_file": output_filename
    }), 200)
    return res
