"""
Set of endpoints for segmentation side
"""

import os
import numpy as np
import logging
import SimpleITK as sitk
import time
import json
import base64

from flask import Blueprint, request, make_response, abort, jsonify, current_app
from celery import Celery

from abcTK.segment.engine import segmentationEngine
import abcTK.database.collections as cl


from app import mongo


bp = Blueprint('api/segment', __name__)
logger = logging.getLogger(__name__)

#########################################################
#* ==================== API =============================
#########################################################

@bp.route('/api/infer/segment', methods=["GET", "POST"])
def infer_segment():
    """
    Endpoint for handling segmentation requests. Only handles one image, one level and one modality at a time.
    Make multiple requests if needed. 
    """

    if request.method == 'POST':
        req = request.get_json()
        logger.info(f"Request received: {req}")
        
        ################## HANDLE REQUEST #########################################
        ## Check required params
        check_params(req, required_params=["input_path", "project", "vertebra"])

        req['loader_function'], loader_name = get_loader_function(req['input_path'])

        if type(req["vertebra"]) == list:
            logger.error("Make multiple requests to use multiple models.")
            abort(400) # Bad request

        if "modality" not in req:
            ## If user doesn't provide modality, add default (CT)
            logger.info("Leaving modality as default: CT")
            req["modality"] = "CT"
        
        if "num_slices" not in req:
            ## If user doesn't provide modality, add default (CT)
            logger.info("Only segmenting reference slice.")
            req["num_slices"] = 0
        elif type(req["num_slices"]) == str:
            req['num_slices'] = int(req['num_slices'])

        if 'series_uuid' not in req:
            # Infer scan id from dicom
            # Filter dicom 
            if loader_name == 'dicom':
                # Infer scan id from dicom
                logger.info("Reading header from first dicom file")
                dcm_files = [x for x in os.listdir(req['input_path']) if x.endswith('.dcm')]
                items = read_dicom_header(os.path.join(req["input_path"], dcm_files[0]), header_keys={'series_uuid': '0020|000e'})
                req['series_uuid'] = items['series_uuid']
                logger.info(f"series_uuid not provided. Reading from DICOM header: {req['series_uuid']}")
                if req['series_uuid'] is None:
                    abort(400)
            else:
                logger.warn("Input is not dicom, assuming series_uuid = filename. Overwrite with 'series_uuid' in request.")
                ext = os.path.splitext(req['input_path'])
                req['series_uuid'] = os.path.basename(req['input_path']).replace(ext, '')

            logger.info(f"++++++++ Series UUID is: {req['series_uuid']} ++++++++++++++")

        if 'patient_id' not in req:
            # Infer patient id from dicom
            logger.info("Reading header from first dicom file")
            dcm_files = [x for x in os.listdir(req['input_path']) if x.endswith('.dcm')]
            items = read_dicom_header(os.path.join(req["input_path"], dcm_files[0]), header_keys={'patient_id': '0010|0020'})
            req['patient_id'] = items['patient_id'].strip('')
            logger.info(f"Patient ID was not provided. Reading from DICOM header: {req['patient_id']}")
            if req['patient_id'] == '':
                logger.error("Patient ID not found in DICOM header. Please provide with request.")
                abort(500)

        if 'worldmatch_correction' not in req:
            logger.info("Worldmatch correction (-1024 HU) will not be applied. Overwrite with 'worldmatch_correction' in request.")
            req['worldmatch_correction'] = False

        if 'generate_bone_mask' not in req:
            logger.info("Bone mask will be regenerated. This might slow things down. Overwrite with 'bone_mask' in request (True-> regenerate; False-> skip).")
            req['generate_bone_mask'] = True
        elif req['generate_bone_mask'] == str:
            try:
                req['generate_bone_mask'] = bool(req['generate_bone_mask'])
            except:
                # If can't be converted to bool assume path#
                logger.info("Path to bone mask provided. Will not regenerate.")


        if "slice_number" not in req:
            ## Check the spine collection for vertebra
            match = mongo.db.spine.find_one({"_id": req['series_uuid']})
            if match is None or req["vertebra"] not in match["prediction"]:
                abort(400, {"message": "Could not find a slice number for the requested vertebra."})
            req['slice_number'] = match["prediction"][req["vertebra"]][-1]
            logger.info(f"Found slice number {req['slice_number']} for {req['vertebra']}")

        if type(req['slice_number']) == str:
            logger.info("Slice number provided as a string, converting to int")
            req['slice_number'] = int(req['slice_number'])
        
        #############################################################################

        output_dir = os.path.join(current_app.config['OUTPUT_DIR'], req["project"], req["patient_id"], req["series_uuid"])
        os.makedirs(output_dir, exist_ok=True)
        req['output_dir'] = output_dir

        ## +++  INFERENCE  ++++
        logger.info(f"Processing request: {req}")
        engine = segmentationEngine(**req)
        start = time.time()
        data, paths_to_sanity = engine.forward(**req)
        end = time.time()
        
        ## Response should include parameters used: modality, model name, stats too?
        res = make_response(jsonify({
            "message": f"Segmentation finished succesfully. Outputs written to: {output_dir}",
            "parameters": f"{[(x, y) for x, y in req.items()]}",
            "statistics": data,
            "time": f"Pipeline took {round(end-start, 3)} seconds"
        }), 200)

        ###### UPDATE DATABASE ########
        database = mongo.db
        query = database.quality_control.find_one({'_id': req['series_uuid'], 'project': req['project']},
                                                   {"_id": 1, "quality_control": 1, "paths_to_sanity_images": 1})
        
        print(query, flush=True)
        ## Update with existing values
        for k, v in query['paths_to_sanity_images'].items():
            paths_to_sanity[k] = v
        qc = {req['vertebra']: 2}
        for k, v in query['quality_control'].items():
            qc[k] = v


        ## Insert into database
        segmentation_update = cl.Segmentation(_id=req['series_uuid'], project=req['project'], input_path=req['input_path'], 
                                              patient_id=req['patient_id'], series_uuid=req['series_uuid'], output_dir=output_dir, statistics=data,
                                              all_parameters={k: str(v) for k, v in req.items()})
        qc_update = cl.QualityControl(_id=req['series_uuid'], project=req['project'], input_path=req['input_path'], patient_id=req['patient_id'],
                                      series_uuid=req['series_uuid'], paths_to_sanity_images=paths_to_sanity, quality_control=qc
                                      )

        database.images.update_one({"_id": req['series_uuid']}, {"$set": {"segmentation_done": True}}, upsert=True)
        logger.info(f"Set segmentation_done to True in collection: images")
        database.segmentation.update_one({"_id": req['series_uuid']}, {'$set': segmentation_update.__dict__}, upsert=True)
        logger.info(f"Inserted {segmentation_update.__dict__} into collection: spine")
        database.quality_control.update_one({"_id": req['series_uuid']}, {"$set": qc_update.__dict__}, upsert=True)
        logger.info(f"Inserted {qc_update.__dict__} into collection: quality_control")
        
        return res

    elif request.method == 'GET':
        return "<h1> Here's some help </h1>"

########################################################
#* =============== HELPER FUNCTIONS =====================
########################################################
    
def get_loader_function(path):
    def load_numpy(path):
        img = np.load(path)
        return sitk.GetImageFromArray(img)

    def load_nifty(path):
        #* Read nii volume
        return sitk.ReadImage(path)
    
    def load_dcm(path):
        #* Read DICOM directory
        reader = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dcm_names)
        return reader.Execute()
    

    if os.path.isdir(path):
        ## Check at least one dicom file
        logger.info("Input is a directory, assuming DICOM.")
        dcm = [x for x in os.listdir(path) if x.endswith(('dcm', 'DICOM'))]
        if len(dcm) != 0:
            return load_dcm, 'dicom' ## Loader function
        else:
            abort(400, {'message': 'Input is not a directory of dicom files. Please use full path if using other format.'})

    elif os.path.isfile(path):
        # If file, check extension.
        ext = os.path.splitext(path)
        logger.info(f"Input file extension: {ext}")
        if ext in ['.nii', '.nii.gz']:
            return load_nifty, 'nifty'
        elif ext in ['.npy', '.npz']:
            return load_numpy, 'numpy'
        
def read_dicom_header(path, header_keys):
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    metadata = {}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        if v == '': #Replace empty string
            v=None
        metadata[k] = v
    data = {}
    for key, val in header_keys.items():
        try:
            data[key] = metadata[val]
        except KeyError:
            data[key] = None # For sql
    return data

def check_params(req, required_params = ["input", "project", "patient_id", "scan_id"]):
    ## Check all args have been provided to inference call
    test = [x in req for x in required_params]
    if not all(test):
        logger.info(f"Some required parameters are missing. Did you provide the following? {required_params}")
        abort(400) ## Bad request
