"""
Set of endpoints for spine labelling
"""
import os
import numpy as np
import torch
from flask import Blueprint, request, make_response, abort, jsonify, current_app
import SimpleITK as sitk
import json
import logging

from abcTK.spine.server import spineApp
from abcTK.writer import sanityWriter
from app import mongo
#import models


bp = Blueprint('api/spine', __name__)
logger = logging.getLogger(__name__)

#########################################################
#* ==================== API =============================
#########################################################

@bp.route('/api/infer/spine', methods=["GET", "POST"])
def infer_spine():
    if request.method == 'POST':
        req = request.get_json()
        logger.info(f"Request received: {req}")

        if not torch.cuda.is_available():
            logger.error("No GPU detected")
            abort(500) ## Internal server error 

        check_params(req, required_params=["input_path", "project"])
        req['loader_function'] = get_loader_function(req['input_path'])

        ## Start the spineApp
        app = init_app()

        if 'series_uuid' not in req:
            # Infer scan id from dicom
            logger.info("Reading header from first dicom file")
            dcm_files = [x for x in os.listdir(req['input_path']) if x.endswith('.dcm')]
            items = read_dicom_header(os.path.join(req["input_path"], dcm_files[0]), header_keys={'series_uuid': '0020|000e'})
            req['series_uuid'] = items['series_uuid']
            logger.info(f"series_uuid not provided. Reading from DICOM header: {req['series_uuid']}")
            if req['series_uuid'] is None:
                abort(400)

        if 'patient_id' not in req:
            # Infer patient id from dicom
            logger.info("Reading header from first dicom file")
            dcm_files = [x for x in os.listdir(req['input_path']) if x.endswith('.dcm')]
            items = read_dicom_header(os.path.join(req["input_path"], dcm_files[0]), header_keys={'patient_id': '0010|0020'})
            req['patient_id'] = items['patient_id']
            logger.info(f"Patient ID was not provided. Reading from DICOM header: {req['patient_id']}")
            if req['patient_id'] == '':
                logger.error("Patient ID not found in DICOM header. Please provide with request.")
                abort(500)
        
        output_dir = os.path.join(current_app.config['OUTPUT_DIR'], req["project"], req['patient_id'], req["series_uuid"])
        os.makedirs(output_dir, exist_ok=True)

        # +++++ INFERENCE +++++
        response = app.infer(request = {"model": "vertebra_pipeline", "image": req['input_path']})
        logger.info(f"Spine labelling complete: {response}")
        
        res, output_filename = handle_response(req['input_path'], response, output_dir, req['loader_function'][0])

        #######  UPDATE DATABASE #######
        database = mongo.db # Access the database

        # Insert image info into the images collection (patID, seriesUID, project, path and what has "happened" to the data)
        payload = {
            "_id": req['series_uuid'] ,"patientID": req['patient_id'], "project": req['project'], "path": req['input_path'], "series_uuid": req['series_uuid'],
            "spine_labelling_done": True
        }
        #payload_id = database.data.insert_one(payload).inserted_id
        database.images.replace_one({"_id": req['series_uuid'] }, payload, upsert=True)
        logger.info(f"Inserted {payload} into collection: images")
        # Then update the results collections

        payload = {
            "_id": req['series_uuid'] ,"patientID": req['patient_id'], "project": req['project'], "path": req['input_path'], "series_uuid": req['series_uuid'],
            "message": res.json['message'],
            "path_to_sanity_image": output_filename,
            "prediction": res.json['prediction'], "all_parameters": {k: str(v) for k, v in req.items()}
        }
        database.spine.replace_one({"_id": req['series_uuid']}, payload, upsert=True)
        logger.info(f"Inserted {payload} into collection: spine")

        return res

    elif request.method == "GET":
        # Return some help
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


def init_app():
    # Initialise the spine app
    
    app_dir = os.path.dirname(__file__)
    studies =  "https://127.0.0.1:8989" #Just points to an empty address - needed for monaiLabelApp
    config = {
        "models": "find_spine,find_vertebra,segment_vertebra",
        "preload": "false",
        "use_pretrained_model": "true"
    }   
    return spineApp(app_dir, studies, config)

def json_to_file(json_payload, output_path, filename='spine-output.json'):
    """
    Convert the json output with levels into a mask.
    Not ideal for storage but better integration with XNAT and reduces transform-related errors
    """
    output_filename = os.path.join(output_path, filename)
    with open(output_filename, 'w') as f:
        json.dump(json_payload, f)

def handle_response(image_path, res, output_dir, loader_function):
    """
    Handle the reply from inference 
    """
    json_output_path = os.path.join(output_dir, 'json')
    os.makedirs(json_output_path, exist_ok=True)
    writer = sanityWriter(output_dir, slice_number=None, num_slices=None)

    label = res["file"]
    label_json = res["params"]

    if label is None and label_json is None:
        #* This is the case if no centroids were detected
        logger.error("No centroids detected")
        res = make_response(jsonify({
            "message": "No centroids detected"
        }), 500)
        output_filename = None
    
    elif label is None and label_json is not None:
        ## Prettify the json
        json_to_file(label_json, json_output_path, filename='all-spine-outputs.json')
        pretty_json = prettify_json(label_json)
        json_to_file(pretty_json, json_output_path)
        output_filename = writer.write_spine_sanity('SPINE', image_path, pretty_json, loader_function)

        res = make_response(jsonify({
            "message": f"Labelling finished succesfully. Output written to: {json_output_path}",
            "quality_control": f"QC image was written to: {output_filename}",
            "prediction": pretty_json
        }), 200)
    
    else:
        # This should never happen... Would like to add this though
        logger.error("Somehow you got here.\
                      This means the spine module tried to write vertebral masks, which hasn't been implemented.\
                      I have no idea how you managed that... Well done!!")
        res = make_response(jsonify({
            "message": "Amazing error, you broke everything"
        }), 500)
        output_filename = None

    return res, output_filename

def prettify_json(input_json):
    ## Clean spine app predictions
    labels, centroids = input_json['label_names'], input_json['centroids']
    vert_lookup = {val: key for key, val in labels.items()}
    dict_ = {}
    for centroid in centroids:
        for val in centroid.values():
            level = vert_lookup[val[0]]
            dict_[level] = [x for x in val[1:]]
    return dict_