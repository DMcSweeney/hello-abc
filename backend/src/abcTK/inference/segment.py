"""
Set of endpoints for segmentation side
"""

import os
import numpy as np
import logging
import SimpleITK as sitk

import dill
from flask import Blueprint
from rq import Queue
from rq.job import Job


from abcTK.segment.engine import segmentationEngine
import abcTK.database.collections as cl


bp = Blueprint('api/segment', __name__)
logger = logging.getLogger(__name__)

#########################################################
#* ==================== API =============================
#########################################################

#@bp.route('/api/infer/segment', methods=["GET", "POST"])
def infer_segment(req):
    """
    Endpoint for handling segmentation requests. Only handles one image, one level and one modality at a time.
    Make multiple requests if needed. 
    """

    # Import here to prevent issues when deserializing
    logger.info(f"Request received: {req}")

    ## Check required params
    check_params(req, required_params=["input_path", "project", "vertebra"])

    req['loader_function'], loader_name = get_loader_function(req['input_path'])
    if type(req["vertebra"]) == list:
        logger.error("Make multiple requests to use multiple models.")
        raise ValueError("Vertebra should be a string representing a single level. Make multiple requests to use different models.")

    req = handle_request(req)
    output_dir = os.path.join(req['APP_OUTPUT_DIR'], req["project"], req["patient_id"], req["series_uuid"])
    os.makedirs(output_dir, exist_ok=True)
    req['output_dir'] = output_dir

    ## ++++++++++++++++   INFERENCE  +++++++++++++++++++++
    logger.info(f"Processing request: {req}")
    engine = segmentationEngine(**req)
    data, paths_to_sanity = engine.forward(**req)
    ###### UPDATE DATABASE ########
    update_database(req, data, paths_to_sanity)
    
    return 



########################################################
#* =============== HELPER FUNCTIONS =====================

def update_database(req, data, paths_to_sanity):
    from app import mongo

    database = mongo.db
    query = database.quality_control.find_one({'_id': req['series_uuid'], 'project': req['project']},
                                                {"_id": 1, "quality_control": 1, "paths_to_sanity_images": 1})
    
    qc = {req['vertebra']: 2}

    if query is not None: #If an entry exists
    ## Update with existing values
        for k, v in query['paths_to_sanity_images'].items():
            paths_to_sanity[k] = v
        
        for k, v in query['quality_control'].items():
            qc[k] = v


    ## Insert into database
    segmentation_update = cl.Segmentation(_id=req['series_uuid'], project=req['project'], input_path=req['input_path'], 
                                            patient_id=req['patient_id'], series_uuid=req['series_uuid'], output_dir=req['output_dir'], statistics=data,
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
    

def handle_request(req):
    from app import mongo 

    dcm_files = [x for x in os.listdir(req['input_path']) if x.endswith('.dcm')]
    if len(dcm_files) == 0:
        raise ValueError(f"No dicom files found in input path: {req['input_path']}")
    
    header_keys = {
        'patient_id': '0010|0020',
        'study_uuid': '0020|000D',
        'series_uuid': '0020|000e',
    }
    items = read_dicom_header(os.path.join(req["input_path"], dcm_files[0]), header_keys=header_keys)

    ## Add to request
    for key, val in items.items():
        if key in req:
            logger.info(f"{key} provided in request, ignoring DICOM header.")
            continue
        if key == 'patient_id' and val == '':
            raise ValueError("Patient ID not found in DICOM header. Please provide with request.")
        if key == 'series_uuid' and val == None:
            raise ValueError("Series UUID not found in DICOM header. Please provide with request.")
        req[key] = val

    # WORLDMATCH OFFSET 
    # TODO HANDLE MR OFFSET - not sure of details
    if 'worldmatch_correction' not in req:
        logger.info("Worldmatch correction (-1024 HU) will not be applied. Overwrite with 'worldmatch_correction' in request.")
        req['worldmatch_correction'] = False
    
    # BONE MASKS
    if 'generate_bone_mask' not in req:
        logger.info("Bone mask will be regenerated. This might slow things down. Overwrite with 'bone_mask' in request (True-> regenerate; False-> skip).")
        req['generate_bone_mask'] = True
    elif req['generate_bone_mask'] == str:
        try:
            req['generate_bone_mask'] = bool(req['generate_bone_mask'])
        except:
            # If can't be converted to bool assume path#
            logger.info("Path to bone mask provided. Will not regenerate.")

    # SLICE NUMBER
    if "slice_number" not in req:
        ## Check the spine collection for vertebra
        match = mongo.db.spine.find_one({"_id": req['series_uuid']})
        if match is None or req["vertebra"] not in match["prediction"]:
            raise ValueError("Could not find a slice number for the requested vertebra.")
        req['slice_number'] = match["prediction"][req["vertebra"]][-1]
        logger.info(f"Found slice number {req['slice_number']} for {req['vertebra']}")

    if type(req['slice_number']) == str:
        req['slice_number'] = int(req['slice_number'])
    
    # MODALITY
    if "modality" not in req:
        ## If user doesn't provide modality, add default (CT)
        #TODO should this come from header? Might not handle CBCTs?
        logger.info("Assuming default modality: CT")
        req["modality"] = "CT"
    
    # NUM SLICES AROUND REFERENCE
    if "num_slices" not in req:
        ## If user doesn't provide modality, add default (CT)
        logger.info("Only segmenting reference slice.")
        req["num_slices"] = 0
    elif type(req["num_slices"]) == str:
        req['num_slices'] = int(req['num_slices'])

    return req


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
            raise ValueError('Input is not a directory of dicom files. Please use full path if using other format.')

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
        raise ValueError(f"Some required parameters are missing. Did you provide the following? {required_params}") ## Bad request
