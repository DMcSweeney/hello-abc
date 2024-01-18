"""
Set of API for abc-spine
"""
import os
import torch
from flask import Blueprint, request, make_response, abort
from abcCore.abc.spine.app import spineApp
import json

bp = Blueprint('api', __name__)

######################################
#* =================API===============
######################################


@bp.route('/api/infer', methods=["GET", "POkST"])
def infer():
    if request.method == 'POST':
        req = request.get_json()
        if not torch.cuda.is_available():
            abort(500) ## Internal server error 

        check_inference_params(req)

        app = init_app()

    print('I can see you', flush=True)
    return '<h1>Hi</h1>'


####################################################
#* ============== HELPER FUNCTIONS =================
######################################################

def check_cuda_available():
    # Check GPU is accessible
    return torch.cuda.is_available()


def check_inference_params(req):
    ## Check all args have been provided to inference call
    ## Input path, project name, patient id, scan id 
    if 'input' in req:
        ...
    else:
        abort(400) ## Bad request


def init_app():
    """
    Initialise the spine app
    """
    app_dir = os.path.dirname(__file__)
    print(f"APP_DIR: {app_dir}")
    studies =  "https://127.0.0.1:8989" #Just points to an empty address - needed for monaiLabelApp
    config = {
        "models": "find_spine,find_vertebra,segment_vertebra",
        "preload": "false",
        "use_pretrained_model": "true"
    }   
    return spineApp(app_dir, studies, config)

def json_to_file(json_payload, output_path):
    """
    Convert the json output with levels into a mask.
    Not ideal for storage but better integration with XNAT and reduces transform-related errors
    """
    output_filename = os.path.join(output_path, 'prediction.json')
    with open(output_filename, 'w') as f:
        json.dump(json_payload, f)

def handle_response(res, output_path):
    """
    Handle the reply from inference 
    """
    label = res["file"]
    label_json = res["params"]

    if label is None and label_json is None:
        #* This is the case if no centroids were detected
        return 500
        
    elif label is None and label_json is not None:
        json_to_file(label_json, output_path)
        return 200
    else:
        # This should never happen... Would like to add this though
        return 500
    
####################################################
#* ===================ERROR HANDL