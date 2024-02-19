"""
Script to re-format documents in an "old" database to the format expected by a new schema 
"""
import os
import SimpleITK as sitk
from pymongo import MongoClient
from datetime import datetime
from database import collections as cl


def handle_request(req):
    ## Handle paramaters and extract info from dicom header if not provided.
    dcm_files = [x for x in os.listdir(req['input_path']) if x.endswith('.dcm')]
    if len(dcm_files) == 0:
        raise ValueError
    
    header_keys = {
        #'patient_id': '0010|0020',
        'study_uuid': '0020|000D',
        #'series_uuid': '0020|000e',
        'pixel_spacing': '0028|0030',
        'slice_thickness': '0018|0050',
        'acquisition_date': '0008|0022'
    }
    items = read_dicom_header(os.path.join(req["input_path"], dcm_files[0]), header_keys=header_keys)

    ## Add to request
    for key, val in items.items():
        if key in req:
            continue
        if key == 'patient_id' and val == '':
            raise ValueError
        if key == 'series_uuid' and val == None:
            raise ValueError
        
        if key == 'acquisition_date' and val is not None:
            val = datetime.strptime(val, '%Y%m%d').date().strftime('%d-%m-%Y')

        if key == 'pixel_spacing' and "\\" in val:
            val = val.split('\\')
            val = [float(x) for x in val]
            req['X_spacing'] = val[0]
            req['Y_spacing'] = val[1]
            continue

        req[key] = val

    return req

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

def update_collection(database, document, collection):
    if collection == 'images':
        params = handle_request(req={'input_path': document['path']})


        update = cl.Images(_id=document['_id'], project=document['project'], patient_id=document['patientID'],
                           input_path=document['path'], series_uuid=document['series_uuid'],
                           labelling_done=document['spine_labelling_done'], **params
                           )

        database.replace_one({'_id': update['_id']}, update.__dict__, upsert=True)


def main():
    client = MongoClient("mongodb://abc-user:abc-toolkit@10.127.3.158:5002/db?authSource=admin")
    database = client.db

    docs = database.images.find({})

    for doc in docs:
        print(doc)
        update_collection(database.images, doc, 'images')
        break



if __name__ == '__main__':
    main()