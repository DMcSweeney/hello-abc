"""
Example script for submitting jobs to the toolkit
"""
import os
import time
import requests

## ABC config
host_ip = '10.127.4.12' ## DANTE
spine_url = f'http://{host_ip}:5001/api/jobs/infer/spine'
segment_url = f'http://{host_ip}:5001/api/jobs/infer/segment'

## DATA config
input_dir = "/mnt/server2/Sarcopenia/Bladder_andrewComp/dicom/"
project = 'Alex_Bladder'

def main():
    contents = []
    dirs = os.listdir(input_dir)

    ## Create all the requests
    for patient in dirs:
        path_in_abc = input_dir.replace('/mnt/server2/', '/data/inputs/')
        patient_path = os.path.join(path_in_abc, patient)

        spine_body = {"input_path": patient_path, "project": project, "patient_id": patient, 'series_uuid': patient}
        segment_body = {"input_path": patient_path, "project": project, "patient_id": patient, 'series_uuid': patient, "vertebra": 'L3', "num_slices": "1"}
        data = {'spine': spine_body, 'segment': segment_body}
        contents.append(data)

    ## Submit jobs
    for x in contents:
        print(f'Request: {x}')
        # Submit spine labelling job
        res = requests.post(spine_url, json=x['spine'])
        x['segment']['depends_on'] = res.json()['job-ID'] ## Update segment job with the job id 
        res = requests.post(segment_url, json=x['segment']) ## Submit segment job



    # rs = (grequests.post(spine_url, json=x['spine']) for x in contents)    
    # status = grequests.map(rs)
    #print(f'Spine finished in: {time.time() -start}')

    # start=time.time()
    # rs = (requests.post(segment_url, json=x['segment']) for x in contents)  
    # status = grequests.map(rs)
    # print(f'Spine finished in: {time.time() -start}')







if __name__ == '__main__':
    main()
