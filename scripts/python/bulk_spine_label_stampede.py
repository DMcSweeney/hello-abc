"""
Script for reading files from CSV and requesting spine labelling 
"""

import os
import requests
import polars as pl
import json

toolkit_url = 'http://10.127.3.38:5001' ## Toolkit url
xnat_url = 'http://10.127.3.38:8080'

## Path inside container
path_to_archive = '/data/inputs/'
path_to_csv = '/home/donal/stampede/baselineAnalysis/outputs/filtered_ACEG.csv'

def get_paths(df):    
    paths = {}
    ## Get request for scan based on URI
    for row in df.rows(named=True):
        uri = row['scanURI']
        print(uri)
        with requests.Session() as sess:
            sess.auth = ('donal', 'mu99le6')

            res = sess.get(f"{xnat_url}{uri}?format=json")
            
            if res.status_code != 200:
                print(f"Error requesting scan resources {uri}: {res.status_code}")
                continue

            res = res.json()
            #print(res)

            ## Filter based on number of files
            if res['items'][0]['children'][0]['items'][0]['data_fields']['file_count'] <= 10:
                print(f"Too few files {res['items'][0]['children'][0]['items'][0]['data_fields']['file_count'] }, omitting")
                continue
            path = os.path.dirname(res['items'][0]['children'][0]['items'][0]['data_fields']['URI'])
            path = path.replace('/data/xnat/archive/', path_to_archive)
            
            paths[uri] = path

            break
        
    return paths

def main():
    df = pl.read_csv(path_to_csv)
    paths = get_paths(df)
    print(paths)
    ## Request processing
    print(f"Attempting to submit {len([x for x in paths.keys()])} requests")
    for uri, path in paths.items():
        
        data = {'input_path': path, 'project': 'testingBaselineCTs', 'series_uuid': 'dave'}
        headers = {"Content-Type": "application/json", 'Accept':'application/json'}
        print(f'Submitting data -- {data} -- with headers {headers}')
        response = requests.post(f"{toolkit_url}/api/infer/spine", json=data, headers=headers)
        print(response)
        


if __name__ == '__main__':
    main()