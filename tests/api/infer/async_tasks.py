"""
Testing asynchronous requests
"""
import os
import time
import requests


url = 'http://localhost:5001/api/infer/segment'

def main():
    input_dir = '/home/donal/ABC-toolkit/web-abc/data/inputs/'
    contents = []
    files = os.listdir(input_dir)
    for file in files:
        body = {"input_path": os.path.join('/data/inputs', file), "project": "testing", "vertebra": 'L3', "slice_number": "5", "num_slices": "1"}
        contents.append(body)#
    start=time.time()
    for x in contents:
        print(f'Request: {x}')
        res =requests.post(url, json=x)
    #rs = (requests.post(url, json=x) for x in contents)
        
    #status = grequests.map(rs)
    print(f'Finished in: {time.time() -start}')

if __name__ == '__main__':
    main()

