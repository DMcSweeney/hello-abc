"""
Script for parsing database files to collect errors, then re-submitting jobs to debug
"""

import sqlite3
import polars as pl
import requests

toolkit_url = 'http://10.127.3.26:5001'

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f'SQLITE version:', sqlite3.version)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn, schema):
    try:
        c = conn.cursor()
        c.execute(schema)
    except sqlite3.Error as e:
        print(e)

def main():
    task = 'spine'
    if task == 'spine':
        conn = create_connection('./outputs/baselineSpine.db')
        cursor = conn.cursor()
        df = pl.read_database("SELECT * FROM spinedb WHERE status_code != 200", conn)
    elif task == 'segmentation':
        ...
    
    print(df)
    paths_to_skip = ['/data/inputs/stampede_ACE/arc001/9050_CT_2/SCANS/1/DICOM', 
                    '/data/inputs/stampede_ACE/arc001/9096_CT_1/SCANS/1/DICOM',
                    '/data/inputs/stampede_ACE/arc001/37036_CT_1/SCANS/3/DICOM' ]

    for _id, exp_id, uri, path, status_code in df.rows(named=False):
        if status_code in [200, 700, 900]:
            continue 

        if task == 'spine':
            data = {'input_path': path, 'project': 'testingBaselineCTs'}
            if 'secondary' in path:
                # Skip secondary files! They don't have expected dicom headers
                payload = {'_id': _id, 'experiment_id': exp_id, 'uri': uri, 'path': path}
                sql = f"""UPDATE spinedb SET status_code = 900 WHERE id = {_id}"""
                print(sql)
                cursor.execute(sql, payload)
                conn.commit()
                continue
            if path in paths_to_skip: 
                ## These paths don't have dicoms (just catalog.xml), need to fix this
                payload = {'_id': _id, 'experiment_id': exp_id, 'uri': uri, 'path': path}
                sql = f"""UPDATE spinedb SET status_code = 900 WHERE id = {_id}"""
                print(sql)
                cursor.execute(sql, payload)
                conn.commit()
                continue
                


            headers = {"Content-Type": "application/json", 'Accept':'application/json'}
            print(f'Submitting data -- {data} -- with headers {headers}')
            response = requests.post(f"{toolkit_url}/api/infer/spine", json=data, headers=headers)

            if response.status_code == 800:
                ## Runtime Error because LoadImage can read directory
                print(response.json()['message'], flush=True)
                payload = {'_id': _id, 'experiment_id': exp_id, 'uri': uri, 'path': path}
                sql = f"""UPDATE spinedb SET status_code = 800 WHERE id = {_id}"""
                print(sql)
                cursor.execute(sql, payload)
                conn.commit()
                continue
            
            if response.json()['message'] == 'No centroids detected':
                ## Update status code to 700
                payload = {'_id': _id, 'experiment_id': exp_id, 'uri': uri, 'path': path}
                sql = f"""UPDATE spinedb SET status_code = 700 WHERE id = {_id}"""
                print(sql)
                cursor.execute(sql, payload)
                conn.commit()
                continue

            if response.status_code == 200:
                print('Fixed it!')
                payload = {'_id': _id, 'experiment_id': exp_id, 'uri': uri, 'path': path}
                sql = f"""UPDATE spinedb SET status_code = 200 WHERE id = {_id}"""
                print(sql)
                cursor.execute(sql, payload)
                conn.commit()
                continue

        
        #break



if __name__ == '__main__':
    main()