"""
Script for parsing database files to collect errors, then re-submitting jobs to debug
"""

import sqlite3
import polars as pl
import requests

toolkit_url = 'http://10.127.3.158:5001'

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

    for _id, exp_id, uri, path, status_code in df.rows(named=False):
        if status_code == 700:
            # Skip if no centroids error
            continue 

        if task == 'spine':
            data = {'input_path': path, 'project': 'testingBaselineCTs'}
            headers = {"Content-Type": "application/json", 'Accept':'application/json'}
            print(f'Submitting data -- {data} -- with headers {headers}')
            response = requests.post(f"{toolkit_url}/api/infer/spine", json=data, headers=headers)
            print(response.json()['message'])
            if response.json()['message'] == 'No centroids detected':
                ## Update status code to 7
                payload = {'_id': _id, 'experiment_id': exp_id, 'uri': uri, 'path': path}
                
                print(payload)
                sql = f"""UPDATE spinedb SET status_code = 700 WHERE id = {_id}"""
                print(sql)
                cursor.execute(sql, payload)
                conn.commit()

                continue
        
        break



if __name__ == '__main__':
    main()