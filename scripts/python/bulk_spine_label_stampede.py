"""
Script for reading files from CSV and requesting spine labelling 
"""

import os
import requests
import polars as pl
import json
import sqlite3

toolkit_url = 'http://10.127.3.38:5001' ## Toolkit url
xnat_url = 'http://10.127.3.38:8080'

## Path inside container
project_id = 'BaselineCT' #Source project
path_to_archive = '/data/inputs/'
path_to_csv = '/home/donal/stampede/baselineAnalysis/outputs/filtered_ACEG.csv'


schema = """CREATE TABLE IF NOT EXISTS spinedb (
    id integer PRIMARY KEY,
    experiment_id text NOT NULL,
    uri text NOT NULL,
    path_to_scan NOT NULL,
    status_code NOT NULL
    );"""

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
    ## Connect to db
    conn = create_connection('./outputs/baselineSpine.db')
    cursor = conn.cursor()
    create_table(conn, schema)

    ## Get experiment IDs
    cursor.row_factory = lambda cursor, row: row[0]
    paths_to_skip = cursor.execute(f"SELECT DISTINCT experiment_id from spinedb").fetchall()
    cursor.row_factory = sqlite3.Row
    print(f"----- Paths to skip: {len(paths_to_skip)} -----")


    dbConn = create_connection("./outputs/baselineScans.db")
    df = pl.read_database("SELECT * FROM dicomdb", dbConn)
    # Request processing
    for id_, exp_id, uri in df.rows(named=False):
        if exp_id in paths_to_skip:
            #print(f'Skipping: {exp_id}')
            continue
        
        # Get path in archive!
        with requests.Session() as sess:
            sess.auth = ('donal', 'mu99le6')
            res = sess.get(f"{uri}?format=json").json()
        path = os.path.dirname(res['items'][0]['children'][0]['items'][0]['data_fields']['URI'])
        path = path.replace('/data/xnat/archive/', path_to_archive)

        #### SUBMIT JOB
        data = {'input_path': path, 'project': 'testingBaselineCTs'}
        headers = {"Content-Type": "application/json", 'Accept':'application/json'}
        print(f'Submitting data -- {data} -- with headers {headers}')
        response = requests.post(f"{toolkit_url}/api/infer/spine", json=data, headers=headers)

        #####


        payload = {'experiment_id': exp_id, 'uri': uri, 
                   'path_to_scan': path, 'status_code': response.status_code}
        print(payload)
        columns = ', '.join(payload.keys())
        placeholders = ':'+', :'.join(payload.keys())
        sql = """INSERT INTO spinedb (%s) VALUES (%s)""" % (columns, placeholders)
        cursor.execute(sql, payload)
        conn.commit()


if __name__ == '__main__':
    main()