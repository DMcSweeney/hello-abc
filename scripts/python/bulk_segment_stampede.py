"""
Script for reading db of spine requests, skip failures and send jobs to segment successes
"""

import os
import requests
import sqlite3

toolkit_url = 'http://10.127.3.158:5001' ## Toolkit url
xnat_url = 'http://10.127.3.158:8080'

input_db = './outputs/baselineSpine.db'
output_db = './outputs/baselineSegmentation.db'

schema = """CREATE TABLE IF NOT EXISTS segmentdb (
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
    ## Fetch successes
    cursor.row_factory = sqlite3.Row
    success_rows = cursor.execute(f"SELECT DISTINCT * from spinedb WHERE status_code=200").fetchall()
    print(f"----- Paths to process: {len(success_rows)} -----")

    # Init db for this task
    conn = create_connection(output_db)
    cursor = conn.cursor()
    create_table(conn, schema)

    ## Get experiments to skip
    cursor.row_factory = lambda cursor, row: row[0]
    paths_to_skip = cursor.execute(f"SELECT DISTINCT experiment_id from segmentdb").fetchall()
    cursor.row_factory = sqlite3.Row
    print(f"----- Paths to skip: {len(paths_to_skip)} -----")

    for row in success_rows:
        if row['experiment_id'] in paths_to_skip:
            print(f"Skipping: {row['experiment_id']}")
            continue

        
        #### SUBMIT JOB
        data = {'input_path': row['path_to_scan'], 'project': 'testingBaselineCTs',
                 'vertebra': 'L3', 'num_slices': 1}
        headers = {"Content-Type": "application/json", 'Accept':'application/json'}
        print(f'Submitting data -- {data} -- with headers {headers}')
        response = requests.post(f"{toolkit_url}/api/infer/segment", json=data, headers=headers)

        payload = {'experiment_id': row['experiment_id'], 'uri': row['uri'], 
                   'path_to_scan': row['path_to_scan'], 'status_code': response.status_code}
        print(payload)
        columns = ', '.join(payload.keys())
        placeholders = ':'+', :'.join(payload.keys())
        sql = """INSERT INTO segmentdb (%s) VALUES (%s)""" % (columns, placeholders)
        cursor.execute(sql, payload)
        conn.commit()

        #break


if __name__ == '__main__':
    main()