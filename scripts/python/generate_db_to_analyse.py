"""
Script for generating sqlite database to use in batch processing
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

schema = """CREATE TABLE IF NOT EXISTS dicomdb (
    id integer PRIMARY KEY,
    experiment_id text NOT NULL,
    uri text NOT NULL
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


def get_experiments(session, project_id):
    ## Get all subjects in project
    return session.get(f'{xnat_url}/data/projects/{project_id}/experiments/').json()


def filter_on_most_files():
    with requests.Session() as sess:
        sess.auth = ('donal', 'mu99le6')
        expts = get_experiments(sess, project_id)
    data = {}
    for exp in expts['ResultSet']['Result']:
        if exp['ID'] in paths_to_skip:
            print(f"Skipping {exp['ID']}")
            continue


        with requests.Session() as sess:
            sess.auth = ('donal', 'mu99le6')
            scans = sess.get(f"{xnat_url}/data/experiments/{exp['ID']}/scans").json()
        tmp = {}
        for scan in scans['ResultSet']['Result']:
            with requests.Session() as sess:
                sess.auth = ('donal', 'mu99le6')
                info = sess.get(f"{xnat_url}/data/experiments/{exp['ID']}/scans/{scan['ID']}?format=json").json()
            num_files = info['items'][0]['children'][0]["items"][0]["data_fields"]["file_count"]
            tmp[f"{xnat_url}/data/experiments/{exp['ID']}/scans/{scan['ID']}"] = num_files
        ## Sort dict by num files and return element with most
        key = max(tmp, key=tmp.get)
        data[exp['ID']] = key

        update = {
            'experiment_id': exp['ID'],
            'uri': key 
        }
        columns = ', '.join(update.keys())
        placeholders = ':'+', :'.join(update.keys())
        sql = """INSERT INTO dicomdb (%s) VALUES (%s)""" % (columns, placeholders)
        cursor.execute(sql, update)
        conn.commit()
    

def main():
    global conn, cursor
    conn = create_connection('./outputs/baselineScans.db')
    cursor = conn.cursor()
    create_table(conn, schema)

    # Check the above worked
    res = cursor.execute("SELECT name from sqlite_master").fetchone()
    assert res is not None, "Database doesn't exist"
    
    ## Get filepaths already analysed
    global paths_to_skip
    cursor.row_factory = lambda cursor, row: row[0]
    paths_to_skip = cursor.execute(f"SELECT DISTINCT experiment_id from dicomdb").fetchall()
    cursor.row_factory = sqlite3.Row
    print(f"----- Paths to skip: {len(paths_to_skip)} -----")

    filter_on_most_files()


if __name__ == '__main__':
    main()