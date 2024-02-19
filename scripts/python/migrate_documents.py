"""
Script to re-format documents in an "old" database to the format expected by a new schema 
"""

from pymongo import MongoClient


def main():
    client = MongoClient("mongodb://abc-user:abc-toolkit@localhost:27017/db?authSource=admin")
    database = client.db

    docs = database.images.find({})

    for doc in docs:
        print(doc)

if __name__ == '__main__':
    main()