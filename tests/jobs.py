

import rq
import requests
import redis

def count_words_at_url(url):
    resp = requests.get(url)
    return len(resp.text.split())

def main():

    REDIS = redis.Redis(host='localhost', port=6379)
    q = rq.Queue(connection = REDIS)

    q.enqueue(count_words_at_url, 'http://nvie.com')



if __name__ == '__main__':
    main()