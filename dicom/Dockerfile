FROM python:3.10

RUN apt-get update && apt-get upgrade -y
RUN apt-get install gcc

# Mount the data directory
VOLUME [ "/data" ]

WORKDIR /src

RUN python -m pip install -U pip

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

## Run unbuffered
CMD ["python", "-u", "start_server.py"] 