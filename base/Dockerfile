FROM python:3.10

RUN apt-get update && apt-get upgrade -y
RUN apt-get install gcc ffmpeg libsm6 libxext6  -y

RUN python -m pip install -U pip

## Install requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt