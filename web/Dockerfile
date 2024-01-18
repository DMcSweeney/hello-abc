FROM python:3.10

RUN apt-get update && apt-get upgrade -y
RUN apt-get install gcc

# Mount the data directory
VOLUME [ "/data" ]

#Set working dir
WORKDIR /app

RUN python -m pip install -U pip

# Is this still needed?
RUN export PATH="/home/appuser/.local/bin:$PATH" 

# # Copy app scripts and models
# COPY app/* .
## ./app/ on host is mounted to /app inside container to enable live-updates


## Install requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

CMD [ "flask", "run" ]

