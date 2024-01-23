FROM python:3.10

RUN apt-get update && apt-get upgrade -y
RUN apt-get install gcc

# Mount the data directory
VOLUME [ "/data" ]

#Set working dir
WORKDIR /src

RUN python -m pip install -U pip

# Clone ABC-core
## RUN git clone https://github.com/DMcSweeney/abc-core.git abcCore

# Is this still needed?
RUN export PATH="/home/appuser/.local/bin:$PATH" 

# Copy app scripts and models
# COPY app/* .
# App folder is mounted from host for live updates

COPY ./models/ ./

## Install requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

CMD [ "flask", "run" ]
