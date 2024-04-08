FROM python:3.10

RUN apt-get update && apt-get upgrade -y
RUN apt-get install gcc ffmpeg libsm6 libxext6  -y

WORKDIR /src

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["rq", "worker"]