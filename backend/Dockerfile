FROM abc-base:latest

# Mount the data directory
VOLUME [ "/data" ]

#Set working dir
WORKDIR /src

RUN python -m pip install -U pip

# Is this still needed?
RUN export PATH="/home/appuser/.local/bin:$PATH" 

COPY ./models/ ./

#CMD [ "flask", "run" ]
CMD ["/bin/bash", "test_gpu.sh"]