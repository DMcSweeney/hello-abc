#!/bin/bash

docker run -it --rm\
    -v "/home/donal/ABC-toolkit/web-abc/data:/data"\
    -p 5001:5001\
    dmcsweeney/abc-spine:latest