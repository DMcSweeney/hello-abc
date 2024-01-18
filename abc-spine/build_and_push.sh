# remove old image
docker image rm dmcsweeney/abc-spine:latest
# build image
docker build -t dmcsweeney/abc-spine:latest . --no-cache
