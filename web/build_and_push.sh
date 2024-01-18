# remove old image
docker image rm dmcsweeney/abc-web:latest
# build image
docker build -t dmcsweeney/abc-web:latest . --no-cache
