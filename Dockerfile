FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install --no-install-recommends -y python3 python3-opencv
WORKDIR /code
