FROM python:3.8
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python
RUN pip install numpy
VOLUME /mnt/data
WORKDIR /mnt/data
CMD ["/bin/bash"]