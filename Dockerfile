FROM python:3.8
RUN apt-get update
RUN pip install opencv-python
VOLUME /mnt/data
WORKDIR /mnt/data
CMD ["/bin/bash"]