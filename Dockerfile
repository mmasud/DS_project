FROM python:3.8.1-slim

# Adds metadata to the image as a key value pair example LABEL version="1.0"
LABEL maintainer="Masud Pervez <masud.pervez@yahoo.com>"
LABEL version="0.1"
LABEL description="This image is for creating a Data Science Environment."

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

COPY requirements.txt ./
RUN apt-get update &&\
    python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

#Setup File System
RUN mkdir ds
ENV HOME=/ds
ENV SHELL=/bin/bash
VOLUME /ds
WORKDIR /ds
COPY . /ds

#COPY run_jupyter.sh /run_jupyter.sh
#RUN chmod +x /run_jupyter.sh

# Open Ports for Jupyter
EXPOSE 7745

# Run the shell
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=7745", "--no-browser", "--allow-root"]
#CMD ["/run_jupyter.sh"]


