FROM python:3.8.1-slim

LABEL maintainer="Masud Pervez"
LABEL version="0.1"
LABEL description="This image is about SEB Data Science challenge."

COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /SEB_Project

COPY . /SEB_project/

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]



