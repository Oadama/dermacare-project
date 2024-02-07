FROM python:3.10.6-bullseye

WORKDIR /dermasaaj_project

COPY . /dermasaaj_project/
COPY model model/

COPY requirements.txt /requirements.txt
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install requests --no-cache-dir
RUN pip install --ignore-installed uvicorn==0.27.0
RUN pip install --ignore-installed pandas==1.4.4
RUN pip install --ignore-installed fastapi==0.109.0
RUN pip install --ignore-installed tensorflow==2.10.0
RUN pip install --ignore-installed google-cloud-storage==2.14.0
RUN pip install --ignore-installed google-auth==2.27.0
RUN pip install python-multipart==0.0.6
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python==4.7.0.72

CMD uvicorn backend:app --host 0.0.0.0 --port $PORT
