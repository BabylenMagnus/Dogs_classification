FROM ubuntu

COPY app.py ./app.py
COPY engine.py ./engine.py
COPY utils.py ./utils.py
COPY config.py ./config.py
COPY templates/index.html ./templates/index.html
COPY weights ./weights/

RUN apt-get update && apt-get -y install python3-pip

COPY requirements.txt ./requirements.txt
COPY old_weights/classification_75.model ./weights/classification_75.model

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113


RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
