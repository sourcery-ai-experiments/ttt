# nvidia's cuda image based
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04@sha256:85fb7ac694079fff1061a0140fd5b5a641997880e12112d92589c3bbb1e8b7ca
COPY requirements.txt /requirements.txt
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y --no-install-recommends python3-venv tzdata && apt-get clean all && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel && \
    /venv/bin/pip install --no-cache-dir -r /requirements.txt
WORKDIR /app
COPY ttt.py /app
USER 65532
ENTRYPOINT ["/venv/bin/python3","-u","/app/ttt.py"]
