# micromaba

FROM ghcr.io/mamba-org/micromamba:latest@sha256:216fa8949f9d2d12cc1d16c5e4a60a704555eda35d689875a4002d1fc9f4aebd

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app
# This is the arrdn filter model hogwash for ffmpeg noise reduction
copy sh.rnnn /app
COPY ttt.py /app

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python","-u","/app/ttt.py" ]
