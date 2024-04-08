# micromaba with cuda 11.8
FROM ghcr.io/mamba-org/micromamba:jammy-cuda-11.8.0@sha256:913c780905bdc9a76ab2b78711c1cab69ff93b3cb9e32be511ee81ece1507a88
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app
COPY ttt.py /app

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python","-u","/app/ttt.py" ]
