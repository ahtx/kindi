# Kindi code image: the Kindi source on top of the dependency base image (docker/base/Dockerfile). Rebuild this one
# after every code change - it takes seconds. See docs/DOCKER.md.
#
#   docker/build-code.sh cuda   (or: docker build -t kindi:0.3.0 -f docker/Dockerfile .)
#   docker run -d --name kindi --gpus all -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data kindi:0.3.0
ARG BASE_IMAGE=kindi-base:0.3.0
FROM ${BASE_IMAGE}

COPY kindi /opt/kindi/kindi
WORKDIR /opt/kindi
# /data holds the converted model (pack/) and the cache (state/); see docs/DOCKER.md
ENV KINDI_PACK=/data/pack KINDI_STATE_DIR=/data/state KINDI_HOST=0.0.0.0
VOLUME ["/data"]
EXPOSE 8093
ENTRYPOINT ["python3", "-m", "kindi"]
CMD ["serve"]
