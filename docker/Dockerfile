# syntax=docker/dockerfile:1
# Kindi runtime image for NVIDIA cards and the CPU engine. CUDA 10.2 supports the Quadro K420 (sm_30) and the GTX 1080 (sm_61); later cards run Kindi's PTX through the
# driver. CUDA 10.2 is the last toolkit with Kepler sm_30 support, and CuPy 9.6 the last CuPy built for it; the host
# needs the NVIDIA driver (470 for Kepler) and the NVIDIA Container Toolkit. See docs/DOCKER.md.
#
#   docker build -t kindi:0.6.0 -f docker/Dockerfile --build-arg KINDI_VERSION=0.6.0 --build-arg KINDI_REVISION=$(git rev-parse HEAD) .
#   docker run -d --name kindi --gpus all -p 127.0.0.1:8093:8093 -v kindi-data:/data -e KINDI_MODEL=bonsai-4b kindi:0.6.0
#
# The base image is pinned by digest, and the CUDA installer by sha256, so a rebuild gives the same toolchain.
FROM ubuntu:18.04@sha256:152dc042452c496007f07ca9127571cb9c29697f42acbfad72324b2bb2e43c98

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates libxml2 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CUDA 10.2 toolkit (runtime libraries, cuBLAS, NVRTC - Kindi compiles its kernels at first start). The static
# libraries, NPP and nvJPEG are not used at run time and are removed in the same layer.
ARG CUDA_RUN=cuda_10.2.89_440.33.01_linux.run
ARG CUDA_SHA256=560d07fdcf4a46717f2242948cd4f92c5f9b6fc7eae10dd996614da913d5ca11
RUN wget -q --tries=5 --timeout=30 https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/${CUDA_RUN} \
    && echo "${CUDA_SHA256}  ${CUDA_RUN}" | sha256sum -c - \
    && sh ${CUDA_RUN} --silent --toolkit --no-opengl-libs --override \
    && rm -f ${CUDA_RUN} \
    && rm -rf /usr/local/cuda-10.2/doc /usr/local/cuda-10.2/samples /usr/local/cuda-10.2/nsight* \
              /usr/local/cuda-10.2/libnsight /usr/local/cuda-10.2/libnvvp /usr/local/cuda-10.2/extras \
    && rm -f /usr/local/cuda-10.2/lib64/*.a /usr/local/cuda-10.2/lib64/libnpp* /usr/local/cuda-10.2/lib64/libnvjpeg*
ENV PATH=/usr/local/cuda-10.2/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

# gcc stays: the CPU engine builds kcpu.c with gcc -fopenmp on first use. (CuPy's fastrlock dependency builds from
# source on Python 3.6 and needs python3-dev for this step only.)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && python3 -m pip install --no-cache-dir --upgrade "pip<21.4" "setuptools<60" \
    && python3 -m pip install --no-cache-dir "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" \
       "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2" "Jinja2==3.0.3" "MarkupSafe==2.0.1" \
    && apt-get purge -y python3-dev && apt-get autoremove -y \
    && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# a fixed non-root user (1000:1000) owns /data; the installer passes its own -u, rootless Docker/Podman use -u 0:0
RUN groupadd -g 1000 kindi && useradd -u 1000 -g 1000 -m -d /home/kindi -s /usr/sbin/nologin kindi \
    && mkdir -p /data && chown 1000:1000 /data

COPY kindi /opt/kindi/kindi
WORKDIR /opt/kindi
# /data holds the models (models/), the packs (packs/ or pack/) and the cache (state/); see docs/DOCKER.md
ENV KINDI_PACK=/data/pack KINDI_STATE_DIR=/data/state KINDI_HOST=0.0.0.0 HOME=/home/kindi
VOLUME ["/data"]
EXPOSE 8093
USER 1000:1000

ARG KINDI_VERSION=dev
ARG KINDI_REVISION=unknown
LABEL org.opencontainers.image.title="Kindi (NVIDIA CUDA)" \
      org.opencontainers.image.description="Kindi: local LLM inference on low-cost GPUs. NVIDIA cards from Kepler on (CUDA 10.2), and the CPU engine." \
      org.opencontainers.image.source="https://github.com/ahtx/kindi" \
      org.opencontainers.image.url="https://amirhusain.com/kindi" \
      org.opencontainers.image.documentation="https://github.com/ahtx/kindi/blob/main/docs/DOCKER.md" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.vendor="Amir Husain" \
      org.opencontainers.image.version="${KINDI_VERSION}" \
      org.opencontainers.image.revision="${KINDI_REVISION}" \
      org.opencontainers.image.base.name="docker.io/library/ubuntu:18.04"
ENTRYPOINT ["python3", "-m", "kindi.container"]
CMD ["serve"]
