# Kindi base image: the dependencies without the Kindi code. Rebuild this only when a dependency changes; the code
# image (docker/Dockerfile) layers the Kindi source on top. See docs/DOCKER.md.
#
# CUDA 10.2 supports the Quadro K420 (sm_30) and the GTX 1080 (sm_61). CUDA 10.2 is the last toolkit with Kepler
# sm_30 support, and CuPy 9.6 the last CuPy built for it; the host needs the NVIDIA 470 driver (the last branch that
# supports Kepler) and the NVIDIA Container Toolkit.
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates libxml2 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CUDA 10.2 toolkit (runtime libraries, cuBLAS, NVRTC - Kindi compiles its kernels at first start)
RUN wget -q --tries=5 --timeout=30 https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run \
    && sh cuda_10.2.89_440.33.01_linux.run --silent --toolkit --no-opengl-libs --override \
    && rm -f cuda_10.2.89_440.33.01_linux.run \
    && rm -rf /usr/local/cuda-10.2/doc /usr/local/cuda-10.2/samples /usr/local/cuda-10.2/nsight* \
              /usr/local/cuda-10.2/libnsight /usr/local/cuda-10.2/libnvvp /usr/local/cuda-10.2/extras
ENV PATH=/usr/local/cuda-10.2/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

# (CuPy's fastrlock dependency builds from source on Python 3.6: the compiler is installed for this step only)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && python3 -m pip install --no-cache-dir --upgrade "pip<21.4" "setuptools<60" \
    && python3 -m pip install --no-cache-dir "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" \
       "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2" "Jinja2==3.0.3" "MarkupSafe==2.0.1" \
    && apt-get purge -y python3-dev && apt-get autoremove -y \
    && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
