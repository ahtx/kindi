#!/usr/bin/env bash
# Rebuild the Kindi base images: the dependencies only (OS, CUDA / ROCm, Python, CuPy, FastAPI) without the Kindi
# code. Run this only when a dependency changes; the code image is built by build-code.sh on top. See docs/DOCKER.md.
#
#   docker/build-base.sh           build both base images (the default)
#   docker/build-base.sh cuda      build kindi-base:<version>
#   docker/build-base.sh rocm      build kindi-base-rocm:<version>
set -euo pipefail

VERSION="${KINDI_VERSION:-0.3.0}"
DOCKER="${DOCKER:-docker}"

cd "$(dirname "$0")/.."

build_cuda() { "$DOCKER" build -t "kindi-base:${VERSION}" -f docker/base/Dockerfile .; }
build_rocm() { "$DOCKER" build -t "kindi-base-rocm:${VERSION}" -f docker/base/Dockerfile.rocm .; }

case "${1:-all}" in
  all)         build_cuda; build_rocm ;;
  cuda|nvidia) build_cuda ;;
  rocm|amd)    build_rocm ;;
  *) echo "usage: $0 [cuda|rocm|all]" >&2; exit 2 ;;
esac
