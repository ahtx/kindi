#!/usr/bin/env bash
# Rebuild the Kindi code images: the Kindi source on top of the base images. Run this after every code change (and
# after build-base.sh when the dependencies changed). The base images must already exist. See docs/DOCKER.md.
#
#   docker/build-code.sh           build both code images (the default)
#   docker/build-code.sh cuda      build kindi:<version>
#   docker/build-code.sh rocm      build kindi-rocm:<version>
set -euo pipefail

VERSION="${KINDI_VERSION:-0.3.0}"
DOCKER="${DOCKER:-docker}"

cd "$(dirname "$0")/.."

build_cuda() { "$DOCKER" build -t "kindi:${VERSION}" --build-arg "BASE_IMAGE=kindi-base:${VERSION}" -f docker/Dockerfile .; }
build_rocm() { "$DOCKER" build -t "kindi-rocm:${VERSION}" --build-arg "BASE_IMAGE=kindi-base-rocm:${VERSION}" -f docker/Dockerfile.rocm .; }

case "${1:-all}" in
  all)         build_cuda; build_rocm ;;
  cuda|nvidia) build_cuda ;;
  rocm|amd)    build_rocm ;;
  *) echo "usage: $0 [cuda|rocm|all]" >&2; exit 2 ;;
esac
