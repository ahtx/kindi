# Docker

This directory holds the image build ([Dockerfile](Dockerfile), [Dockerfile.rocm](Dockerfile.rocm))
and the monitoring compose stack. The build steps are in [docs/DOCKER.md](../docs/DOCKER.md).

## Monitoring

`docker compose up -d` here runs the Kindi server together with Prometheus and Grafana. See
[docs/MONITORING.md](../docs/MONITORING.md) for the full walk-through, the metric reference and how
the numbers are collected.
