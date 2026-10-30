# Monitoring Kindi

A `docker-compose` stack that runs the Kindi server and exposes its runtime metrics in a Grafana
dashboard. It makes no change to the Kindi code or image: a `json-exporter` sidecar translates
Kindi's JSON `GET /health` into Prometheus metrics, Prometheus stores the history, and Grafana
renders a provisioned dashboard.

```
kindi ──(/health JSON)──► json-exporter ──(/probe)──► prometheus ──► grafana
 :8093                      :7979                  :9090          :3000
```

## Prerequisites

- The `kindi:0.3.0` image, built as described in [DOCKER.md](DOCKER.md).
- A converted model pack, as produced in [INSTALL.md](INSTALL.md).
- Docker with the NVIDIA Container Toolkit (or the AMD device passthrough) to run the `kindi`
  service itself; see [DOCKER.md](DOCKER.md) and [AMD.md](AMD.md).

## Run

```sh
cd docker
docker compose up -d
```

- Grafana: http://localhost:3000 (user `admin`, password `admin` unless `GRAFANA_ADMIN_PASSWORD`
  is set).
- Prometheus: http://localhost:9090
- Kindi API: http://localhost:8093/v1

The `Kindi` dashboard is provisioned automatically. Stop with `docker compose down`; add `-v` to
also drop the stored history.

## Configuration

- `KINDI_DATA_DIR` (default `~/.kindi`) — the host folder mounted at `/data`; it holds the converted
  model in `pack/` and the cache in `state/`.
- GPU passthrough: the compose file ships with NVIDIA (`gpus: all`); the AMD variant is in the
  comments in [`docker/docker-compose.yml`](../docker/docker-compose.yml).
- Grafana runs with `GF_LOG_LEVEL=warn` (warnings and errors only) and a configurable admin
  password.

## Metrics

Kindi reports these values in `/health`, which the exporter exposes as gauge metrics:

| Metric | Meaning |
|---|---|
| `kindi_uptime_s` | server uptime, with `model` and `head` labels |
| `kindi_gpu_free_mib` / `kindi_gpu_total_mib` | GPU memory (absent on a CPU-only host); allocated = total − free |
| `kindi_stats_requests` | total requests served |
| `kindi_stats_prompt_tokens` / `kindi_stats_cached_tokens` / `kindi_stats_completion_tokens` | cumulative token counts |
| `kindi_stats_prefill_seconds` / `kindi_stats_decode_seconds` | cumulative prompt and decode wall-clock time |
| `kindi_busy_generated` etc. | live progress, present only while a request runs |

## How the numbers are collected

`/health` mixes three kinds of value, and none of them is a "delta since the last call":

- **Cumulative counters** (`stats.*`). Monotonic totals since the server started; each completed
  request adds to them. The increase between two samples is computed downstream, not emitted.
- **Momentary snapshots** (`busy`, `gpu_*`, `uptime_s`). Sampled at the instant `/health` is
  called. `busy` exists only while a request is running and is `null` otherwise; the GPU figures
  are queried from the driver on every call.
- **Derived rates.** The `tokens/s` panels are not in `/health` at all; Prometheus computes them
  from the counters with `rate()`.

Two separate intervals are involved:

| Interval | Value | Where | Meaning |
|---|---|---|---|
| Scrape interval | 5s | [`prometheus.yml`](../docker/prometheus.yml) | how often `/health` is sampled and recorded |
| Rate window | 5m | the `[5m]` in the PromQL queries | the look-back over which each rate is averaged |

So a `tokens/s` number at any instant is a trailing five-minute average of the slope of the
corresponding counter, rebuilt from samples taken every five seconds. Lower `scrape_interval` for
finer sampling, or shorten the `[5m]` window to make the rates react faster.

## Throughput panels

The dashboard derives these from the counters, so they measure active work rather than the
idle-inclusive average that `rate(kindi_stats_completion_tokens[5m])` gives:

- **Prefill tokens/s** = `(rate(kindi_stats_prompt_tokens[5m]) − rate(kindi_stats_cached_tokens[5m])) / rate(kindi_stats_prefill_seconds[5m])`.
- **Generation tokens/s** = `rate(kindi_stats_completion_tokens[5m]) / rate(kindi_stats_decode_seconds[5m])`.

## Notes

- `json-exporter` emits gauges only, so the cumulative counters have no native counter type; use
  `rate()`/`increase()` in PromQL and treat a sudden decrease as a server restart, not a counter
  reset.
- The `kindi_busy_*` metrics are absent while the server is idle; the dashboard derives a 0/1
  "busy" signal with `count(kindi_busy_generated) OR vector(0)`.
