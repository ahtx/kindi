# Backends



Kindi's inference backends sit behind one plugin contract, so adding a model is

a contained engine-path change: one backend module plus docs, with no edits to

the CLI. Serving it over HTTP is a separate integration (see "Serving boundary"

below).



This slice lands the contract itself: the interface, the registry, the

deterministic fake backend and `kindi backends`. The existing engine families

are ported in follow-up adapter PRs.

## Built-in adapters

The five family adapters are stacked PRs on this contract; the contract alone
registers only `FakeBackend`. Their status here is CPU-test verification only:
none of the adapter PRs ran a GPU gate, and no verify/bench parity is claimed.

| adapter | model id | GPU-gate status |
|---|---|---|
| Bonsai (`kindi/backends/bonsai.py`) | `bonsai2-*` | Not run for the adapter; CPU tests only. The legacy engine's earlier evidence is not an adapter gate. |
| Gemma 4 (`kindi/backends/gemma4.py`) | `gemma4-e4b` | Blocked: no Gemma 4 pack on the host. CPU tests only; no GPU run. |
| Nemotron Lightning (`kindi/backends/lightning.py`) | `nemotron-lightning-30b` | Not run in the adapter PR. CPU tests only; no verify/bench parity. |
| Nanbeige (`kindi/backends/nanbeige.py`) | `nanbeige-33b` | Blocked: the catalog GGUF is corrupt. CPU tests only; no GPU run. |
| DeepSeek V4.1 (`kindi/backends/dsv41.py`) | `deepseek-v41-flash` | Blocked: no V4.1 weights. CPU tests only; no GPU run. |



## The contract



`kindi/backends/base.py` defines:



- `BackendCapabilities`: name, model ids, GGUF architectures, devices, max

  context, thinking/tools support, the tool-call family key (`tool_format`),

  speculation, token-filter and JSON-grammar support, cache kind, max

  concurrent sequences and greedy determinism.

- `SamplingParams`: max tokens, temperature, top-p/top-k, min-p, stop strings,

  seed and an optional `token_filter`.

- `token_filter(state)` receives `{"prompt_ids": [...], "generated": [...]}`

  with the real prompt ids.

- `GenerationEvent`: one yielded item, a token, a reasoning piece or the final

  `done` event with finish reason, timing and speculation stats.

- `CancelToken`: a flag polled during prompt processing and decode; a plain

  `threading.Event` satisfies it.

- `StopScanner`: decoded-text stop matching over only the not-yet-scanned tail,

  including a stop string that straddles two decoded pieces.

- `Backend`: lifecycle (`load`/`close`/`info`), text primitives (`tokenize`,

  `detokenize`, `apply_chat_template`), `generate`, `cache_stats`, and the

  optional hooks `park`, `reference_gguf`, `verify_prompts`.



The contract is dependency-free and importable on a CPU-only host, including

the Python 3.6 CUDA image. No CuPy import happens at module import time.



## Registry



`kindi/backends/registry.py`:



- `register_backend(factory, ...)` registers a backend class or zero-arg

  factory. Duplicate names are refused.

- `resolve(model_id_or_gguf_path, device=None, config=None)` returns a backend

  instance. A `.gguf` path is matched by its `general.architecture` metadata; a

  model id is matched against declared model ids; `device` filters by declared

  device names. When nothing matches, or when two backends match equally well,

  it raises `BackendNotFound` naming the candidates.

- `kindi.engine.make_engine` asks each registered adapter's

  `model_id_for_manifest` hook, resolves the pack through the registry and logs

  the backend and model id it chose. It refuses instead of guessing when a

  manifest is claimed by two adapters, or when a pack names a family no adapter

  serves. A dense pack that no adapter claims remains the legacy-builder

  fallback, and the log names the manifest fields to check.

- External backends load through `KINDI_BACKEND_MODULES=pkg.mod,pkg2.mod`.

  Importing the module registers the backend. Python entry points are deferred

  because `importlib.metadata` does not exist on Python 3.6.



`kindi/modelids.py` is the one manifest-to-model-id resolver, shared by the
gateway and `model_name: auto`; it reads the hook from each registered adapter
module, so a family cannot be added to one path and forgotten in the other.



## FakeBackend



`kindi/backends/fake.py` is a deterministic backend for tests: byte-level

tokenization, seeded pseudo-random generation or scripted output, controllable

per-step latency and cancel support. It implements the whole contract with no

GPU, weights or tokenizer files. The contract tests

(`tests/backends/test_contract.py`) run over the fake backend and auto-extend

to every registered backend.



## Inspecting registered backends



`kindi backends` prints one line per registered backend, including external

backends loaded from `KINDI_BACKEND_MODULES`; it does not load a pack or touch

the GPU. `kindi backends --json` prints the same registry rows as JSON.



## Serving boundary



`kindi serve` is not a generic host for every registry backend. Adding a

backend module makes it resolvable by the engine gateway, not automatically

servable over HTTP; an HTTP integration must supply the prompt cache, chat

template, streaming and metrics contract this server uses.



## Authoring a backend



1. Subclass `kindi.backends.Backend` and set `capabilities`.

2. Implement lifecycle, text primitives, `generate` and `cache_stats`.

3. Call `register_backend` when the module is imported.

4. Add a contract test and verify with `kindi backends --json`.



Tracked as T4.1/T4.3 (experimental).
