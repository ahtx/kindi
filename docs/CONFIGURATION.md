# Configuration

Kindi reads its settings in this order - later sources win:

1. built-in defaults;
2. a config file: `--config FILE`, else `$KINDI_CONFIG`, else `./kindi.conf`, else `~/.kindi/kindi.conf` (INI, one
   `[kindi]` section). Unknown keys are an error, so typos do not pass silently;
3. environment variables `KINDI_<KEY>` (for example `KINDI_PORT=9000`, `KINDI_API_KEY=...`);
4. command-line flags (`--pack`, `--state-dir`, `--ctx`, and for `serve`: `--host`, `--port`, `--api-key`, `--head`).

`kindi info` prints the final values and the file that each value came from. The Docker image sets
`KINDI_PACK=/data/pack`, `KINDI_STATE_DIR=/data/state` and `KINDI_HOST=0.0.0.0`.

## Keys

### Paths

| key | default | |
|---|---|---|
| `pack` | `~/.kindi/pack` | the converted model (`kindi convert` writes it) and, optionally, `head_clusters.npz` |
| `state_dir` | `~/.kindi/state` | the kernel cache (`kernels/`), saved prompt openings (`openings/`) and the head mode set by `kindi head` |

### Server

| key | default | |
|---|---|---|
| `host` | `127.0.0.1` | address to listen on; `0.0.0.0` serves your network - then also set `api_key` |
| `port` | `8093` | |
| `api_key` | (empty) | if set, every `/v1` request needs `Authorization: Bearer <api_key>`; `/health` stays open |
| `queue_max` | `64` | generation requests (chat completions, messages) that may wait while every decode slot is busy; the next one gets HTTP 429 `server_busy` with `Retry-After: 1` before its prompt is read. `0`: no cap |
| `max_body_mb` | `32` | a larger request body gets HTTP 413 `request_too_large`; the body is not read past the cap |
| `rate_limit_per_min` | `0` | at most this many `/v1` requests per client address in any 60 s; more get HTTP 429 `rate_limit_exceeded` with the seconds to wait in `Retry-After`. `0`: off |
| `model_name` | `auto` | the model id `/v1/models` reports; `auto` takes it from the pack (bonsai2-2b, bonsai2-4b or bonsai2-9b); clients may send any model name |

### Engine

| key | default | |
|---|---|---|
| `ctx` | `24576` | context length in tokens (prompt + answer). For GPU memory, see the notes below the table |
| `slots` | `1` | requests written at the same time (`kindi serve --slots N`, `KINDI_SLOTS=N`, at most 8). Each slot is one conversation with its own K/V cache, recurrent state and prompt cache; one pass of the model writes one token for every slot that is writing, so the weight reads are shared. Each slot's greedy output is the same as when it runs alone. Needs the plan engine on one card with the tensor-core plans (`KINDI_TC=1`, and `KINDI_TP=1` for the 2B, 4B and 9B). The context fit gives every slot the same context. Speculative decoding (drafting: `KINDI_SPEC`, the MTP head, DSpine) is off with more than one slot; the log says so. The default is one slot, and nothing sets more by itself: the installer leaves it at 1 |
| `head` | `exact` | `exact` = full output head (token-exact with llama.cpp); `fast` = shortlist head (needs `kindi fast-head`). `kindi head exact\|fast` overrides it at run time |
| `fast_clusters` | `256` | clusters the fast head scores exactly (of 1,024). Fewer is faster and less exact (see the notes below the table) |
| `kv_cache` | `auto` | AH-Nanbeige4.3-33B only: the K/V cache, `auto`, `f16` or `q8_0` (`KINDI_KV_CACHE=q8_0`). `q8_0` keeps K and V as int8 with an fp16 scale for each 32 values: half the memory (64K tokens: about 6.6 GiB instead of 13 GiB), so the 33B gets a 64K context on one 24 GB card. Needs compute capability 6.1 or later. `auto` uses `f16` when the model and its f16 cache fit on the cards at `ctx`, and `q8_0` when they do not. `kindi verify` checks `q8_0` against llama.cpp: the tokens that agree with the f16 reference must be at least as many as llama.cpp's own q8_0 cache keeps (on 2x RTX 3080: Kindi 475, llama.cpp 471, of 586) |

Notes:

- `ctx`: GPU memory is ~12 KiB per token for the fp16 K/V cache, on top of ~570 MB for weights and buffers.
- `ctx`: 24,576 leaves ~115 MB of the K420's 980 MB free. Lower it if something else needs the card.
- `fast_clusters`: 128 clusters found the full head's top token 98.1 % of the time, 256 clusters 99.4 %.

### Prompt cache

| key | default | |
|---|---|---|
| `checkpoint_every` | `256` | interval between recurrent-state checkpoints (host RAM, ~19 MB each). Smaller = less to read again after a change, more RAM |
| `prefix_max` | `8` | saved openings kept on disk (each ~12 KiB per token + 19 MB; ~80 MB for a 5,000-token opening) |
| `prefix_min` | `512` | the shortest opening that Kindi saves |
| `park_max` | `3` | other conversations kept in host RAM (each: its checkpoints + its K/V cache) |
| `checkpoint_budget_mb` | `0` | host RAM for the checkpoints and the parked conversations; `0` = 20 % of the RAM. When the cache is full, Kindi drops copies in this order: the checkpoints that only a parked conversation holds, the oldest checkpoints of the active conversation, then whole parked conversations (the oldest first) |
| `turn_save` | `1` | save the state at the end of each turn to disk, in the background, so that after a restart a conversation resumes where its last turn ended. `0` = off |
| `turn_keep` | `4` | saved turns: the number of conversations kept on disk (the newest turn of each) |
| `turn_disk_mb` | `4096` | saved turns: the disk space that they can use. The K/V rows are in blocks of 1,024 tokens that the turns of one conversation share, so a new turn writes only its new rows and its recurrent state |
| `turn_min` | `64` | saved turns: the shortest conversation that Kindi saves (tokens) |

### API for agent harnesses

| endpoint | |
|---|---|
| `GET /v1/models` | the model and its real context: `context_length`, `max_model_len`, `max_context_length` and `meta.n_ctx` |
| `POST /v1/tokenize` (also `/tokenize`) | `{"content": text}` counts plain text. `{"messages": [...], "tools": [...]}` counts the whole chat prompt, as the chat endpoint builds it. The answer is `count`, `tokens`, `max_model_len` and `fits` |
| `POST /v1/detokenize` (also `/detokenize`) | `{"tokens": [...]}` gives the text (`prompt` and `content`) |
| `POST /v1/chat/completions` with `"dry_run": true` | the exact `usage.prompt_tokens` and `fits`, without generation |

A prompt that does not fit gets HTTP 400 at once, also while another request uses the engine. The body is
`{"error": {"code": "context_length_exceeded", "type": "invalid_request_error", "message": "This model's maximum
context length is ..."}}`, OpenAI's wording, so that litellm and the OpenAI SDKs report a context-window error and
the harness shortens the conversation. Note: harnesses that use litellm (terminus-2, hermes) do not read
`/v1/models`. Give them the window yourself (for example harbor's `model_info` with `max_input_tokens`), or they
assume a much larger one.

While a stream waits for the engine or reads a long prompt, Kindi sends an SSE comment every 10 s
(`KINDI_KEEPALIVE_S`): `: kindi progress {"phase": "queued"}`, then `{"phase": "reading prompt", "read": ...,
"to_read": ..., ...}`. SSE clients ignore comments. A non-streaming request can send `"kindi_keepalive": true` (or
the header `X-Kindi-Keepalive: 1`) to get a space every 10 s before its JSON answer. JSON allows the spaces, but
the status is then 200 before the answer is known, so an error comes as an `{"error": ...}` body with status 200.

### Generation defaults

Used when a request does not set them: `max_tokens` (`1024`), `temperature` (`0.7`; `0` = greedy), `top_p`
(`0.8`), `top_k` (`20`). These are the model publisher's recommended non-thinking settings.

Off by default: `min_p` (`0`) and `repeat_penalty` (`1.0`), with `repeat_last_n` (`64`; `-1` = the full context).
They work as in llama.cpp. When `repeat_penalty` is not 1.0, the output is not token-exact with the greedy output,
and Kindi does not use speculative drafts for that request. The Bonsai models use them. The Gemma 4 and Nanbeige backends
do not use them yet.

## Environment variables for the engine

These variables have no config key. Use them to test or to compare.

| variable | default | |
|---|---|---|
| `KINDI_DEVICE` | `gpu` | `cpu` selects the CPU engine for Bonsai 2B, 4B and 9B and for Flash-Next. The Bonsai backend also uses its CPU engine when CuPy is unavailable. |
| `KINDI_CPU_CACHE` | `~/.cache/kindi` | build cache for the native CPU libraries |
| `KINDI_FN_CPUS` | physical cores | CPU list for Flash-Next inference. For example, `0-19,40-59` selects those logical CPU IDs. |
| `KINDI_FNCPU_CHUNK` | `1024` | Tokens per Flash-Next CPU prefill chunk, from 1 to 4,096. Larger chunks compile with more CPU scratch capacity. |
| `KINDI_FNCPU_GEMM` | `16` | Flash-Next CPU prefill chunks with at least this many tokens use fp32 matrix products and tiled attention. `0` keeps the int8 kernels. |
| `KINDI_FNCPU_GEMM_EMIN` | `6` | Experts with fewer tokens in a prefill chunk use the int8 kernels. |
| `KINDI_FNCPU_ATTN_TILE` | `1` | `0` turns off the tiled prefill attention of the fp32 path. |
| `KINDI_FNCPU_BARRIER` | two-level | `flat` uses the thread pool's single-counter barrier instead of the per-NUMA-node barrier. |
| `KINDI_FNCPU_WARM_PLE` | `bg` | Warms the 28.8 GB per-layer embedding table into the page cache when RAM allows: `bg` in the background while serving, `1` before serving, `0` off. |
| `KINDI_FNCPU_K6` | `0` | `1` stores Q6_K and Q5_K matrices as 6-bit codes. Opt-in: no net write gain was measured. |
| `KINDI_FNCPU_PRIV` | `0` | `1` makes each worker quantize a decode step's input itself instead of waiting at a barrier. Opt-in: measured slower. |
| `KINDI_FNCPU_QUERY_PARALLEL` | `0` | `1` parallelizes whole attention queries during Flash-Next CPU prefill when a chunk contains at least one token per worker and the fp32 path is off. |
| `KINDI_FNCPU_QUERY_PAIR` | `0` | `1` shares key and value loads across two query heads with `KINDI_FNCPU_QUERY_PARALLEL=1`. |
| `KINDI_FNCPU_PACKED_GDN` | `1` | Packs Gated DeltaNet state into contiguous worker-local panels for Flash-Next CPU prefill when the chunk contains at least one token per worker. `0` turns it off. |
| `KINDI_QXG_PROFILE` | `1` | The 0.8 Flash-Next prompt-read profile, used on one Ampere or newer card with page-locked experts (`KINDI_QXG_PIN=1`). It sets the upload, ring, staging, expert-batch and selected-key attention settings by prompt length and fits the expert batch to the free card memory. In fast mode it also uses fp16 prompt GEMMs, the fused prompt kernels and causal key limits. `0` turns it off. A variable set in the environment overrides the profile's value. |
| `KINDI_QXG_STREAM_UPLOAD` | `packed-pipeline` with the prompt profile, else `legacy` | `packed-pipeline` coalesces adjacent cold-expert weights and overlaps down-projection uploads with gate/up compute in Flash-Next GPU prefill. |
| `KINDI_QXG_RING` | `64` with the prompt profile, else `32` | Expert slots in the double-buffered Flash-Next GPU prefill uploader. The coalesced pipeline was measured at `64`. |
| `KINDI_QXG_COMPACT_STREAM` | `1` with the prompt profile, else `0` | `1` reuses the same gate/up and activation scratch for each group of expert uploads during Flash-Next prefill. The RTX 3060 long-prompt test used it with `KINDI_QXG_LAYER_MAJOR=1` and `KINDI_QXG_LM_TOK=8192`. |
| `KINDI_QXG_PREFETCH_PREFIX` | `0` | Maximum cold experts to stage before their layer routes a batch of at least 4,096 Flash-Next prompt tokens. The matched RTX 3060 4K tests used `448`, compact scratch, and `KINDI_FN_EXPERT_GB=0.4`. |
| `KINDI_QXG_PREFETCH_ORDER` | `index` | `usage` ranks cold experts by the configured routing profile before picking which ones to upload early. |
| `KINDI_QXG_PREFETCH_PHASES` | `1` with the prompt profile, else `0` | `1` lets gate and up projections run as soon as their weights arrive, while the down weights are still uploading. |
| `KINDI_QXG_PREFETCH_SLICE` | `0` | Tokens per expert-compute slice within one staged layer batch. Zero uses the entire batch. |
| `KINDI_QXG_PREFETCH_COMPUTE_GROUP` | all ready experts | Maximum prefetched experts per compute group. The long-prompt RTX 3060 profile used `128`. |
| `KINDI_QXG_TENSOR_QSA` | `0` | `fp16` runs selected-key attention on tensor cores from 4,096 cached positions on Ampere and newer CUDA cards with Triton installed. The engine uses the standard attention path when Triton is unavailable. |
| `KINDI_LT_ADAPT_PREFILL` | `1` | `0` keeps Lightning at 512-token chunks. On, Lightning batches up to 1,024 prompt tokens on eligible cards that normally use 512. It reuses GPU scratch and preserves attention tile boundaries and single-token tails. Set a fixed `KINDI_LT_CHUNK` to override the adaptive path. |
| `KINDI_MTP_STAGE_PROMPT` | `1` with the prompt profile, else `0` | `1` lets Flash-Next stage experts for smaller layer-major MTP prompt batches when `KINDI_QXG_PREFETCH_PREFIX` is set. Warm verification graphs can stay resident through a read of at most 4,096 total tokens when at least 2 GiB are available. If the read exhausts GPU memory, Kindi releases the graphs and retries with staging off and a smaller batch. |
| `KINDI_QXG_WIN_FREE` | auto | `1` releases Flash-Next verification graphs before prompt prefill. `0` keeps them resident. The automatic policy releases them unless it can retain them for a short staged MTP read. |
| `KINDI_QXG_LM_TOK_DRAFT` | `3072` | Size of Flash-Next layer-major MTP expert batches. Larger batches can reuse expert weights across more tokens and need more GPU memory. |
| `KINDI_FN_DRAFT_K` | `2` | Maximum Flash-Next tokens proposed by the MTP head before target verification. |
| `KINDI_SPLIT_CARD` | `KINDI_SPLIT_STREAM` (12 by default) | Maximum streamed experts dispatched to the main GPU per DeepSeek prompt layer when using the GPU/CPU split. The RTX 3060 + M4000 replay tested 16 with 81,920 context and DSpark. |
| `KINDI_DSV_PREFILL_REUSE_UPLOADS` | `0` | `1` keeps up to five streamed experts in existing GPU upload slots between adjacent pipelined DeepSeek prompt chunks. The tested wider profile used `KINDI_SPLIT_CARD=16`. |
| `KINDI_FN_DRAFT_CACHE` | `0` | `1` keeps one pair of Flash-Next target and MTP prompt states in VRAM. An exact append can reuse completed prefill chunks, then rereads the previous partial chunk with the new suffix. |
| `KINDI_CKPT_STREAM` | `1` with decode slots, else `0` | `1` = the prompt cache copies the recurrent state (checkpoints, parked conversations, saved turns) on a side stream into pinned host memory, so that another slot's pass does not wait for the copy. Without slots it is a speed option for hosts with RAM to spare: the copies are about 10% faster (27B on a PCIe x1 link), but pinned memory then holds up to the whole checkpoint budget (`checkpoint_budget_mb`, default 20% of the RAM) |
| `KINDI_SLOTS_ROWS` | `1` | decode slots: `0` = the batched pass launches the per-token kernels once per slot (slower; for tests) |
| `KINDI_SLOTS_WAIT_MS` | `2` | decode slots: how long a writing slot waits for the others to hand in their tokens before a pass runs without them |
| `KINDI_TP` | (auto) | `1` = the plan engine, `0` = the single-card engine. The 27B always uses the plan engine |
| `KINDI_DENSE_FUSE` | `1` | `0` turns off the fused recurrent and attention kernel launches in the single-card Bonsai engine. The GTX 1080 replay verified exact generated tokens and final logits on 2B, 4B and 9B. |
| `KINDI_CARDS` | (all, at most 2) | the number of cards for the plan engine |
| `KINDI_TP_SPLIT` | `auto` | two cards: when the halves do not leave room for the context on the card with less free memory, `auto` gives the other card a larger part of the FFN and of the output head, so both cards have the same room for the context. That card then does more of the work, so the halves stay whenever they fit. `even` = the halves; a number (0.1 to 0.9) = card 0's part |
| `KINDI_CHUNK` | `256` | prompt tokens per chunk; `0` = one token at a time. The default is `128` on Kepler (sm_30) |
| `KINDI_CHUNK_MIN` | `32` | the smallest part of a prompt that goes through a chunk |
| `KINDI_CHUNK_Q8` | `1` | sm_61 and later: int8 activations in the chunk products (dp4a). `0` = fp32 |
| `KINDI_CHUNK_TILE` | (auto) | `128` = the 128 x 128 tile for the 2-bit chunk products |
| `KINDI_CHUNK_ATT` | `flash` | `exact` = chunk attention with the same arithmetic as one token at a time (slower) |
| `KINDI_G5L` | `1` | sm_61 and later: the 5-trit lane-per-byte GEMV; `0` = one row per thread |
| `KINDI_DENSE_SPACING` | `256` | the spacing of the dense checkpoints in an agent opening; `0` = none |
| `KINDI_STREAM_LOAD` | `1` | the plan engine loads the cached 5-trit weights in 256 MB parts |
| `KINDI_MEM_RESERVE_MB` | `96` (`32` for the plan engine) | GPU memory that the context fit keeps free |
| `KINDI_FD` | `0` (`1` with drafting) | `1` = the split flash-decode attention of the 27B on one card (with `KINDI_TC=1`). When drafting is on (`KINDI_SPEC` set) with `KINDI_TC=1` and `KINDI_FD` is not set, Kindi uses `1`: with `0`, a drafted answer can differ from plain decoding at near-ties |
| `KINDI_TC_DP4A` | `0` | `1` = the Pascal dp4a kernels of the `KINDI_TC=1` path (tcp.cu) on a newer card, for tests. On sm_61 they are always used |
| `KINDI_D1` | `0` (`1` on Pascal) | `1` = the one-pass RMS fold (Hadamard block 512 or 1024, with `KINDI_TC=1`) |
| `KINDI_TC_HP` | `0` | `1` = the `KINDI_TC=1` products take two int8 rows per token (about 14 bits of activation precision instead of 8), at most 4 tokens per pass. The 8-bit products move the logits by up to about 0.15 from the default path, which changes the token at near-ties (#31) |
| `KINDI_ATT_TC` | `0` | `1` = the attention of the prompt chunks on the tensor cores (sm_80 and later, with `KINDI_TC=1`) |
| `KINDI_CKPT_VRAM` | `0` | `1` = checkpoints in card memory; the copy to the host waits until the server is idle (one card) |
| `KINDI_CKPT_PINNED` | `1` under WSL2, else `0` | `1` = the host copies of the checkpoints are in pinned memory. Under Windows (WDDM), this makes the first token faster |
| `KINDI_HOST_RAM_CHECK` | `1` | before the weights of a card load, the plan engine stops when the free host RAM is too small. `0` = no check |
| `KINDI_NB_VOLTA` | `hmma` on sm_70 | AH-Nanbeige4.3-33B on Volta: `hmma` = the prompt products and the prompt attention on the fp16 tensor cores (mma m8n8k4), `dp4a` = the prompt products on dp4a tiles, `0` = the prompt in passes of 8 tokens (before 0.7.0). `hmma` and `dp4a` give the same products, bit for bit, as the int8 tensor cores of sm_80 and later. On sm_61 to sm_75 the variable turns a mode on (not tested there) |
| `KINDI_BLOCKING_SYNC` | `1` before the 580 driver, `0` on 580+ | the host thread sleeps while it waits for the card, and does not use a CPU core. `0` = it waits in a loop (the driver default). On the NVIDIA 580 driver and later the sleeping thread wakes late (5 to 15% of the write speed), so it is off there unless set; the boot log says which |
