# Using Kindi with agents and chat clients

Kindi is an OpenAI-compatible endpoint. Use these values:

- Base URL: `http://<host>:8093/v1` (on the same computer: `http://127.0.0.1:8093/v1`).
- Model: `bonsai2-2b` (or your `model_name`).
- API key: your `api_key`. If you did not set an `api_key`, any string works.

For tool calls, Kindi uses the standard `tools` / `tool_calls` fields.

## What to set in any client

1. **Context window: 24,576 tokens** (or your `ctx`). Clients that summarise old turns ("compaction") should do it
   before that.
2. **A long request timeout - 30 to 60 minutes.** Kindi reads new prompt text at about 25-33 tokens/s. The first request
   from an agent that writes code often has 5,000+ tokens of system prompt and tool definitions. The first read of
   that text takes about 3 minutes. The cache then saves it (see the README), so later turns and later sessions are
   fast. Some clients stop after 60-120 s and send the request again. This makes the problem worse: the retry waits
   behind the request that Kindi still reads.
3. **One retry at most**, and no parallel requests: Kindi serves one request at a time, unless the server runs with
   `--slots N` ([CONFIGURATION.md](CONFIGURATION.md)).
4. **A short system prompt and few tools.** Every tool schema is prompt text. Keep the tool list fixed between calls.
   A different set of tools per call changes the start of the prompt, and then the cache cannot help.
5. **Temperature 0.1-0.7.** Greedy (`0`) is reproducible; the model publisher recommends 0.7 / top_p 0.8 / top_k 20.

Watch a long first read with `curl -s localhost:8093/health` - the `busy` field shows `prompt_read` progress.

## Examples

**Any OpenAI SDK**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8093/v1", api_key="none", timeout=3600, max_retries=1)
```

**Agents for code that have an OpenAI-compatible provider setting** (for example pi, or similar terminal agents):

1. Add a provider with these values (for pi, the exact `models.json` entry is in
   [DOCKER.md](DOCKER.md#pi-and-other-agents-beside-the-container); do not use pi's built-in OpenAI provider, which
   talks to OpenAI's servers):
   - Base URL: `http://127.0.0.1:8093/v1`.
   - API style: "openai-completions" / "chat completions".
   - Model: `bonsai2-2b`.
   - Context window: `24576`.
   - Max output: `2048`.
2. Increase the request timeout of the agent, as above.

A shorter custom system prompt helps the first turn.

**Chat UIs** (Open WebUI, LibreChat and similar): add an OpenAI connection with the base URL above.

## What to expect from a 2B model

For its size, the model does short shell and Python tasks, file edits, factual questions, summaries and tool calls
well. It also makes the usual mistakes of a small model:

- Wrong tool arguments. Agents usually recover when they send the error back to the model.
- Repetition in long answers.
- Weak multi-step reasoning.

Give it small, concrete steps.
