# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""Kindi servers: find them, read their live state, report the features, and connect agents to one.

Bridge finds Kindi servers in three places: the server that Bridge uses now (the `endpoint` setting), the servers
that the Kindi installer started (`~/.kindi/install.json`, or `$KINDI_HOME/install.json`), and the providers in the
pi configuration of Bridge. It reads `/health` and `/v1/models` from each one. Nothing here changes a server.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request

DEFAULT = 'http://127.0.0.1:8093'
KINDI_VERSION = '0.5.0'

# The features of Kindi 0.5.0, from its README, CHANGELOG and docs. `status` is "release" (in Kindi 0.5.0) or
# "preview" (in development for a later release). `probe` names the /health field that shows the feature live.
FEATURES = [
    ('Models', 'Ternary-Bonsai-2 2B, 4B, 9B and 27B', 'release', None,
     'The main models. Greedy output is token-for-token identical to llama.cpp on the same file (kindi verify).'),
    ('Models', 'AH-Nanbeige4.3-33B IQ3_XXS', 'release', None,
     'A separate 33B model with its own backend. NVIDIA sm_80 and later; one card or two.'),
    ('Models', 'NVIDIA Nemotron 3.5 Bolt E4B', 'release', None, 'Experimental support (Gemma 4 engine).'),
    ('Models', 'Nemotron 3 Nano 30B-A3B (Lightning MoE)', 'preview', None,
     'Mixture-of-experts backend with experts in host memory. Planned as a supported model in 0.6.0.'),
    ('Models', 'DeepSeek V4.1 Flash', 'preview', None,
     'Large MoE model on one GPU, a second expert card and the CPU, with DSpark drafting. Planned for 0.6.0.'),
    ('Cards', 'NVIDIA Kepler to Blackwell', 'release', None,
     'Quadro K420 to the RTX 50 series, Tesla P100, the CMP 170HX (40 GB). `kindi verify` passes on each.'),
    ('Cards', 'AMD Radeon 780M, 840M and 8060S', 'release', None,
     'ROCm images: kindi-rocm (RDNA 3) and kindi-rocm7 (RDNA 3.5).'),
    ('Cards', 'CPU engine', 'release', None, 'x86-64 with AVX2. The installer offers CPU servers next to GPU servers.'),
    ('Cards', 'Two cards in tensor parallel', 'release', None,
     'One model over two cards, for example the 27B on two K1200 or two CMP 170HX cards.'),
    ('Cards', 'Several servers on several cards', 'release', None,
     'The installer can start one server for each card (kindi, kindi-2, ...), each on its own port.'),
    ('Speed', 'Exact and fast output heads', 'release', 'head',
     'exact = the full head (token-exact). fast = a shortlist head, about 15% faster. `kindi head fast|exact` switches.'),
    ('Speed', 'Speculative decoding (draft model)', 'release', 'draft',
     'On RTX 30, 40 and 50 cards the 27B uses the 2B as a draft model: up to 1.82x faster, the same output.'),
    ('Speed', 'MTP head for the 2B', 'release', 'draft', 'A multi-token prediction head on tensor-core cards (docs/MTP.md).'),
    ('Speed', 'DSpark drafting', 'preview', 'dspark', 'The DeepSeek V4.1 Flash drafter.'),
    ('Memory', 'Prompt cache: checkpoints', 'release', 'cached_tokens',
     'Recurrent-state checkpoints every 256 tokens. A request that shares a start resumes at the last checkpoint.'),
    ('Memory', 'Prompt cache: parked conversations', 'release', None,
     'Up to 3 other conversations wait in host RAM, so an agent can switch between its main and side prompts.'),
    ('Memory', 'Prompt cache: saved openings', 'release', None,
     'A shared opening of 512+ tokens (a system prompt with tools) is saved on disk and loads in a fraction of a second.'),
    ('API', 'GET /health', 'release', 'ok', 'Version, commit, model, context, head, live progress, counters, GPU memory.'),
    ('API', 'GET /v1/models', 'release', None, 'The model id and its context length.'),
    ('API', 'POST /v1/chat/completions', 'release', None,
     'OpenAI chat: streaming, tools and tool_calls, thinking (reasoning_content), sampling, stop, seed, usage.'),
]


def base_of(url):
    """http://host:port from any URL of a server (…/health, …/v1, or the base itself)"""
    url = (url or DEFAULT).strip().rstrip('/')
    for tail in ('/health', '/v1'):
        if url.endswith(tail):url = url[:-len(tail)]
    if '://' not in url:url = 'http://'+url
    return url


def _get(url, timeout=2.0, key=None):
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer '+key} if key else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def install_state(home=None):
    """the servers in the Kindi installer's state file, as (label, base URL)"""
    home = Path(home or os.environ.get('KINDI_HOME') or Path.home()/'.kindi')
    try:state = json.loads((home/'install.json').read_text())
    except (OSError, ValueError):return []
    return [(f'{s.get("container","kindi")} ({s.get("model","?")})', f'http://127.0.0.1:{s["port"]}')
            for s in state.get('servers', []) if s.get('port')]


def pi_dir():
    return Path(os.environ.get('PI_CODING_AGENT_DIR') or Path.home()/'.pi'/'agent')


def pi_providers():
    """the OpenAI-compatible providers in Bridge's pi configuration, as (name, base URL, key)"""
    try:data = json.loads((pi_dir()/'models.json').read_text())
    except (OSError, ValueError):return []
    return [(name, base_of(p['baseUrl']), p.get('apiKey')) for name, p in (data.get('providers') or {}).items()
            if p.get('baseUrl') and p.get('api', 'openai-completions') == 'openai-completions']


def servers(store=None):
    """every known server once: (label, base URL, api key, source), the current one first"""
    found, seen = [], set()
    current = base_of(store.setting('endpoint')) if store and store.setting('endpoint') else None
    def add(label, base, key, source):
        if base not in seen:seen.add(base);found.append((label, base, key, source))
    if current:add(store.setting('endpoint_label') or current.split('//')[-1], current, store.setting('endpoint_key'), 'current')
    for label, base in install_state():add(label, base, None, 'installer')
    for name, base, key in pi_providers():add(name, base, key, 'pi')
    if not found:add('default', DEFAULT, None, 'default')
    return found


def family(model):
    m = (model or '').lower()
    for prefix, name in [('bonsai', 'Ternary-Bonsai-2'), ('nanbeige', 'AH-Nanbeige4.3-33B'), ('nemotron', 'Nemotron'),
                         ('deepseek', 'DeepSeek V4.1 Flash'), ('dsv4', 'DeepSeek V4.1 Flash')]:
        if prefix in m:return name
    return 'other'


def probe(base, key=None, timeout=2.0):
    """the live state of one server; `online` is False when it does not answer"""
    base = base_of(base);info = {'base': base, 'online': False}
    try:health = _get(base+'/health', timeout)
    except (OSError, ValueError, urllib.error.URLError) as e:
        info['error'] = str(getattr(e, 'reason', e));return info
    info.update(online=True, health=health)
    info['kindi'] = health.get('kindi')
    info['model'] = health.get('model')
    info['ctx'] = health.get('ctx')
    info['head'] = health.get('head')
    info['draft'] = health.get('draft') or health.get('spec') or health.get('mtp')
    info['dspark'] = health.get('dspark')
    info['busy'] = health.get('busy') or (health.get('state') if health.get('state') not in (None, 'idle') else None)
    info['stats'] = health.get('stats') or {}
    if health.get('gpu_total_mib'):info['gpu'] = (health.get('gpu_total_mib', 0)-health.get('gpu_free_mib', 0), health['gpu_total_mib'])
    info['engine'] = 'Kindi' if (health.get('kindi') or health.get('status') == 'ok') else 'unknown'
    try:
        models = _get(base+'/v1/models', timeout, key).get('data') or []
        info['models'] = [m.get('id') for m in models]
        if models and models[0].get('context_length'):info['ctx'] = info['ctx'] or models[0]['context_length']
    except (OSError, ValueError, urllib.error.URLError) as e:
        info['models_error'] = 'needs an API key' if getattr(e, 'code', None) == 401 else str(getattr(e, 'reason', e))
    info['family'] = family(info.get('model') or (info.get('models') or [''])[0])
    return info


def live(feature, info):
    """what the server shows for a feature, or None"""
    field = feature[3]
    if not info.get('online') or not field:return None
    if field == 'head':return info.get('head')
    if field == 'draft':return 'on' if info.get('draft') else None
    if field == 'dspark':return 'on' if info.get('dspark') else None
    if field == 'cached_tokens':
        n = (info.get('stats') or {}).get('cached_tokens')
        return f'{n:,} tokens reused' if n else None
    if field == 'ok':return 'answers'
    return None


def summary(info):
    """one line for a server"""
    if not info.get('online'):return 'offline · '+info.get('error', 'no answer')
    parts = [info.get('model') or '?', f'ctx {info["ctx"]:,}' if info.get('ctx') else None,
             f'Kindi {info["kindi"]}' if info.get('kindi') else None, f'head {info["head"]}' if info.get('head') else None,
             'drafting' if info.get('draft') or info.get('dspark') else None,
             'busy' if info.get('busy') else 'idle']
    return ' · '.join(p for p in parts if p)


def report(store=None, timeout=2.0):
    """the text of Kindi > Status & features"""
    lines = []
    probes = [(label, source, probe(base, key, timeout)) for label, base, key, source in servers(store)]
    lines.append('SERVERS')
    for label, source, info in probes:
        mark = '>' if source == 'current' else ' '
        lines.append(f'{mark} {label}  {info["base"]}')
        lines.append(f'    {summary(info)}')
        if info.get('gpu'):lines.append(f'    GPU memory {info["gpu"][0]:,} of {info["gpu"][1]:,} MiB')
        stats = info.get('stats') or {}
        if stats:lines.append(f'    {stats.get("requests",0):,} requests · {stats.get("completion_tokens",0):,} tokens written · {stats.get("cached_tokens",0):,} reused from the cache')
    current = next((info for _, source, info in probes if source == 'current'), probes[0][2] if probes else {})
    lines += ['', f'KINDI {KINDI_VERSION} FEATURES  (live values from the current server)']
    area = None
    for feature in FEATURES:
        if feature[0] != area:
            area = feature[0];lines += ['', area.upper()]
        value = live(feature, current)
        tag = '' if feature[2] == 'release' else '  [preview]'
        lines.append(f'  {feature[1]}{tag}'+(f'  — now: {value}' if value else ''))
        lines.append(f'      {feature[4]}')
    lines += ['', 'Kindi > Use a server for agents connects every new agent to the chosen server.',
              'The Kindi manual has the details for each feature: kindi info, kindi verify, kindi bench.']
    return '\n'.join(lines)


def use(store, base, label=None, key=None, ctx=None, model=None):
    """make one server Bridge's server: telemetry, and a pi provider + default model for the agents.

    The pi provider is written only into Bridge's own pi folder (PI_CODING_AGENT_DIR). A pi folder outside Bridge's
    state is left as it is, and the function says what to add instead."""
    base = base_of(base)
    info = probe(base, key)
    if not info.get('online'):raise ValueError(f'{base} does not answer on /health')
    model = model or info.get('model') or (info.get('models') or [None])[0]
    if not model:raise ValueError(f'{base} reports no model')
    ctx = int(ctx or info.get('ctx') or 24576)
    store.setting('endpoint', base+'/health');store.setting('endpoint_label', label or base.split('//')[-1])
    store.setting('endpoint_key', key or '')
    folder = pi_dir()
    from .store import state_dir
    if os.environ.get('PI_CODING_AGENT_DIR') is None or not str(folder.resolve()).startswith(str(state_dir().resolve())):
        return {'base': base, 'model': model, 'ctx': ctx, 'pi': 'unchanged',
                'note': f'Add this provider to {folder/"models.json"}: '+json.dumps(provider(base, model, ctx, key))}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'models.json'
    try:data = json.loads(path.read_text())
    except (OSError, ValueError):data = {}
    name = 'kindi-'+base.split('//')[-1].replace(':', '-').replace('.', '-')
    data.setdefault('providers', {})[name] = provider(base, model, ctx, key)
    tmp = path.with_suffix('.tmp');tmp.write_text(json.dumps(data, indent=2));tmp.replace(path)
    spath = folder/'settings.json'
    try:settings = json.loads(spath.read_text())
    except (OSError, ValueError):settings = {}
    settings.update(defaultProvider=name, defaultModel=model)
    # A first agent prompt can take minutes to read on a small card: a long timeout, and no retry that would queue
    # behind the request that is still running (the Kindi manual, "Meet pi").
    settings.setdefault('retry', {'provider': {'timeoutMs': 1800000, 'maxRetries': 1}})
    tmp = spath.with_suffix('.tmp');tmp.write_text(json.dumps(settings, indent=2));tmp.replace(spath)
    store.setting('kindi_provider', name);store.setting('kindi_model', model)
    return {'base': base, 'model': model, 'ctx': ctx, 'pi': name}


def provider(base, model, ctx, key=None):
    """a pi provider for one Kindi server (the form that the Kindi manual recommends: maxTokens 8192)"""
    return {'baseUrl': base+'/v1', 'api': 'openai-completions', 'apiKey': key or 'none',
            'compat': {'supportsDeveloperRole': False, 'supportsReasoningEffort': False,
                       'thinkingFormat': 'qwen-chat-template', 'thinkingTokenBudgetField': 'thinking_budget_tokens'},
            'models': [{'id': model, 'name': model, 'reasoning': True, 'input': ['text'],
                        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
                        'contextWindow': ctx, 'maxTokens': 8192}]}
