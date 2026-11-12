"""Native Kindi rendering of a Gemma 4 GGUF chat template."""
import json
from jinja2 import Environment, ChainableUndefined
from .gguf import GGUF

def _raise(message):
    raise ValueError(str(message))

_compiled={}

def _template(model_path):
    """the compiled chat template and the BOS/EOS strings of a GGUF, read once per file"""
    if model_path not in _compiled:
        kv=GGUF(model_path).kv
        template=kv.get("tokenizer.chat_template")
        if not template: raise ValueError("kindi: Gemma 4 GGUF has no chat template")
        token_list=kv.get("tokenizer.ggml.tokens",[])
        bos=int(kv.get("tokenizer.ggml.bos_token_id",2))
        eos=int(kv.get("tokenizer.ggml.eos_token_id",1))
        env=Environment(undefined=ChainableUndefined,   # a missing key is false and empty, as in llama.cpp
                        autoescape=False,trim_blocks=True,lstrip_blocks=True)
        env.filters["tojson"]=lambda value,**kw: json.dumps(value,ensure_ascii=False,separators=(",",":"))
        env.filters["items"]=lambda v: list(v.items()) if hasattr(v,"items") else []   # Jinja2 3.1's |items (image: 3.0)
        env.globals["raise_exception"]=_raise
        _compiled[model_path]=(env.from_string(template),token_list[bos] if 0<=bos<len(token_list) else "",
                               token_list[eos] if 0<=eos<len(token_list) else "")
    return _compiled[model_path]

def render(model_path,messages,tools=None,template_kwargs=None,add_generation_prompt=True):
    tpl,bos,eos=_template(model_path)
    values={"messages":messages,"tools":tools or [],"add_generation_prompt":bool(add_generation_prompt),
            "bos_token":bos,"eos_token":eos,"enable_thinking":False,"conversational_mode":False}
    values.update(template_kwargs or {})
    try: return tpl.render(**values)
    except Exception as e: raise ValueError("kindi Gemma 4 chat template failed: %s" % e)
