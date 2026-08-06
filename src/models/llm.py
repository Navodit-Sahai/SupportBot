from __future__ import annotations
import time
from functools import lru_cache
from typing import List, Dict

from config import MODEL, GENERATION
from src.models.device import resolve_device
from src.utils.logger import log_event


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = resolve_device()
    dtype = MODEL.llm_dtype
    if dtype == "auto":
        torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL.llm_model, revision=MODEL.llm_revision, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL.llm_model,
        revision=MODEL.llm_revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    log_event("llm", "loaded",
              model=MODEL.llm_model, device=device, dtype=str(torch_dtype),
              load_ms=round((time.perf_counter() - t0) * 1000, 1))
    return tokenizer, model, device


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    tokenizer, _, _ = _load()
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"


def classify_choice(messages: List[Dict[str, str]], choices: List[str]) -> str:
    """Pick the most likely choice via first-token log-prob."""
    import torch
    tokenizer, model, device = _load()

    prompt = _build_prompt(messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]        # (vocab,)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    scores: Dict[str, float] = {}
    for c in choices:
        best = float("-inf")
        for variant in (c, " " + c):
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if ids:
                best = max(best, float(log_probs[ids[0]].item()))
        scores[c] = best

    log_event("llm", "classify_choice",
              scores={k: round(v, 3) for k, v in scores.items()})
    return max(scores, key=scores.get)


def generate(
    messages: List[Dict[str, str]],
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Run a chat-formatted generation. `messages` follows OpenAI-style chat schema."""
    import torch
    tokenizer, model, device = _load()

    max_new_tokens = GENERATION.max_new_tokens if max_new_tokens is None else max_new_tokens
    temperature = GENERATION.temperature if temperature is None else temperature
    top_p = GENERATION.top_p if top_p is None else top_p

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        prompt += "\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = temperature > 0
    t0 = time.perf_counter()
    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    log_event("llm", "generated",
              latency_ms=round((time.perf_counter() - t0) * 1000, 1),
              output_tokens=int(generated.shape[0]))
    return text