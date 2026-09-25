"""
system2.py
----------
System-2 step: a local frontier model (via Ollama) designs a taxonomy from a sample of the
data. It never runs per row. This module only *proposes* a taxonomy; validating, saving and
versioning it is taxonomy_service's job.
"""
from __future__ import annotations

import logging
import random
from typing import Any

import requests
from pydantic import BaseModel

from app.core.errors import System2Error

log = logging.getLogger(__name__)


class TaxonomyResponse(BaseModel):
    """JSON schema handed to Ollama's structured-output `format`: constrains keys and types, not just syntax."""
    taxonomy: dict[str, list[str]]


def _call_ollama_structured(prompt: str, schema: type[BaseModel], settings, seed: int | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0}
    if seed is not None:
        options["seed"] = seed
    payload = {"model": settings.system2_model, "messages": [{"role": "user", "content": prompt}],
               "format": schema.model_json_schema(), "stream": False, "options": options}
    url = f"{settings.ollama_base_url}/api/chat"
    try:
        resp = requests.post(url, json=payload, timeout=settings.system2_timeout_sec)
        resp.raise_for_status()
        return schema.model_validate_json(resp.json()["message"]["content"]).model_dump()
    except requests.exceptions.ConnectionError as e:
        raise System2Error(f"Cannot reach Ollama at {settings.ollama_base_url}.",
                           hint="Start it (`ollama serve`) or set OPENJEV_OLLAMA_BASE_URL. "
                                "You can also create the taxonomy by hand on the Taxonomy page.", detail=str(e))
    except requests.exceptions.Timeout:
        raise System2Error(f"Ollama did not answer within {settings.system2_timeout_sec}s.",
                           hint="Raise OPENJEV_SYSTEM2_TIMEOUT_SEC or use a smaller System-2 model.")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        hint = f"Run `ollama pull {settings.system2_model}`." if code == 404 else "Check the Ollama server log."
        raise System2Error(f"Ollama returned HTTP {code}.", hint=hint,
                           detail=(e.response.text[:500] if e.response is not None else str(e)))
    except (KeyError, ValueError) as e:                          # bad JSON or schema mismatch
        raise System2Error("Ollama returned a response that is not a valid taxonomy.",
                           hint="Try again, or use a stronger System-2 model.", detail=f"{type(e).__name__}: {e}")


def normalise_and_trim(taxonomy: dict[str, list[str]], settings, profile_key: str) -> dict[str, list[str]]:
    """Enforce the size caps and the catch-all rule regardless of what the model returned."""
    others = settings.others_label
    max_main = settings.eff(profile_key, "max_main_categories")
    max_sub = settings.eff(profile_key, "max_sub_categories")

    body: dict[str, list[str]] = {}
    for main, subs in taxonomy.items():
        name = " ".join(str(main).split())
        if name and name.casefold() != others.casefold() and name not in body:
            body[name] = list(dict.fromkeys(" ".join(str(s).split()) for s in subs if str(s).strip()))[:max_sub]
    body = dict(list(body.items())[: max_main - 1])              # leave room for the catch-all
    body[others] = []
    return body


def generate_taxonomy(narrations: list[str], settings, profile_key: str) -> dict[str, list[str]]:
    """Sample the data, ask System 2, return a validated taxonomy (not yet saved)."""
    rng = random.Random(settings.system2_seed)                   # seed=None -> free sampling
    sample = rng.sample(narrations, min(settings.taxonomy_sample_size, len(narrations)))
    prompt = settings.prompt(
        profile_key, "taxonomy_generation_prompt", n=len(sample),
        sample="\n".join(f"- {s[:300]}" for s in sample),
        max_main=settings.eff(profile_key, "max_main_categories"),
        max_sub=settings.eff(profile_key, "max_sub_categories"))
    log.info("System 2 (%s) designing taxonomy for profile '%s' from %d samples",
             settings.system2_model, profile_key, len(sample))
    result = _call_ollama_structured(prompt, TaxonomyResponse, settings, seed=settings.system2_seed)
    return normalise_and_trim(result["taxonomy"], settings, profile_key)


def ollama_status(settings) -> dict[str, Any]:
    """Cheap reachability probe for Diagnostics."""
    try:
        r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
        return {"reachable": True, "model_available": settings.system2_model in names, "models": names}
    except Exception as e:
        return {"reachable": False, "model_available": False, "error": f"{type(e).__name__}: {e}"}
