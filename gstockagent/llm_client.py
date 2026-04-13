import json
import hashlib
import os
from pathlib import Path
import requests
from .config import PRED_CACHE_DIR, OPENPATHS_API_KEY, OPENPATHS_BASE_URL

MODEL_MAP = {
    "gemini-flash": "gemini-flash",
    "gemini-3.1": "gemini-3.1-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-lite": "gemini-flash-lite",
    "glm-5": "glm-5",
    "glm-5.1": "zai-org/GLM-5.1",
    "glm-4-plus": "glm-4-plus",
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
GEMINI_MODELS = {"gemini-flash", "gemini-3.1", "gemini-3.1-pro", "gemini-3.1-lite"}
TOGETHER_MODELS = {"glm-5.1"}
ENABLE_THINKING = os.environ.get("GSTOCK_THINKING", "").lower() in ("1", "true", "yes")
USE_DIRECT_GEMINI = os.environ.get("GSTOCK_DIRECT_GEMINI", "").lower() in ("1", "true", "yes")


def _cache_path(model: str, prompt_hash: str, date_str: str) -> Path:
    d = PRED_CACHE_DIR / model.replace("/", "_") / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_hash}.json"


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _call_openpaths(model: str, prompt: str, temperature: float = 0.2) -> str:
    api_key = OPENPATHS_API_KEY
    if not api_key:
        raise RuntimeError("OPENPATHS_API_KEY env var required")
    url = f"{OPENPATHS_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=180)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError(f"Empty response from {model}")
    return content


def _call_together(model_id: str, prompt: str, temperature: float = 0.2,
                   thinking: bool = False) -> str:
    api_key = TOGETHER_API_KEY
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY env var required")
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096,
        "reasoning": {"enabled": thinking},
    }
    r = requests.post(url, json=payload, headers=headers,
                      timeout=300 if thinking else 120)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError(f"Empty response from {model_id}")
    return content


def _call_gemini_direct(model: str, prompt: str, temperature: float = 0.2) -> str:
    api_key = GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var required")
    model_id = MODEL_MAP.get(model, model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_llm(prompt: str, model: str = "gemini-flash", temperature: float = 0.2,
             date_str: str = "", use_cache: bool = True) -> str:
    ph = _hash_prompt(prompt)
    cp = _cache_path(model, ph, date_str)
    if use_cache and cp.exists():
        return cp.read_text()

    resolved = MODEL_MAP.get(model, model)
    if model in TOGETHER_MODELS and TOGETHER_API_KEY:
        resp = _call_together(resolved, prompt, temperature, thinking=ENABLE_THINKING)
    elif USE_DIRECT_GEMINI and model in GEMINI_MODELS:
        resp = _call_gemini_direct(model, prompt, temperature)
    else:
        resp = _call_openpaths(resolved, prompt, temperature)
    cp.write_text(resp)
    return resp


def parse_allocation(response: str) -> dict:
    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {}
        else:
            return {}
    if "allocations" in data:
        data = data["allocations"]
    return data
