"""Multi-provider LLM wrapper for trading decisions.

Supports: Gemini, OpenAI, Anthropic
All return TradePlan via structured output / JSON mode.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore", message=".*is not a valid ThinkingLevel.*")

from llm_hourly_trader.cache import get_cached, set_cached
from llm_hourly_trader.gemini_wrapper import TradePlan

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str = "gemini-2.5-flash", max_retries: int = 5,
                thinking_level: str | None = None) -> TradePlan:
    cached = get_cached(model, prompt)
    if cached is not None:
        return TradePlan(**cached)

    from google import genai
    from google.genai import types

    SCHEMA = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["direction", "buy_price", "sell_price", "confidence", "reasoning"],
        properties={
            "direction": genai.types.Schema(type=genai.types.Type.STRING),
            "buy_price": genai.types.Schema(type=genai.types.Type.STRING),
            "sell_price": genai.types.Schema(type=genai.types.Type.STRING),
            "confidence": genai.types.Schema(type=genai.types.Type.STRING),
            "reasoning": genai.types.Schema(type=genai.types.Type.STRING),
        },
    )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Build config with optional thinking
    config_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": SCHEMA,
    }
    if thinking_level:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
            )
        except Exception:
            pass  # older SDK versions may not support thinking
    else:
        config_kwargs["temperature"] = 0.3
    config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config,
            )
            data = json.loads(response.text)
            plan = TradePlan(
                direction=data.get("direction", "hold").lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=float(data.get("confidence", 0) or 0),
                reasoning=data.get("reasoning", ""),
            )
            set_cached(model, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# OpenAI (GPT-4.1, GPT-5.x, o3, o4-mini, etc.)
# ---------------------------------------------------------------------------

def call_openai(prompt: str, model: str = "gpt-4.1-mini", max_retries: int = 5) -> TradePlan:
    cached = get_cached(model, prompt)
    if cached is not None:
        return TradePlan(**cached)

    from openai import OpenAI

    # Try codex auth first, fall back to OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY", "")
    codex_path = os.path.expanduser("~/.codex/auth.json")
    if os.path.exists(codex_path):
        try:
            auth = json.load(open(codex_path))
            api_key = auth["tokens"]["access_token"]
        except Exception:
            pass

    client = OpenAI(api_key=api_key)

    json_schema = {
        "name": "trade_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["direction", "buy_price", "sell_price", "confidence", "reasoning"],
            "additionalProperties": False,
            "properties": {
                "direction": {"type": "string", "description": "long, short, or hold"},
                "buy_price": {"type": "number", "description": "Limit entry price, 0 if no entry"},
                "sell_price": {"type": "number", "description": "Take-profit price, 0 if no exit"},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
        },
    }

    # Determine if this is a reasoning model (o3, o4, gpt-5.x, etc.)
    is_reasoning = model.startswith("o3") or model.startswith("o4") or model.startswith("gpt-5")

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": json_schema},
            }
            if is_reasoning:
                kwargs["max_completion_tokens"] = 4096
                kwargs["reasoning_effort"] = "high"
            else:
                kwargs["max_tokens"] = 1024
                kwargs["temperature"] = 0.3

            resp = client.chat.completions.create(**kwargs)
            data = json.loads(resp.choices[0].message.content)
            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=float(data.get("confidence", 0) or 0),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(model, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# Anthropic (Claude Sonnet 4.6, etc.)
# ---------------------------------------------------------------------------

def call_anthropic(prompt: str, model: str = "claude-sonnet-4-6", max_retries: int = 5) -> TradePlan:
    cached = get_cached(model, prompt)
    if cached is not None:
        return TradePlan(**cached)

    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY

    tool_schema = {
        "name": "submit_trade_plan",
        "description": "Submit your trading decision",
        "input_schema": {
            "type": "object",
            "required": ["direction", "buy_price", "sell_price", "confidence", "reasoning"],
            "properties": {
                "direction": {"type": "string", "description": "long, short, or hold"},
                "buy_price": {"type": "number", "description": "Limit entry price, 0 if no entry"},
                "sell_price": {"type": "number", "description": "Take-profit price, 0 if no exit"},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
        },
    }

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": "submit_trade_plan"},
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract tool use block
            for block in resp.content:
                if block.type == "tool_use" and block.name == "submit_trade_plan":
                    data = block.input
                    plan = TradePlan(
                        direction=str(data.get("direction", "hold")).lower().strip(),
                        buy_price=float(data.get("buy_price", 0) or 0),
                        sell_price=float(data.get("sell_price", 0) or 0),
                        confidence=float(data.get("confidence", 0) or 0),
                        reasoning=str(data.get("reasoning", "")),
                    )
                    set_cached(model, prompt, plan.__dict__)
                    return plan
            return TradePlan("hold", 0, 0, 0, "No tool use in response")
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# DeepSeek (v3.2, reasoner, etc.) - uses OpenAI-compatible API
# ---------------------------------------------------------------------------

def call_deepseek(prompt: str, model: str = "deepseek-chat", max_retries: int = 5) -> TradePlan:
    cached = get_cached(model, prompt)
    if cached is not None:
        return TradePlan(**cached)

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    is_reasoning = "reasoner" in model

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a trading assistant. Always respond with valid JSON containing: direction (long/short/hold), buy_price (number), sell_price (number), confidence (0-1), reasoning (string)."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            if is_reasoning:
                kwargs["max_tokens"] = 4096
            else:
                kwargs["max_tokens"] = 1024
                kwargs["temperature"] = 0.3

            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            data = json.loads(text)
            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=float(data.get("confidence", 0) or 0),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(model, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# Codex CLI (GPT-5.x via ChatGPT Pro plan)
# ---------------------------------------------------------------------------

# JSON schema file for codex --output-schema
_CODEX_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "codex_trade_schema.json")

def _ensure_codex_schema() -> str:
    """Write the JSON schema file once, return path."""
    if not os.path.exists(_CODEX_SCHEMA_PATH):
        schema = {
            "type": "object",
            "required": ["direction", "buy_price", "sell_price", "confidence", "reasoning"],
            "additionalProperties": False,
            "properties": {
                "direction": {"type": "string", "description": "long, short, or hold"},
                "buy_price": {"type": "number", "description": "Limit entry price, 0 if no entry"},
                "sell_price": {"type": "number", "description": "Take-profit price, 0 if no exit"},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
        }
        with open(_CODEX_SCHEMA_PATH, "w") as f:
            json.dump(schema, f)
    return _CODEX_SCHEMA_PATH


def call_openai_responses(prompt: str, model: str = "gpt-5.4", max_retries: int = 5,
                          reasoning_effort: str = "low") -> TradePlan:
    """Call GPT-5.x via OpenAI Responses API with structured outputs."""
    cached = get_cached(model, prompt)
    if cached is not None:
        return TradePlan(**cached)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    json_schema = {
        "type": "json_schema",
        "name": "trade_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["direction", "buy_price", "sell_price", "confidence", "reasoning"],
            "additionalProperties": False,
            "properties": {
                "direction": {"type": "string", "description": "long, short, or hold"},
                "buy_price": {"type": "number", "description": "Limit entry price, 0 if no entry"},
                "sell_price": {"type": "number", "description": "Take-profit price, 0 if no exit"},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
        },
    }

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                text={"format": json_schema},
                reasoning={"effort": reasoning_effort, "summary": "auto"},
                store=False,
            )
            text = resp.output_text
            data = json.loads(text)
            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=float(data.get("confidence", 0) or 0),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(model, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


def call_codex(prompt: str, model: str = "gpt-5.4", max_retries: int = 2,
               reasoning_effort: str = "low") -> TradePlan:
    """Call GPT-5.x via codex CLI (routes through ChatGPT Pro plan)."""
    cached = get_cached(model, prompt)
    if cached is not None:
        return TradePlan(**cached)

    schema_path = _ensure_codex_schema()

    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                tmp_path = tmp.name

            # codex exec with structured output schema + reasoning effort control
            result = subprocess.run(
                [
                    "codex", "exec",
                    "--model", model,
                    "--ephemeral",
                    "--output-schema", schema_path,
                    "-o", tmp_path,
                    "--dangerously-bypass-approvals-and-sandbox",
                    "-c", f"model_reasoning_effort={reasoning_effort}",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Read output file
            with open(tmp_path) as f:
                text = f.read().strip()
            os.unlink(tmp_path)

            if not text:
                raise ValueError(f"Empty response from codex (rc={result.returncode}, stderr={result.stderr[:200]})")

            # Parse JSON - codex may wrap in markdown code blocks
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\n?", "", text)
                text = re.sub(r"\n?```$", "", text)

            data = json.loads(text)
            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=float(data.get("confidence", 0) or 0),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(model, prompt, plan.__dict__)
            return plan
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                time.sleep(wait)
                continue
            print(f"  codex error after {max_retries} attempts: {e}")
    return TradePlan("hold", 0, 0, 0, "codex API exhausted")


# ---------------------------------------------------------------------------
# Shared retry logic
# ---------------------------------------------------------------------------

def _handle_retry(e: Exception, attempt: int, max_retries: int) -> None:
    err_str = str(e)
    if "429" in err_str or "rate" in err_str.lower():
        delay_match = re.search(r"retry.?(?:in|after).?(\d+\.?\d*)", err_str, re.IGNORECASE)
        wait = float(delay_match.group(1)) + 1 if delay_match else 15 * (attempt + 1)
        if attempt < max_retries - 1:
            time.sleep(min(wait, 120))
            return
    elif attempt < max_retries - 1:
        time.sleep(2 * (attempt + 1))
        return


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PROVIDER_FNS = {
    "gemini": call_gemini,
    "openai": call_openai,
    "openai_responses": call_openai_responses,
    "anthropic": call_anthropic,
    "deepseek": call_deepseek,
    "codex": call_codex,
}

# Model -> provider mapping for convenience
MODEL_PROVIDERS = {
    # Gemini
    "gemini-2.5-flash": "gemini",
    "gemini-3.1-flash-lite-preview": "gemini",
    "gemini-2.0-flash": "gemini",
    # OpenAI (direct API - needs billing credits)
    "gpt-4.1-mini": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-nano": "openai",
    "gpt-4o": "openai",
    "o3": "openai",
    "o3-mini": "openai",
    "o4-mini": "openai",
    # OpenAI via Responses API
    "gpt-5.4": "openai_responses",
    # Anthropic
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
    # DeepSeek
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
}


def call_llm(prompt: str, model: str, provider: Optional[str] = None,
             thinking_level: Optional[str] = None,
             reasoning_effort: Optional[str] = None) -> TradePlan:
    """Call any LLM provider with auto-detection."""
    if provider is None:
        provider = MODEL_PROVIDERS.get(model)
        if provider is None:
            if "gemini" in model:
                provider = "gemini"
            elif "claude" in model:
                provider = "anthropic"
            elif "deepseek" in model:
                provider = "deepseek"
            else:
                provider = "openai"
    fn = PROVIDER_FNS[provider]
    if provider == "gemini" and thinking_level:
        return fn(prompt, model=model, thinking_level=thinking_level)
    if provider == "openai_responses":
        return fn(prompt, model=model, reasoning_effort=reasoning_effort or "low")
    return fn(prompt, model=model)
