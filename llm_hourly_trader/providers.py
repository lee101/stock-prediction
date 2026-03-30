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
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*is not a valid ThinkingLevel.*")

from llm_hourly_trader.cache import get_cached, set_cached
from llm_hourly_trader.gemini_wrapper import TradePlan


class CacheMissError(RuntimeError):
    """Raised when cache-only replay is requested but a prompt has no cached response."""


def _normalize_confidence(val) -> float:
    """Normalize confidence to 0.0-1.0 range.

    Handles models that return 0-100 instead of 0-1, or string values.
    """
    try:
        c = float(val or 0)
    except (ValueError, TypeError):
        return 0.5
    if c > 1.0:
        c = c / 100.0  # 85 -> 0.85, 3.0 -> 0.03
    return max(0.0, min(1.0, c))


def _gemini_timeout_ms() -> int | None:
    raw = os.environ.get("GEMINI_HTTP_TIMEOUT_MS", "120000").strip()
    if not raw:
        return None
    try:
        timeout_ms = int(raw)
    except ValueError:
        return 120000
    return timeout_ms if timeout_ms > 0 else None

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str = "gemini-2.5-flash", max_retries: int = 3,
                thinking_level: str | None = None,
                cache_model: str | None = None,
                provider_call_models: list[str] | None = None) -> TradePlan:
    cache_key = cache_model or model
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

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

    timeout_ms = _gemini_timeout_ms()
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=timeout_ms) if timeout_ms else None,
    )

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
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=data.get("reasoning", ""),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# OpenAI (GPT-4.1, GPT-5.x, o3, o4-mini, etc.)
# ---------------------------------------------------------------------------

def call_openai(prompt: str, model: str = "gpt-4.1-mini", max_retries: int = 5,
                cache_model: str | None = None,
                provider_call_models: list[str] | None = None) -> TradePlan:
    cache_key = cache_model or model
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

    from openai import OpenAI

    # Try codex auth first, fall back to OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY", "")
    codex_path = Path("~/.codex/auth.json").expanduser()
    if codex_path.exists():
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
                "confidence": {"type": "number", "description": "Confidence as decimal 0.0-1.0 (e.g. 0.75 not 75)"},
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
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# Anthropic (Claude Sonnet 4.6, etc.)
# ---------------------------------------------------------------------------

def call_anthropic(prompt: str, model: str = "claude-sonnet-4-6", max_retries: int = 5,
                   thinking: bool = False, effort: str | None = None,
                   cache_model: str | None = None,
                   provider_call_models: list[str] | None = None) -> TradePlan:
    """Call Anthropic Claude API.

    Args:
        thinking: Enable extended thinking (adaptive mode).
        effort: Output effort level — "low" for cheap backtesting, "high"/"max" for production.
    """
    cache_key = cache_model or (model + (f"_t={thinking}_e={effort}" if (thinking or effort) else ""))
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Try loading from env_real.py (project-local secrets file)
        try:
            import importlib.util, sys as _sys
            _root = Path(__file__).resolve().parent.parent
            _spec = importlib.util.spec_from_file_location("env_real", str(_root / "env_real.py"))
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass

    client = anthropic.Anthropic(api_key=api_key or None)  # uses ANTHROPIC_API_KEY

    tool_schema = {
        "name": "submit_trade_plan",
        "description": "Submit your trading decision. confidence MUST be a decimal between 0.0 and 1.0 (NOT 0-100).",
        "input_schema": {
            "type": "object",
            "required": ["direction", "buy_price", "sell_price", "confidence"],
            "properties": {
                "direction": {"type": "string", "description": "long, short, or hold"},
                "buy_price": {"type": "number", "description": "Limit entry price in USD (must match asset price scale), 0 if no entry"},
                "sell_price": {"type": "number", "description": "Take-profit price in USD (must match asset price scale), 0 if no exit"},
                "confidence": {"type": "number", "description": "Confidence as decimal 0.0 to 1.0 (e.g. 0.75 for 75% confident). NEVER use 0-100 scale."},
            },
        },
    }

    for attempt in range(max_retries):
        try:
            if thinking:
                # Thinking mode: can't force tool_use, ask for JSON text instead
                json_instruction = (
                    "\n\nRespond with ONLY a JSON object (no markdown fences): "
                    '{"direction": "long"|"short"|"hold", "buy_price": <number>, '
                    '"sell_price": <number>, "confidence": <0.0-1.0>, "reasoning": "<brief>"}'
                )
                kwargs = {
                    "model": model,
                    "max_tokens": 16000,
                    "thinking": {"type": "enabled", "budget_tokens": 10000},
                    "messages": [{"role": "user", "content": prompt + json_instruction}],
                }
                resp = client.messages.create(**kwargs)
                text = ""
                for block in resp.content:
                    if block.type == "text":
                        text = block.text
                        break
                # Strip markdown fences if present
                text = re.sub(r"^```(?:json)?\n?", "", text.strip())
                text = re.sub(r"\n?```$", "", text)
                data = json.loads(text)
            else:
                kwargs = {
                    "model": model,
                    "max_tokens": 1024,
                    "tools": [tool_schema],
                    "tool_choice": {"type": "tool", "name": "submit_trade_plan"},
                    "messages": [{"role": "user", "content": prompt}],
                }
                resp = client.messages.create(**kwargs)
                data = None
                for block in resp.content:
                    if block.type == "tool_use" and block.name == "submit_trade_plan":
                        data = block.input
                        break
                if data is None:
                    return TradePlan("hold", 0, 0, 0, "No tool use in response")

            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# DeepSeek (v3.2, reasoner, etc.) - uses OpenAI-compatible API
# ---------------------------------------------------------------------------

def call_deepseek(prompt: str, model: str = "deepseek-chat", max_retries: int = 5,
                  cache_model: str | None = None,
                  provider_call_models: list[str] | None = None) -> TradePlan:
    cache_key = cache_model or model
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

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
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# xAI Grok (grok-4-1-fast, grok-4.20-multi-agent-0309, etc.)
# ---------------------------------------------------------------------------

def call_xai(prompt: str, model: str = "grok-4-1-fast", max_retries: int = 5,
             thinking: bool = False,
             cache_model: str | None = None,
             provider_call_models: list[str] | None = None) -> TradePlan:
    """Call xAI Grok API with structured outputs via xai_sdk.

    Uses Pydantic-based structured output for guaranteed schema compliance.
    Supports reasoning models (grok-4.20-*) and fast models (grok-4-1-fast).
    """
    cache_key = cache_model or (model + ("_think" if thinking else ""))
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

    from pydantic import BaseModel, Field

    class GrokTradePlan(BaseModel):
        direction: str = Field(description="One of: long, short, hold")
        buy_price: float = Field(description="Limit entry price, 0 if no entry")
        sell_price: float = Field(description="Take-profit/exit price, 0 if no exit")
        confidence: float = Field(description="Confidence 0.0-1.0 (NOT 0-100)")
        reasoning: str = Field(description="Brief explanation of the decision")
        allocation_pct: float = Field(default=0.0, description="Capital allocation 0-100%")

    for attempt in range(max_retries):
        try:
            from xai_sdk import Client
            from xai_sdk.chat import system, user

            api_key = os.environ.get("XAI_API_KEY", "")
            if not api_key:
                return TradePlan("hold", 0, 0, 0, "XAI_API_KEY not set")

            client = Client(api_key=api_key)
            chat = client.chat.create(model=model)
            chat.append(system(
                "You are an expert trader. Analyze the market data and produce a structured "
                "trading decision. confidence MUST be a decimal between 0.0 and 1.0 (NOT 0-100). "
                "buy_price and sell_price must match the asset's price scale."
            ))
            chat.append(user(prompt))

            response, parsed = chat.parse(GrokTradePlan)

            plan = TradePlan(
                direction=parsed.direction.lower().strip(),
                buy_price=float(parsed.buy_price),
                sell_price=float(parsed.sell_price),
                confidence=_normalize_confidence(parsed.confidence),
                reasoning=parsed.reasoning,
                allocation_pct=float(parsed.allocation_pct),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "xAI API exhausted")


# ---------------------------------------------------------------------------
# xAI Grok with tools (web_search, x_search for market intel)
# ---------------------------------------------------------------------------

def call_xai_with_tools(prompt: str, model: str = "grok-4.20-multi-agent-0309",
                        max_retries: int = 3,
                        use_web_search: bool = True,
                        use_x_search: bool = True,
                        cache_model: str | None = None,
                        provider_call_models: list[str] | None = None) -> TradePlan:
    """Call Grok with web/X search tools for real-time market intelligence.

    The model can search the web and X (Twitter) for recent news, sentiment,
    and market-moving events before making a trading decision.
    """
    cache_key = cache_model or f"{model}_tools"
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

    from pydantic import BaseModel, Field

    class GrokTradePlanWithContext(BaseModel):
        direction: str = Field(description="One of: long, short, hold")
        buy_price: float = Field(description="Limit entry price, 0 if no entry")
        sell_price: float = Field(description="Take-profit/exit price, 0 if no exit")
        confidence: float = Field(description="Confidence 0.0-1.0")
        reasoning: str = Field(description="Brief explanation including any web/X insights")
        allocation_pct: float = Field(default=0.0, description="Capital allocation 0-100%")

    for attempt in range(max_retries):
        try:
            from xai_sdk import Client
            from xai_sdk.chat import system, user
            from xai_sdk.tools import web_search, x_search

            api_key = os.environ.get("XAI_API_KEY", "")
            if not api_key:
                return TradePlan("hold", 0, 0, 0, "XAI_API_KEY not set")

            client = Client(api_key=api_key)

            tools = []
            if use_web_search:
                tools.append(web_search())
            if use_x_search:
                tools.append(x_search())

            chat = client.chat.create(model=model, tools=tools)
            chat.append(system(
                "You are an expert trader with access to real-time web and X (Twitter) search. "
                "BEFORE making your trading decision, search for recent news and sentiment about "
                "the assets mentioned. Look for: earnings, macro events, whale movements, "
                "regulatory news, and social sentiment. "
                "Then produce a structured trading decision. "
                "Be AGGRESSIVE — optimize for maximum PnL. Enter trades boldly when you see edge. "
                "confidence MUST be a decimal between 0.0 and 1.0."
            ))
            chat.append(user(prompt))

            response, parsed = chat.parse(GrokTradePlanWithContext)

            plan = TradePlan(
                direction=parsed.direction.lower().strip(),
                buy_price=float(parsed.buy_price),
                sell_price=float(parsed.sell_price),
                confidence=_normalize_confidence(parsed.confidence),
                reasoning=parsed.reasoning,
                allocation_pct=float(parsed.allocation_pct),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "xAI tools API exhausted")


# ---------------------------------------------------------------------------
# Two-step pipeline: Grok X-search → Gemini decision
# ---------------------------------------------------------------------------

def call_grok_context_gemini(prompt: str, model: str = "gemini-3.1-flash-lite-preview",
                             context_model: str = "grok-4.20-multi-agent-0309",
                             max_retries: int = 3,
                             thinking_level: str | None = "HIGH",
                             cache_model: str | None = None,
                             provider_call_models: list[str] | None = None) -> TradePlan:
    """Two-step pipeline: Grok searches web/X for market context, Gemini trades.

    Step 1: Grok 4.2 with web_search + x_search gathers real-time market intel
    Step 2: That context is prepended to the trading prompt, then Gemini decides
    """
    cache_key = cache_model or f"grok_ctx_gemini_{model}"
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)

    # Step 1: Get market context from Grok with tools
    context = ""
    try:
        from xai_sdk import Client
        from xai_sdk.chat import system, user
        from xai_sdk.tools import web_search, x_search

        api_key = os.environ.get("XAI_API_KEY", "")
        if api_key:
            # Extract symbol from prompt for targeted search
            import re as _re
            symbol_match = _re.search(r'SYMBOL:\s*(\S+)', prompt)
            symbol = symbol_match.group(1) if symbol_match else "the asset"

            client = Client(api_key=api_key)
            chat = client.chat.create(model=context_model, tools=[web_search(), x_search()])
            chat.append(system(
                "You are a financial research assistant. Search for the most recent and "
                "relevant market information. Focus on: breaking news, earnings, analyst "
                "upgrades/downgrades, macro events, whale movements, and social sentiment. "
                "Return a brief summary (3-5 bullet points) of what you found."
            ))
            chat.append(user(
                f"Search for the latest news and X/Twitter sentiment about {symbol}. "
                f"What are the key market-moving factors right now?"
            ))
            response = chat.send()
            context = response.content or ""
            if provider_call_models is not None:
                provider_call_models.append(context_model)
    except Exception as e:
        context = f"(context search failed: {e})"

    # Step 2: Prepend context to prompt, call Gemini
    if context:
        enhanced_prompt = (
            f"## REAL-TIME MARKET INTELLIGENCE (from web/X search):\n"
            f"{context}\n\n"
            f"Use the above intelligence to inform your trading decision.\n\n"
            f"{prompt}"
        )
    else:
        enhanced_prompt = prompt

    return call_gemini(
        enhanced_prompt,
        model=model,
        thinking_level=thinking_level,
        cache_model=cache_key,
        provider_call_models=provider_call_models,
    )


# ---------------------------------------------------------------------------
# xAI Grok via OpenAI-compatible API (fallback / simpler path)
# ---------------------------------------------------------------------------

def call_xai_openai(prompt: str, model: str = "grok-4-1-fast", max_retries: int = 5,
                    cache_model: str | None = None,
                    provider_call_models: list[str] | None = None) -> TradePlan:
    """Call xAI Grok via OpenAI-compatible API at api.x.ai/v1."""
    cache_key = cache_model or model
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

    from openai import OpenAI

    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        return TradePlan("hold", 0, 0, 0, "XAI_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

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
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
        },
    }

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert trader. Produce a structured trading decision. confidence MUST be a decimal 0.0-1.0."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": json_schema},
                max_tokens=2048,
                temperature=0.3,
            )
            data = json.loads(resp.choices[0].message.content)
            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "xAI API exhausted")


# ---------------------------------------------------------------------------
# Zhipu GLM (glm-5.1, etc.) — OpenAI-compatible API
# ---------------------------------------------------------------------------

def call_zhipu(prompt: str, model: str = "glm-5.1", max_retries: int = 5,
               cache_model: str | None = None,
               provider_call_models: list[str] | None = None) -> TradePlan:
    """Call Zhipu GLM via OpenAI-compatible API at api.z.ai."""
    cache_key = cache_model or model
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        return TradePlan("hold", 0, 0, 0, "ZHIPU_API_KEY not set")

    from openai import OpenAI

    # Try api.z.ai first, fallback to open.bigmodel.cn
    client = OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    for attempt in range(max_retries):
        try:
            json_instruction = (
                "\n\nRespond with ONLY a JSON object (no markdown, no explanation): "
                '{"direction": "long"|"short"|"hold", "buy_price": <number>, '
                '"sell_price": <number>, "confidence": <0.0-1.0>, "reasoning": "<brief>"}'
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert trader. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt + json_instruction},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "Zhipu API exhausted")


# ---------------------------------------------------------------------------
# Codex CLI (GPT-5.x via ChatGPT Pro plan)
# ---------------------------------------------------------------------------

# JSON schema file for codex --output-schema
_CODEX_SCHEMA_PATH = Path(__file__).resolve().parent / "codex_trade_schema.json"

def _ensure_codex_schema() -> str:
    """Write the JSON schema file once, return path."""
    if not _CODEX_SCHEMA_PATH.exists():
        schema = {
            "type": "object",
            "required": ["direction", "buy_price", "sell_price", "confidence", "reasoning"],
            "additionalProperties": False,
            "properties": {
                "direction": {"type": "string", "description": "long, short, or hold"},
                "buy_price": {"type": "number", "description": "Limit entry price, 0 if no entry"},
                "sell_price": {"type": "number", "description": "Take-profit price, 0 if no exit"},
                "confidence": {"type": "number", "description": "Confidence as decimal 0.0-1.0 (e.g. 0.75 not 75)"},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
        }
        with open(_CODEX_SCHEMA_PATH, "w") as f:
            json.dump(schema, f)
    return _CODEX_SCHEMA_PATH


def call_openai_responses(prompt: str, model: str = "gpt-5.4", max_retries: int = 5,
                          reasoning_effort: str = "low",
                          cache_model: str | None = None,
                          provider_call_models: list[str] | None = None) -> TradePlan:
    """Call GPT-5.x via OpenAI Responses API with structured outputs."""
    cache_key = cache_model or model
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

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
                "confidence": {"type": "number", "description": "Confidence as decimal 0.0-1.0 (e.g. 0.75 not 75)"},
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
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


def call_codex(prompt: str, model: str = "gpt-5.4", max_retries: int = 2,
               reasoning_effort: str = "low",
               cache_model: str | None = None,
               provider_call_models: list[str] | None = None) -> TradePlan:
    """Call GPT-5.x via codex CLI (routes through ChatGPT Pro plan)."""
    cache_key = cache_model or model
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(model)

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
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
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
    is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower() or "exhausted" in err_str.lower()
    if is_rate_limit:
        delay_match = re.search(r"retry.?(?:in|after).?(\d+\.?\d*)", err_str, re.IGNORECASE)
        if delay_match:
            wait = float(delay_match.group(1)) + 1
        else:
            wait = min(30 * (2 ** attempt), 300)
        if attempt < max_retries - 1:
            time.sleep(wait)
            return
    elif attempt < max_retries - 1:
        wait = min(2 * (2 ** attempt), 60)
        time.sleep(wait)
        return


# ---------------------------------------------------------------------------
# OpenRouter (routes Claude/GPT/etc through openrouter.ai)
# ---------------------------------------------------------------------------

def call_openrouter(prompt: str, model: str = "anthropic/claude-opus-4-6", max_retries: int = 5,
                    cache_model: str | None = None,
                    provider_call_models: list[str] | None = None) -> TradePlan:
    """Call any model via OpenRouter (OpenAI-compatible, uses OPENROUTER_API_KEY).

    Model names use OpenRouter format, e.g.:
      anthropic/claude-opus-4-6
      anthropic/claude-sonnet-4-6
    """
    cache_key = cache_model or f"openrouter/{model}"
    cached = get_cached(cache_key, prompt)
    if cached is not None:
        return TradePlan(**cached)
    if provider_call_models is not None:
        provider_call_models.append(f"openrouter/{model}")

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

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
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
        },
    }

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": json_schema},
                max_tokens=1024,
                temperature=0.3,
            )
            data = json.loads(resp.choices[0].message.content)
            plan = TradePlan(
                direction=str(data.get("direction", "hold")).lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=_normalize_confidence(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
            set_cached(cache_key, prompt, plan.__dict__)
            return plan
        except Exception as e:
            _handle_retry(e, attempt, max_retries)
    return TradePlan("hold", 0, 0, 0, "API exhausted")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PROVIDER_FNS = {
    "gemini": call_gemini,
    "openai": call_openai,
    "openai_responses": call_openai_responses,
    "anthropic": call_anthropic,
    "deepseek": call_deepseek,
    "xai": call_xai,
    "xai_tools": call_xai_with_tools,
    "grok_ctx_gemini": call_grok_context_gemini,
    "xai_openai": call_xai_openai,
    "zhipu": call_zhipu,
    "codex": call_codex,
    "openrouter": call_openrouter,
}

# Model -> provider mapping for convenience
MODEL_PROVIDERS = {
    # Gemini
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-pro": "gemini",
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
    # Anthropic (direct) — only if ANTHROPIC_API_KEY is set
    "claude-opus-4-6": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
    # Anthropic via OpenRouter (use openrouter/ prefix)
    "openrouter/anthropic/claude-opus-4-6": "openrouter",
    "openrouter/anthropic/claude-sonnet-4-6": "openrouter",
    # DeepSeek
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
    # xAI Grok (via xai_sdk with structured Pydantic outputs)
    "grok-4-1-fast": "xai",
    "grok-4.20-reasoning": "xai",
    "grok-4.20-multi-agent-0309": "xai",
    "grok-4.20-0309-reasoning": "xai",
    # Zhipu GLM
    "glm-5.1": "zhipu",
    "glm-4-plus": "zhipu",
}


def _serialize_trade_plan(plan: TradePlan) -> str:
    return json.dumps(
        {
            "direction": plan.direction,
            "buy_price": float(plan.buy_price),
            "sell_price": float(plan.sell_price),
            "confidence": float(plan.confidence),
            "reasoning": plan.reasoning,
            "allocation_pct": float(getattr(plan, "allocation_pct", 0.0) or 0.0),
        },
        indent=2,
        sort_keys=True,
    )


def _build_reprompt_prompt(
    original_prompt: str,
    prior_plan: TradePlan,
    *,
    pass_index: int,
    total_passes: int,
) -> str:
    return (
        "You are reviewing and potentially revising a trading plan.\n"
        f"This is review pass {pass_index} of {total_passes}.\n"
        "Re-read the full original task, then decide whether the prior plan should be kept "
        "or improved. Optimize the complete plan for executable limit prices, expected net "
        "PnL after fees/slippage, and consistency with the portfolio/position constraints.\n"
        "You may keep the plan unchanged if it is already best. You may also revise any field, "
        "including direction, buy_price, sell_price, confidence, reasoning, and allocation_pct.\n"
        "Return ONLY the final trade-plan JSON in the same schema as before.\n\n"
        "ORIGINAL TASK:\n"
        f"{original_prompt}\n\n"
        "PRIOR PLAN JSON:\n"
        f"{_serialize_trade_plan(prior_plan)}\n"
    )


def _plan_is_actionable(plan: TradePlan) -> bool:
    if str(plan.direction).lower().strip() != "hold":
        return True
    if float(getattr(plan, "buy_price", 0.0) or 0.0) > 0.0:
        return True
    if float(getattr(plan, "sell_price", 0.0) or 0.0) > 0.0:
        return True
    return False


def _plan_has_entry(plan: TradePlan) -> bool:
    return float(getattr(plan, "buy_price", 0.0) or 0.0) > 0.0


def _should_run_reprompt(plan: TradePlan, reprompt_policy: str) -> bool:
    if reprompt_policy == "always":
        return True
    if reprompt_policy == "actionable":
        return _plan_is_actionable(plan)
    if reprompt_policy == "entry_only":
        return _plan_has_entry(plan)
    raise ValueError(
        "reprompt_policy must be one of ('always', 'actionable', 'entry_only'), "
        f"got {reprompt_policy!r}"
    )


def _passes_reprompt_filters(
    plan: TradePlan,
    *,
    review_max_confidence: float | None = None,
) -> bool:
    if review_max_confidence is not None:
        confidence = float(getattr(plan, "confidence", 0.0) or 0.0)
        if confidence > review_max_confidence:
            return False
    return True


def _resolve_provider_and_model(
    model: str,
    provider: Optional[str] = None,
) -> tuple[str, str]:
    resolved_model = model
    resolved_provider = provider or MODEL_PROVIDERS.get(resolved_model)
    if resolved_provider is None:
        if "gemini" in resolved_model:
            resolved_provider = "gemini"
        elif resolved_model.startswith("openrouter/"):
            resolved_provider = "openrouter"
        elif "claude" in resolved_model:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                resolved_provider = "openrouter"
                resolved_model = (
                    f"openrouter/anthropic/{resolved_model}"
                    if not resolved_model.startswith("anthropic/")
                    else f"openrouter/{resolved_model}"
                )
            else:
                resolved_provider = "anthropic"
        elif "deepseek" in resolved_model:
            resolved_provider = "deepseek"
        elif "grok" in resolved_model:
            resolved_provider = "xai"
        elif "glm" in resolved_model:
            resolved_provider = "zhipu"
        else:
            resolved_provider = "openai"
    return resolved_provider, resolved_model


def call_llm(prompt: str, model: str, provider: Optional[str] = None,
             thinking_level: Optional[str] = None,
             review_thinking_level: Optional[str] = None,
             reasoning_effort: Optional[str] = None,
             cache_only: bool = False,
             reprompt_passes: int = 1,
             reprompt_policy: str = "always",
             review_max_confidence: float | None = None,
             review_model: str | None = None,
             review_cache_namespace: str | None = None,
             call_models: list[str] | None = None,
             provider_models: list[str] | None = None) -> TradePlan:
    """Call any LLM provider with auto-detection.

    For Claude models when ANTHROPIC_API_KEY is not set, use model name
    "openrouter/anthropic/claude-opus-4-6" to route via OpenRouter.
    """
    if reprompt_passes < 1:
        raise ValueError(f"reprompt_passes must be >= 1, got {reprompt_passes}")
    _should_run_reprompt(TradePlan("hold", 0, 0, 0, ""), reprompt_policy)
    primary_provider, primary_model = _resolve_provider_and_model(model, provider)
    review_model_name = review_model or primary_model
    review_thinking = review_thinking_level if review_thinking_level is not None else thinking_level

    def _call_once(
        prompt_text: str,
        *,
        model_name: str,
        thinking: Optional[str],
        cache_key_model: str,
    ) -> TradePlan:
        provider_name, resolved_model = _resolve_provider_and_model(
            model_name,
            provider if model_name == primary_model else None,
        )
        if call_models is not None:
            call_models.append(resolved_model)
        if cache_only:
            cached = get_cached(cache_key_model, prompt_text)
            if cached is None:
                raise CacheMissError(f"No cached response for model={cache_key_model}")
            return TradePlan(**cached)
        cached = get_cached(cache_key_model, prompt_text)
        if cached is not None:
            return TradePlan(**cached)
        fn = PROVIDER_FNS[provider_name]
        if provider_name == "gemini" and thinking:
            return fn(
                prompt_text,
                model=resolved_model,
                thinking_level=thinking,
                cache_model=cache_key_model,
                provider_call_models=provider_models,
            )
        if provider_name == "openai_responses":
            return fn(
                prompt_text,
                model=resolved_model,
                reasoning_effort=reasoning_effort or "low",
                cache_model=cache_key_model,
                provider_call_models=provider_models,
            )
        if provider_name == "anthropic":
            anthropic_thinking = bool(thinking)
            return fn(
                prompt_text,
                model=resolved_model,
                thinking=anthropic_thinking,
                effort=reasoning_effort,
                cache_model=cache_key_model,
                provider_call_models=provider_models,
            )
        if provider_name == "xai":
            return fn(
                prompt_text,
                model=resolved_model,
                thinking=bool(thinking),
                cache_model=cache_key_model,
                provider_call_models=provider_models,
            )
        if provider_name == "openrouter":
            # Strip "openrouter/" prefix for the actual model name passed to OpenRouter
            or_model = resolved_model.removeprefix("openrouter/")
            return fn(
                prompt_text,
                model=or_model,
                cache_model=cache_key_model,
                provider_call_models=provider_models,
            )
        return fn(
            prompt_text,
            model=resolved_model,
            cache_model=cache_key_model,
            provider_call_models=provider_models,
        )

    plan = _call_once(
        prompt,
        model_name=primary_model,
        thinking=thinking_level,
        cache_key_model=primary_model,
    )

    for pass_index in range(2, reprompt_passes + 1):
        if not _should_run_reprompt(plan, reprompt_policy):
            break
        if not _passes_reprompt_filters(
            plan,
            review_max_confidence=review_max_confidence,
        ):
            break
        reprompt = _build_reprompt_prompt(
            prompt,
            plan,
            pass_index=pass_index,
            total_passes=reprompt_passes,
        )
        review_cache_model = review_model_name
        if review_cache_namespace:
            review_cache_model = f"{review_model_name}::{review_cache_namespace}"
        plan = _call_once(
            reprompt,
            model_name=review_model_name,
            thinking=review_thinking,
            cache_key_model=review_cache_model,
        )
    return plan
