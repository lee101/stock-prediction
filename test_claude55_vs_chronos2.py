import json, os, re, sys, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, anthropic
from google import genai
from google.genai import types

N = int(os.getenv("N", "40"))
CTX = 10
cache_dir = Path(".llm_cache_v2"); cache_dir.mkdir(exist_ok=True)
ac = anthropic.Anthropic(default_headers={"Accept-Encoding": "gzip"})
gc = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
SYS = "You are a number guessing system. Final line of your reply must be only the numeric prediction."

def cached(key, fn):
    p = cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if p.exists():
        return json.loads(p.read_text())["r"]
    r = fn()
    if r is not None:
        p.write_text(json.dumps({"r": r}))
    return r

def parse(s):
    if not s: return None
    m = re.findall(r'-?\d*\.?\d+(?:[eE][-+]?\d+)?', s.strip().splitlines()[-1])
    return float(m[-1]) if m else None

def claude(model, prompt):
    def f():
        for th in ({"type": "enabled", "budget_tokens": 8000}, {"type": "adaptive"}):
            try:
                m = ac.messages.create(model=model, max_tokens=16000, thinking=th,
                    messages=[{"role": "user", "content": f"{SYS}\n\n{prompt}"}])
                return "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            except Exception as e:
                err = e
        print("claude err", model, err, file=sys.stderr)
    return cached(model + prompt, f)

def gemini(model, prompt):
    def f():
        try:
            r = gc.models.generate_content(model=model, contents=f"{SYS}\n\n{prompt}",
                config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="HIGH")))
            return r.text
        except Exception as e:
            print("gemini err", model, e, file=sys.stderr)
    return cached(model + prompt, f)

d = pd.read_csv("trainingdata/BTCUSD.csv").sort_values("timestamp").reset_index(drop=True)
d["ret"] = d.close.pct_change()
d = d.dropna().reset_index(drop=True)
idxs = list(range(len(d) - N, len(d)))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper
w = Chronos2OHLCWrapper.from_pretrained("amazon/chronos-2", device_map="cuda", target_columns=("close",), default_context_length=512)
def chron(t):
    c = d.iloc[:t][["timestamp", "open", "high", "low", "close", "volume"]].copy()
    b = w.predict_ohlc(c, symbol="BTCUSD", prediction_length=1, context_length=min(512, len(c)))
    return float(b.quantile_frames[0.5]["close"].iloc[0]) / float(c.close.iloc[-1]) - 1

models = {
    "opus-5-5": lambda p: claude("claude-opus-5-5", p),
    "sonnet-5-5": lambda p: claude("claude-sonnet-5-5", p),
}
prompts = {t: f"Given these recent daily BTC return values: {d.ret.iloc[t-CTX:t].tolist()}, predict the next daily return as a decimal number. End with the numeric prediction alone on the last line." for t in idxs}
rows = {"actual": [d.ret.iloc[t] for t in idxs], "zero": [0.0] * N, "chronos2": [chron(t) for t in idxs]}
with ThreadPoolExecutor(8) as ex:
    for name, fn in models.items():
        rows[name] = [parse(r) for r in ex.map(lambda t: fn(prompts[t]), idxs)]
df = pd.DataFrame(rows)
df.to_csv("claude55_vs_chronos2.csv", index=False)
a = df.actual
print(f"N={N}")
for k in df.columns[1:]:
    p = df[k]; ok = p.notna()
    print(f"{k:16s} n={ok.sum():3d} MAE={np.abs(a[ok]-p[ok]).mean():.5f} dir_acc={(np.sign(a[ok])==np.sign(p[ok])).mean():.3f} corr={np.corrcoef(a[ok],p[ok])[0,1] if p[ok].std()>0 else float('nan'):.3f}")
