import json, sys

text = sys.stdin.read()
decoder = json.JSONDecoder()
pos = 0
last = None
while pos < len(text):
    idx = text.find("{", pos)
    if idx < 0:
        break
    try:
        obj, end = decoder.raw_decode(text, idx)
        last = obj
        pos = end
    except Exception:
        pos = idx + 1

if last and "median_total_return" in last:
    med = last["median_total_return"] * 100
    p10 = last["p10_total_return"] * 100
    neg = last["negative_windows"]
    sortino = last["median_sortino"]
    print(f"med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sortino={sortino:.2f}")
else:
    print("FAILED")
