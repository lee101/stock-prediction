#!/usr/bin/env python3
"""Filter claude stream-json output to human-readable TXT/TOOL/RESULT lines."""
import json
import sys


def main() -> None:
    for line in sys.stdin:
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type", "")
        if t == "assistant":
            msg = d.get("message", {})
            for c in (msg.get("content") or []):
                if c.get("type") == "text":
                    txt = (c.get("text") or "").strip()
                    if txt:
                        print("TXT:", txt[:500], flush=True)
                elif c.get("type") == "tool_use":
                    inp = str(c.get("input", ""))[:220]
                    name = c.get("name", "")
                    print(f"TOOL: {name} | {inp}", flush=True)
        elif t == "result":
            res = str(d.get("result", ""))[:3000]
            print("RESULT:", res, flush=True)


if __name__ == "__main__":
    main()
