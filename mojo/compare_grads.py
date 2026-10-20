import sys, os, re, numpy as np
CTX = sys.argv[1] if len(sys.argv) > 1 else "512"
d = "mojo/data/train"
rows = []
for f in sorted(os.listdir(f"{d}/grads_{CTX}")):
    k = f[:-4]
    a = np.fromfile(f"{d}/grads_{CTX}/{f}", dtype=np.float32)
    p = f"{d}/grads_mojo_{CTX}/{f}"
    b = np.fromfile(p, dtype=np.float32) if os.path.exists(p) else np.full_like(a, np.nan)
    rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-20)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rows.append((rel, cos, k, np.linalg.norm(a), np.linalg.norm(b)))
rows.sort(reverse=True)
cat = {}
for rel, cos, k, na, nb in rows:
    c = re.sub(r"^l\d+_", "", k)
    cat.setdefault(c, []).append((rel, cos))
for c, v in sorted(cat.items(), key=lambda kv: -max(x[0] for x in kv[1])):
    print(f"  {c:6s} rel err median {np.median([x[0] for x in v]):.5f} max {max(x[0] for x in v):.5f}  cos min {min(x[1] for x in v):.6f}  n={len(v)}")
print("worst 5:", [(round(r, 5), k) for r, c, k, _, _ in rows[:5]])
