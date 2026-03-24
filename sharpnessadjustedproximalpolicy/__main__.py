"""Entry point: python -m sharpnessadjustedproximalpolicy [sweep|forever|evaluate|scaled|caches]"""
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
sys.argv = [sys.argv[0]] + sys.argv[2:]

if cmd == "sweep":
    from .sweep import main
elif cmd == "forever":
    from .run_forever import main
elif cmd == "evaluate":
    from .evaluate import main
elif cmd == "scaled":
    from .run_scaled import main
elif cmd == "caches":
    from .generate_caches import main
elif cmd == "allpairs":
    from .sweep_all_pairs import main
else:
    print("Usage: python -m sharpnessadjustedproximalpolicy [sweep|forever|evaluate|scaled|caches|allpairs] [args...]")
    sys.exit(1)

main()
