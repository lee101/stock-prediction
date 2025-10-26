import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(2)

    path = Path(sys.argv[1]).expanduser()
    index = int(sys.argv[2])
    position = int(sys.argv[3])

    data = json.loads(path.read_text())
    try:
        value = data[index]
    except IndexError:
        sys.exit(3)

    if position < 0 or position >= len(value):
        sys.exit(1)

    sys.exit(ord(value[position]))


if __name__ == "__main__":
    main()
