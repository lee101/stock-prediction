import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(2)

    path = Path(sys.argv[1]).expanduser()
    index = int(sys.argv[2])
    data = path.read_bytes()
    if index < 0 or index >= len(data):
        sys.exit(1)

    sys.exit(data[index])


if __name__ == "__main__":
    main()
