from pathlib import Path

from neuraldailymarketsimulator.simulator import _load_non_tradable_file


def test_load_non_tradable_json(tmp_path: Path) -> None:
    path = tmp_path / "nt.json"
    path.write_text("""{\n  \"non_tradable\": [{\"symbol\": \"AAPL\"}, \"msft\"]\n}\n""", encoding="utf-8")
    loaded = _load_non_tradable_file(path)
    assert {s.upper() for s in loaded} == {"AAPL", "MSFT"}


def test_load_non_tradable_lines(tmp_path: Path) -> None:
    path = tmp_path / "nt.txt"
    path.write_text("AAPL\nmsft\n\nBTCUSD", encoding="utf-8")
    loaded = _load_non_tradable_file(path)
    assert loaded == ["AAPL", "msft", "BTCUSD"]
