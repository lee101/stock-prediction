import pathlib

import pytest

from tools.summarize_results import cleanup_preview_shards, write_preview_assets


def test_write_preview_assets_creates_expected_files(tmp_path: pathlib.Path) -> None:
    preview_dir = tmp_path / "preview"
    markdown = "# Title\nsecond line\nthird"

    write_preview_assets(markdown, preview_dir, max_chars=10)

    preview_file = preview_dir / "results_preview.txt"
    assert preview_file.read_text(encoding="utf-8") == markdown[:10]

    shards = sorted(preview_dir.glob("results_preview_char_*.txt"))
    shard_contents = [path.read_text(encoding="utf-8") for path in shards]
    assert shard_contents == list(markdown[:10])


@pytest.mark.parametrize("keep_preview", (True, False))
def test_cleanup_preview_shards(tmp_path: pathlib.Path, keep_preview: bool) -> None:
    preview_dir = tmp_path
    preview_file = preview_dir / "results_preview.txt"
    preview_file.write_text("abc", encoding="utf-8")

    for idx, char in enumerate("abc"):
        (preview_dir / f"results_preview_char_{idx}.txt").write_text(
            char, encoding="utf-8"
        )

    cleanup_preview_shards(preview_dir, keep_preview_file=keep_preview)

    assert not list(preview_dir.glob("results_preview_char_*.txt"))
    assert preview_file.exists() is keep_preview
