from datetime import datetime

from toolbox import context, io, models


def test_batched_basic():
    """The `batched` function should yield consecutive lists of length n,
    preserving order, with a final shorter batch if needed.
    """
    assert list(io.batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batched_n_lt_1():
    """The `batched` function should treat non-positive n as a request for
    a single empty batch.
    This matches the current implementation contract. Do I care?
    """
    assert list(io.batched([1, 2], 0)) == [[]]


def test_friendly_size_units():
    """The `friendly_size` function should format byte counts into stable
    human-readable units (bytes/kb/MB) using expected thresholds and casing.
    """
    assert io.friendly_size(10) == "10 bytes"
    assert io.friendly_size(1024) == "1024 bytes"
    assert io.friendly_size(1025) == "1 kb"
    assert io.friendly_size(1024 * 1024 + 10) == "1 MB"


def test_read_csv_missing_yields_nothing(tmp_path):
    """The `read_csv` function should be tolerant of missing files and
    should yield no rows rather than raising.
    """
    rows = list(io.read_csv(tmp_path / "missing.csv"))
    assert rows == []


def test_read_csv_yields_rows(tmp_path):
    """The `read_csv` function should yield dictionaries keyed by csv headers
    with string values from each row.
    """
    p = tmp_path / "a.csv"
    p.write_text("pid,date,message\n1,2,hi\n", encoding="utf-8")
    rows = list(io.read_csv(p))
    assert rows == [{"pid": "1", "date": "2", "message": "hi"}]


def test_linecount_missing_returns_0(tmp_path):
    """The `linecount` function should return 0 for a missing file path
    (fast-path for non-existent inputs).
    """
    assert io.linecount(tmp_path / "nope.txt") == 0


def test_linecount_counts_lines(tmp_path):
    """The `linecount` function should return the number of newline-delimited
    lines in an existing text file.
    """
    p = tmp_path / "x.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    assert io.linecount(p) == 3


def test_rotate_output_archive_rotates_and_prunes(tmp_path, monkeypatch, config_for):
    """The `rotate_output_archive` function should archive a non-empty output
    directory into a timestamped archive folder, and recreate the output dir,
    and prune older archives beyond the retention count.
    """
    export_dir = tmp_path / "export"
    download_dir = tmp_path / "downloads"
    output_dir = tmp_path / "out"
    for d in (export_dir, download_dir, output_dir):
        d.mkdir()
    cfg = config_for(
        tmp_path,
        export_dir=export_dir,
        download_dir=download_dir,
        output_dir=output_dir,
    )
    ctx = context.Context(
        args=models.CliArgs(mode="download_files"),
        config=cfg,
        path=context.paths(cfg),
        dry_run=True,
    )

    # make output non-empty so it will archive
    (output_dir / "something.txt").write_text("x", encoding="utf-8")
    # create existing archives to trigger pruning
    archive_dir = output_dir.with_name(output_dir.name + ".archive")
    archive_dir.mkdir()
    for i in range(5):
        (archive_dir / f"old{i}").mkdir()

    class FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2020, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(io, "datetime", FixedDateTime)

    # prune down to 2
    io.rotate_output_archive(ctx, count=2)
    # output should exist again and be empty (aside from new work)
    assert output_dir.exists()
    assert (output_dir / "something.txt").exists() is False

    # archives should be at most 2 + the new one created
    dirs = [p for p in archive_dir.iterdir() if p.is_dir()]
    assert len(dirs) <= 3  # old pruned + new archive (timestamp) + maybe some remain


def test_log_writes_line(ctx, monkeypatch):
    """The `log` function should append a timestamped line containing the
    provided message to the run's log file.
    """

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2020, 1, 2, 3, 4, 5)

    monkeypatch.setattr(io, "datetime", FixedDateTime)
    io.log(ctx, text="hello")
    txt = ctx.path.log.read_text(encoding="utf-8")
    assert "hello" in txt
    assert "2020-01-02" in txt
