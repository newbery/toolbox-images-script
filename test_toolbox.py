
import csv
import json
import sys
from argparse import Namespace
from datetime import datetime as real_datetime
from datetime import timezone
from pathlib import Path

import pytest


def _import_toolbox():
    """Import toolbox, falling back to loading ../toolbox.py if needed."""
    try:
        import toolbox  # type: ignore
        return toolbox
    except Exception:
        # Fallback: load toolbox.py next to project root
        import importlib.util

        here = Path(__file__).resolve()
        root = here.parents[1] if len(here.parents) > 1 else here.parent
        toolbox_path = root / "toolbox.py"
        if not toolbox_path.exists():
            raise
        spec = importlib.util.spec_from_file_location("toolbox", toolbox_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["toolbox"] = mod
        spec.loader.exec_module(mod)
        return mod


toolbox = _import_toolbox()


class DummyAliveBar:
    """Replacement for alive_progress.alive_bar used in unit tests."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []

    def __enter__(self):
        def bar(n=1):
            self.calls.append(n)
        return bar

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture(autouse=True)
def _patch_alive_bar(monkeypatch):
    monkeypatch.setattr(toolbox, "alive_bar", DummyAliveBar)
    yield


@pytest.fixture
def context_tmp(tmp_path) -> Namespace:
    export_dir = tmp_path / "export"
    download_dir = tmp_path / "downloads"
    output_dir = tmp_path / "out"
    export_dir.mkdir()
    download_dir.mkdir()
    output_dir.mkdir()

    cfg = Namespace(
        export_dir=str(export_dir),
        download_dir=str(download_dir),
        output_dir=str(output_dir),
        old_url="https://old.example.com/",
        old_url_thumb="https://old.example.com/thumb/",
        new_url="https://new.example.com/",
        skip_days="0",
        test_post_id="",
        dry_run="true",
    )

    context = Namespace(config=cfg)
    context.path = toolbox.paths(cfg)
    context.args = Namespace(mode="download_files", verbose=False, apply=False, dry_run=True, yes=True)

    # Most unit tests run in dry-run mode by default.
    context.dry_run = True

    return context


# -------------------------
# Small, pure helper tests
# -------------------------

def test_batched_basic():
    """The `batched` function should yield consecutive lists of length n,
    preserving order, with a final shorter batch if needed.
    """
    assert list(toolbox.batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batched_n_lt_1():
    """The `batched` function should treat non-positive n as a request for
    a single empty batch.
    
    This matches the current implementation contract. Do I care?
    """
    assert list(toolbox.batched([1, 2], 0)) == [[]]


def test_friendly_size_units():
    """The `friendly_size` function should format byte counts into stable
    human-readable units (bytes/kb/MB) using expected thresholds and casing.
    """
    assert toolbox.friendly_size(10) == "10 bytes"
    assert toolbox.friendly_size(1024) == "1024 bytes"
    assert toolbox.friendly_size(1025) == "1 kb"
    assert toolbox.friendly_size(1024 * 1024 + 10) == "1 MB"


def test_find_urls_func_filters_and_sorts():
    """The `find_urls_func` function should return a sorted, de-duplicated
    list of urls that match the configured legacy prefix, excluding
    non-matching hosts.
    """
    find_urls = toolbox.find_urls_func("https://old.example.com/")
    html = (
        '<p>'
        '<img src="https://old.example.com/a.jpg"/>'
        '<img src="https://old.example.com/b.jpg"/>'
        '<img src="https://other.example.com/c.jpg"/>'
        '<img src="https://old.example.com/a.jpg"/>'
        "</p>"
    )
    assert find_urls(html) == ["https://old.example.com/a.jpg", "https://old.example.com/b.jpg"]


def test_find_legacy_urls_in_img_and_link():
    """The `find_legacy_urls` function should extract legacy attachment references
    from both <a href='/file?id=...'> links and legacy hosted <img src='...'> urls.
    """
    html = (
        '<a href="/file?id=123">x</a>'
        '<img src="http://files.websitetoolbox.com/999/123/a.jpg"/>'
        '<img src="https://example.com/ignore.jpg"/>'
    )
    assert toolbox.find_legacy_urls(html) == [
        "/file?id=123",
        "http://files.websitetoolbox.com/999/123/a.jpg",
    ]


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://example.com/file?id=123", "123"),
        ("https://s3.amazonaws.com/files.websitetoolbox.com/999/123/a.jpg", "123"),
        ("https://cdn.example.com/999/123/a.jpg", "123"),
        ("https://cdn.example.com/x/123", "123"),
        ("not a url", None),
    ],
)
def test_fileid_from_url(url, expected):
    """The `fileid_from_url` function should extract the fileid component from
    supported url shapes and and should return None when the input is not a
    recognized file url.
    """
    assert toolbox.fileid_from_url(url) == expected


def test_remove_bad_url_de_links_image_and_adds_notice():
    """The `remove_bad_url` function should remove src/href references to a
    known-bad url and insert a visible '(missing image)' marker plus an HTML
    comment for traceability.
    """
    bad = "https://old.example.com/999/missing.jpg"
    html = f'<p><a href="{bad}"><img src="{bad}"/></a> hello</p>'
    out = toolbox.remove_bad_url(html, bad)

    # src and href should be removed
    assert 'src="' not in out
    assert 'href="' not in out

    # notice inserted
    assert "missing-image" in out
    assert "(missing image)" in out

    # comment includes Bad URL marker
    assert "Bad URL:" in out


def test_get_new_url_func_basic_and_thumb():
    """The `get_new_url_func` function should rewrite old urls to the new prefix,
    preserving the '/thumb/' variant when present.
    """
    f = toolbox.get_new_url_func(
        old_prefix="https://old.example.com/",
        thumb_prefix="https://old.example.com/thumb/",
        new_prefix="https://new.example.com/",
    )
    assert f("https://old.example.com/123/a.jpg") == "https://new.example.com/123/a.jpg"
    assert f("https://old.example.com/thumb/123/a.jpg") == "https://new.example.com/thumb/123/a.jpg"


def test_get_new_url_func_handles_param_quote_unquote():
    """The `get_new_url_func` function should safely quote urls when embedding
    them into query parameters, and should unquote them when converting from
    param-encoded to path-style urls.
    """

    # Old has no param, new has param -> safe_quote should be used
    f = toolbox.get_new_url_func(
        old_prefix="https://old.example.com/",
        thumb_prefix="",
        new_prefix="https://new.example.com/?url=",
    )
    
    # '#' should remain quoted so it doesn't become a fragment
    out = f("https://old.example.com/a#b.jpg")
    assert out.startswith("https://new.example.com/?url=")
    assert "%23" in out  # # is quoted

    # Old has param but new does not -> unquote should be used
    g = toolbox.get_new_url_func(
        old_prefix="https://old.example.com/?url=",
        thumb_prefix="",
        new_prefix="https://new.example.com/",
    )
    out2 = g("https://old.example.com/?url=a%23b.jpg")
    assert out2 == "https://new.example.com/a#b.jpg"


# -------------------------
# IO helpers
# -------------------------

def test_read_csv_missing_yields_nothing(tmp_path):
    """The `read_csv` function should be tolerant of missing files and
    should yield no rows rather than raising.
    """
    rows = list(toolbox.read_csv(tmp_path / "missing.csv"))
    assert rows == []


def test_read_csv_yields_rows(tmp_path):
    """The `read_csv` function should yield dictionaries keyed by csv headers
    with string values from each row.
    """
    p = tmp_path / "a.csv"
    p.write_text("pid,date,message\n1,2,hi\n", encoding="utf-8")
    rows = list(toolbox.read_csv(p))
    assert rows == [{"pid": "1", "date": "2", "message": "hi"}]


def test_linecount_missing_returns_0(tmp_path):
    """The `linecount` function should return 0 for a missing file path
    (fast-path for non-existent inputs).
    """
    assert toolbox.linecount(tmp_path / "nope.txt") == 0


def test_linecount_counts_lines(tmp_path):
    """The `linecount` function should return the number of newline-delimited
    lines in an existing text file.
    """
    p = tmp_path / "x.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    assert toolbox.linecount(p) == 3


def test_rotate_output_archive_rotates_and_prunes(tmp_path, monkeypatch):
    """The `rotate_output_archive` function should archive a non-empty output
    directory into a timestamped archive folder, and recreate the output dir,
    and prune older archives beyond the retention count.
    """
    export_dir = tmp_path / "export"
    download_dir = tmp_path / "downloads"
    output_dir = tmp_path / "out"
    for d in (export_dir, download_dir, output_dir):
        d.mkdir()

    cfg = Namespace(export_dir=str(export_dir), download_dir=str(download_dir), output_dir=str(output_dir))
    context = Namespace(config=cfg)
    context.path = toolbox.paths(cfg)

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
            return real_datetime(2020, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(toolbox, "datetime", FixedDateTime)

    # prune down to 2
    toolbox.rotate_output_archive(context, count=2)

    # output should exist again and be empty (aside from new work)
    assert output_dir.exists()
    assert (output_dir / "something.txt").exists() is False

    # archives should be at most 2 + the new one created
    dirs = [p for p in archive_dir.iterdir() if p.is_dir()]
    assert len(dirs) <= 3  # old pruned + new archive (timestamp) + maybe some remain


def test_log_writes_line(context_tmp, monkeypatch):
    """The `log` function should append a timestamped line containing the
    provided message to the run's log file.
    """
    class FixedDateTime:
        @classmethod
        def now(cls):
            return real_datetime(2020, 1, 2, 3, 4, 5)

    monkeypatch.setattr(toolbox, "datetime", FixedDateTime)
    toolbox.log(context_tmp, text="hello")
    txt = context_tmp.path.log.read_text(encoding="utf-8")
    assert "hello" in txt
    assert "2020-01-02" in txt


def test_config_merges_dotenv_and_env(monkeypatch):
    """The `config` function should merge .env and .env.secrets (secrets win),
    then apply TOOLBOX_* environment overrides while ignoring unrelated
    environment variables.
    """

    # Patch dotenv_values to avoid touching disk
    def fake_dotenv_values(filename):
        if filename == ".env":
            return {"A": "1", "B": "2"}
        if filename == ".env.secrets":
            return {"B": "secret", "C": "3"}
        return {}

    monkeypatch.setattr(toolbox, "dotenv_values", fake_dotenv_values)
    monkeypatch.setenv("TOOLBOX_D", "4")
    monkeypatch.setenv("OTHER", "x")

    cfg = toolbox.config()

    # dotenv keys should be lowercased
    assert cfg.a == "1"

    # secrets override
    assert cfg.b == "secret"
    assert cfg.c == "3"

    # env override
    assert cfg.d == "4"
    assert not hasattr(cfg, "other")


def test_paths_builds_expected_paths(tmp_path):
    """The `paths` function should build the derived filesystem paths
    (exports/downloads/output and expected filenames) from the config directories.
    """
    cfg = Namespace(
        export_dir=str(tmp_path / "export"),
        download_dir=str(tmp_path / "downloads"),
        output_dir=str(tmp_path / "out"),
    )
    paths = toolbox.paths(cfg)
    assert paths.export_dir.name == "export"
    assert paths.posts.name == "posts.csv"
    assert paths.fileids_to_delete.name == "fileids_to_delete.json"


# -------------------------
# Core pipeline function tests (mocking clients)
# -------------------------

def test_posts_from_export_collects_urls(context_tmp):
    """The `posts_from_export` function should parse the export posts csv,
    extract image urls, and write a normalized `posts_from_export` csv with
    per-post url lists.
    """

    # write export/posts.csv
    posts_csv = context_tmp.path.export_dir / "posts.csv"
    posts_csv.write_text(
        "pid,date,message\n"
        "1,100,<p><img src=\"https://old.example.com/123/a.jpg\"/></p>\n"
        "2,101,<p>no image</p>\n",
        encoding="utf-8",
    )

    posts = toolbox.posts_from_export(context_tmp)
    assert set(posts) == {"1", "2"}
    assert posts["1"]["image_urls"] == ["https://old.example.com/123/a.jpg"]
    assert posts["2"]["image_urls"] == []

    out_rows = list(toolbox.read_csv(context_tmp.path.posts_from_export))
    assert out_rows[0]["pid"] == "1"


def test_posts_from_api_stops_when_pid_already_seen(context_tmp):
    """The `posts_from_api` function should paginate api results into the posts
    dict, extract image urls, and stop early once it encounters a postId already
    present in the seed posts map.
    """

    class FakeApiRequests:
        def __init__(self, pages):
            self.pages = pages
            self.closed = False

        def __iter__(self):
            yield from self.pages

        def close(self):
            self.closed = True

    class FakeClient:
        def __init__(self, pages):
            self._pages = pages

        def list_posts(self):
            return FakeApiRequests(self._pages)

    # Existing post from export already processed
    posts = {"1": {"date": "100", "image_urls": []}}
    pages = [
        {"data": [{"postId": 2, "postTimestamp": "200", "message": '<img src="https://old.example.com/2.jpg"/>'}]},
        {"data": [{"postId": 1, "postTimestamp": "199", "message": "stop here"}]},
        {"data": [{"postId": 3, "postTimestamp": "198", "message": "should not be reached"}]},
    ]
    context_tmp.api_client = FakeClient(pages)

    out = toolbox.posts_from_api(context_tmp, posts)
    assert "2" in out
    assert out["2"]["image_urls"] == ["https://old.example.com/2.jpg"]

    # "3" should not be processed due to early stop
    assert "3" not in out


def test_files_from_posts_toolbox_parses_fileids_and_thumb(context_tmp):
    """The `files_from_posts` function should group image urls by fileid,
    detect thumb urls, record canonical paths, and accumulate the set of
    post ids referencing each file.
    """

    # Make it look like a toolbox/cloudfront url so toolbox=True
    context_tmp.config.old_url = "https://abc.cloudfront.net/"
    context_tmp.config.old_url_thumb = "https://abc.cloudfront.net/thumb/"
    context_tmp.config.skip_days = "0"

    posts = {
        "1": {"date": "0", "image_urls": ["https://abc.cloudfront.net/999/123/a.jpg"]},
        "2": {"date": "0", "image_urls": ["https://abc.cloudfront.net/thumb/999/123/a.jpg"]},
    }
    files = toolbox.files_from_posts(context_tmp, posts)
    assert "123" in files

    f = files["123"]
    assert f["url"].endswith("/999/123/a.jpg")
    assert f["url_thumb"].endswith("/thumb/999/123/a.jpg")
    assert f["pids"] == {"1", "2"}
    assert f["path"] == "999/123/a.jpg"


def test_files_from_posts_skips_recent_or_nonmatching_test_post(context_tmp):
    """The `files_from_posts` function should mark files as skipped when the
    containing post is newer than the configured skip_days threshold.
    """
    context_tmp.config.old_url = "https://abc.cloudfront.net/"
    context_tmp.config.old_url_thumb = ""
    context_tmp.config.skip_days = "1"  # skip anything newer than 1 day ago

    now_ts = int(real_datetime.now(timezone.utc).timestamp())
    posts = {"1": {"date": str(now_ts), "image_urls": ["https://abc.cloudfront.net/1/111/a.jpg"]}}

    files = toolbox.files_from_posts(context_tmp, posts)
    assert files["111"]["result"] == toolbox.File.skipped


def test_files_from_export_builds_new_url(context_tmp):
    """The `files_from_export` function should resolve legacy '/file?id=' links
    using attachment metadata to produce a concrete legacy file url for later
    checking/updating.
    """

    # Posts with legacy /file?id= urls; attachments.csv supplies filename
    context_tmp.config.old_url = "https://abc.cloudfront.net/"
    posts = {"1": {"date": "0", "image_urls": ["/file?id=123"]}}

    attach = context_tmp.path.export_dir / "attachment.csv"
    attach.write_text("fileid,filename\n123,a.jpg\n", encoding="utf-8")

    files = toolbox.files_from_export(context_tmp, posts)
    assert files["123"]["new_url"] == "https://abc.cloudfront.net/123/a.jpg"


def test_download_files_marks_downloaded_and_errors(context_tmp, monkeypatch):
    """The `download_files` function should download files marked missing,
    update per-file result state to downloaded or error based on the download
    outcome, and preserve skipped entries.
    """

    # Create a downloader that writes dummy files and returns size
    class FakeDownloader:
        def __init__(self):
            self.calls = []

        def download(self, url, path_new):
            self.calls.append((url, str(path_new)))
            path_new.parent.mkdir(parents=True, exist_ok=True)
            path_new.write_bytes(b"abc")
            return 3

    context_tmp.downloader = FakeDownloader()

    files = {
        "1": {"url": "https://x/1.jpg", "url_thumb": "", "url_file": "", "path": "1.jpg", "pids": {"p1"}, "result": toolbox.File.default},
        "2": {"url": "https://x/2.jpg", "url_thumb": "https://x/t2.jpg", "url_file": "", "path": "2.jpg", "pids": {"p2"}, "result": toolbox.File.default},
        "3": {"url": "https://x/3.jpg", "url_thumb": "", "url_file": "", "path": "3.jpg", "pids": {"p3"}, "result": toolbox.File.skipped},
    }

    # Make download fail for one file
    _orig_download = context_tmp.downloader.download

    def fake_download(url, path_new):
        if str(url).endswith("1.jpg"):
            return 0
        return _orig_download(url, path_new)

    context_tmp.downloader.download = fake_download  # type: ignore

    out = toolbox.download_files(context_tmp, files)
    assert out["1"]["result"] == toolbox.File.error
    assert out["2"]["result"] == toolbox.File.downloaded
    assert out["3"]["result"] == toolbox.File.skipped


def _write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_summarize_writes_posts_and_files(context_tmp):
    """The `summarize` function should write consolidated posts.csv and files.csv,
    excluding posts whose only referenced files are skipped/error according to
    the computed file results.
    """

    # posts_from_export and posts_from_api inputs
    _write_csv(
        context_tmp.path.posts_from_export,
        ["pid", "date", "image_urls", "message"],
        [
            ["1", "0", "[]", "<p>no</p>"],
            ["2", "0", "['https://old.example.com/123/a.jpg']", "<img src='https://old.example.com/123/a.jpg'/>"],
        ],
    )
    _write_csv(
        context_tmp.path.posts_from_api,
        ["pid", "date", "image_urls", "message"],
        [
            ["3", "0", "['https://old.example.com/123/a.jpg']", "<img src='https://old.example.com/123/a.jpg'/>"],
        ],
    )

    files = {
        "123": {
            "url": "https://old.example.com/123/a.jpg",
            "url_thumb": "",
            "url_file": "",
            "path": "123/a.jpg",
            "pids": {"2", "3"},
            "result": toolbox.File.downloaded,
        },
        "999": {
            "url": "https://old.example.com/999/missing.jpg",
            "url_thumb": "",
            "url_file": "",
            "path": "999/missing.jpg",
            "pids": {"1"},
            "result": toolbox.File.skipped,
        },
    }

    toolbox.summarize(context_tmp, files)

    posts_out = list(toolbox.read_csv(context_tmp.path.posts))

    # pid 1 should be skipped due to skipped file
    assert [r["pid"] for r in posts_out] == ["2", "3"]

    files_out = list(toolbox.read_csv(context_tmp.path.files))
    assert any(r["fileid"] == "123" for r in files_out)


def test_check_new_urls_respects_skip_and_generates_file_scheme(context_tmp, tmp_path, monkeypatch):
    """The `check_new_urls` function should check only non-skipped files and,
    in test-run mode with a local new_url root, generate file:// urls for
    validation via `url_ok`.
    """

    # Use local directory as "new_url" and dry-run=True to enable file:// prefix
    new_root = tmp_path / "new"
    new_root.mkdir()
    context_tmp.config.new_url = str(new_root)  # not http(s)
    context_tmp.dry_run = True

    # posts.csv with two urls, one skipped, one checked
    _write_csv(
        context_tmp.path.posts,
        ["pid", "date", "image_urls", "message"],
        [
            ["1", "0", "['https://old.example.com/1.jpg', 'https://old.example.com/2.jpg']", "x"],
        ],
    )

    files = {
        "https://old.example.com/1.jpg": {"result": toolbox.File.skipped},
        "https://old.example.com/2.jpg": {"result": toolbox.File.downloaded},
    }

    def url_ok(url: str) -> bool:
        # It should be a file:// url pointing into new_root
        assert url.startswith("file://")
        return True

    context_tmp.url_ok = url_ok

    assert toolbox.check_new_urls(context_tmp, files) is True


def test_grep_urls_in_file_finds_matching_pids(tmp_path):
    """The `grep_urls_in_file` function should return the pids of rows whose
    content contains any of the provided URL patterns, ignoring empty patterns.
    """
    updates = tmp_path / "updates.csv"
    updates.write_text(
        "pid,result,content\n"
        "1,success,hello https://a.example.com/x.jpg\n"
        "2,success,bye\n"
        "3,success,see https://b.example.com/y.jpg\n",
        encoding="utf-8",
    )
    out = toolbox.grep_urls_in_file(updates, ["https://b.example.com/y.jpg", ""])
    assert out.split() == ["3"]


def test_check_old_urls_detects_in_updated_or_nonupdated(context_tmp, tmp_path):
    """The `check_old_urls` function should return False when legacy urls
    (or legacy file references) still appear in either updated content or in
    posts that were never updated.
    """

    # Create updates.csv (updated posts content)
    _write_csv(
        context_tmp.path.updates,
        ["pid", "result", "content"],
        [
            ["10", "success", "contains https://old.example.com/123/a.jpg"],
            ["11", "success", "ok"],
        ],
    )

    # Create non-updated posts csvs that still contain a fileid pattern
    _write_csv(
        context_tmp.path.posts_from_export,
        ["pid", "date", "image_urls", "message"],
        [
            ["20", "0", "[]", "legacy =123 somewhere"],
        ],
    )
    _write_csv(
        context_tmp.path.posts_from_api,
        ["pid", "date", "image_urls", "message"],
        [
            ["21", "0", "[]", "nope"],
        ],
    )

    files_to_check = [
        {"url": "https://old.example.com/123/a.jpg", "url_thumb": "", "fileid": "123"},
    ]

    ok = toolbox.check_old_urls(context_tmp, files_to_check, legacy=False)
    assert ok is False


def test_update_posts_updates_content_and_writes_outputs(context_tmp, monkeypatch):
    """The `update_posts` function should rewrite legacy urls in post content to
    the new host, record per-post update results, and emit the set of fileids
    eligible for deletion.
    """

    # Avoid sleeping
    monkeypatch.setattr(toolbox.time, "sleep", lambda *_args, **_kwargs: None)

    # Prepare posts.csv to update
    msg = (
        "<p>"
        "<img src='https://old.example.com/123/a.jpg'/>"
        "<img src='https://old.example.com/thumb/123/a.jpg'/>"
        "<a href='/file?id=123'>file</a>"
        "</p>"
    )
    _write_csv(
        context_tmp.path.posts,
        ["pid", "date", "image_urls", "message"],
        [
            ["1", "0", "['https://old.example.com/123/a.jpg', 'https://old.example.com/thumb/123/a.jpg']", msg],
            # This one should be unchanged (skipped file)
            ["2", "0", "['https://old.example.com/555/a.jpg']", "<img src='https://old.example.com/555/a.jpg'/>"],
        ],
    )

    # Prepare files.csv input
    _write_csv(
        context_tmp.path.files,
        ["fileid", "pids", "url", "url_thumb", "url_file", "new_url", "result"],
        [
            ["123", "{\'1\'}", "https://old.example.com/123/a.jpg", "https://old.example.com/thumb/123/a.jpg", "/file?id=123", "", str(toolbox.File.downloaded.value)],
            ["555", "{'2'}", "https://old.example.com/555/a.jpg", "", "", "", str(toolbox.File.skipped.value)],
        ],
    )

    # Patch check_new_urls/check_old_urls
    monkeypatch.setattr(toolbox, "check_new_urls", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(toolbox, "check_old_urls", lambda *_args, **_kwargs: True)

    # Fake API client: should NOT be called in dry-run mode.
    class FakeClient:
        def update_post(self, _pid, _message):
            raise AssertionError("update_post should not be called in dry-run mode")

    context_tmp.api_client = FakeClient()

    toolbox.update_posts(context_tmp, legacy=False)

    # updates.csv should include a dry-run result with rewritten content for pid 1
    updates_rows = list(toolbox.read_csv(context_tmp.path.updates))
    row1 = next(r for r in updates_rows if r["pid"] == "1")
    assert row1["result"] == "dry_run"
    assert "https://new.example.com/123/a.jpg" in row1["content"]
    assert "https://new.example.com/thumb/123/a.jpg" in row1["content"]
    
    # /file?id link replaced with full url
    assert "/file?id=123" not in row1["content"]

    # In dry-run mode, fileids_to_delete.json is intentionally left empty,
    # while the would-delete set is written to fileids_to_delete.dry_run.json.
    fileids = json.loads(context_tmp.path.fileids_to_delete.read_text(encoding="utf-8"))
    assert fileids == []
    dry_fileids = json.loads(context_tmp.path.fileids_to_delete_dry_run.read_text(encoding="utf-8"))
    assert dry_fileids == ["123"]


def test_delete_files_batches_and_calls_client(context_tmp, monkeypatch):
    """The `delete_files` function should load `fileids_to_delete.json` and invoke
    the admin client's `delete_files` in batches of 100 fileids.
    """
    context_tmp.path.fileids_to_delete.write_text(json.dumps([str(i) for i in range(1, 205)]), encoding="utf-8")
    monkeypatch.setattr(toolbox.time, "sleep", lambda *_a, **_k: None)

    calls = []

    class FakeAdmin:
        def check_admin_auth(self):
            return True

        def delete_files(self, fileids):
            calls.append(list(fileids))

    context_tmp.admin_client = FakeAdmin()

    # Run in apply mode so the admin client is invoked.
    context_tmp.dry_run = False
    context_tmp.args = Namespace(yes=True)

    toolbox.delete_files(context_tmp)

    # Should batch at 100
    assert len(calls) == 3
    assert len(calls[0]) == 100
    assert len(calls[1]) == 100
    assert len(calls[2]) == 4


# -------------------------
# CLI wrappers
# -------------------------

def test_parse_args_parses_mode_and_verbose(monkeypatch):
    """The `parse_args` function should accept a valid mode, set args.mode,
    and parse -v/--verbose into args.verbose.
    """
    monkeypatch.setattr(toolbox, "modes", lambda: {"download_files": lambda _ctx: None})

    args = toolbox.parse_args(["download_files"])
    assert args.mode == "download_files"
    assert args.verbose is False

    args2 = toolbox.parse_args(["-v", "download_files"])
    assert args2.mode == "download_files"
    assert args2.verbose is True


def test_parse_args_rejects_unknown_mode(monkeypatch):
    """The `parse_args` function should reject unknown modes via argparse by
    raising SystemExit for invalid choices.
    """
    monkeypatch.setattr(toolbox, "modes", lambda: {"download_files": lambda _ctx: None})

    with pytest.raises(SystemExit):
        toolbox.parse_args(["nope"])


def test_init_context_populates_context_and_dry_run_false(monkeypatch, capsys):
    """The `init_context` function should populate context with args/config/paths
    and set `dry_run` False when `config.dry_run` is exactly the string 'false'.
    """
    cfg = Namespace(dry_run="false")
    paths_obj = Namespace(p="x")

    monkeypatch.setattr(toolbox, "config", lambda: cfg)
    monkeypatch.setattr(toolbox, "paths", lambda _cfg: paths_obj)

    args = Namespace(mode="download_files", verbose=False)
    context = toolbox.init_context(args)

    assert context.args is args
    assert context.config is cfg
    assert context.path is paths_obj
    assert context.dry_run is False
    assert capsys.readouterr().out == ""


def test_init_context_sets_dry_run_true_and_prints_banner(monkeypatch, capsys):
    """The `init_context` function should set `dry_run` True for any `config.dry_run`
    value other than literal 'false' and print the dry-run banner.
    """
    cfg = Namespace(dry_run="true")

    monkeypatch.setattr(toolbox, "config", lambda: cfg)
    monkeypatch.setattr(toolbox, "paths", lambda _cfg: Namespace())

    args = Namespace(mode="download_files", verbose=False)
    context = toolbox.init_context(args)

    assert context.dry_run is True
    out = capsys.readouterr().out
    assert "---- Dry Run (no remote changes) ----" in out


def test_init_context_reuses_existing_namespace(monkeypatch):
    """The `init_context` function should update and return the provided context
    Namespace rather than allocating a new one.
    """
    cfg = Namespace(dry_run="false")

    monkeypatch.setattr(toolbox, "config", lambda: cfg)
    monkeypatch.setattr(toolbox, "paths", lambda _cfg: Namespace())

    args = Namespace(mode="download_files", verbose=False)
    existing = Namespace(existing=1)

    ctx = toolbox.init_context(args, context=existing)
    assert ctx is existing
    assert ctx.existing == 1
    assert ctx.args is args
    assert ctx.config is cfg


def test_init_clients_configures_session_clients_and_url_ok(monkeypatch):
    """The `init_clients` function should mount FileAdapter for file://,
    set User-Agent, attach client helpers, and expose the `url_ok` function
    that accepts 200/206 and rejects other status codes.
    """
    class FakeResp:
        def __init__(self, code):
            self.status_code = code

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeSession:
        def __init__(self):
            self.mounted = []
            self.headers = {}
            self.head_calls = []

        def mount(self, prefix, adapter):
            self.mounted.append((prefix, adapter))

        def head(self, url, allow_redirects=True, timeout=30):
            self.head_calls.append((url, allow_redirects, timeout))
            if "partial" in url:
                return FakeResp(206)
            return FakeResp(200 if "ok" in url else 404)

    class FakeAdapter:
        pass

    monkeypatch.setattr(toolbox, "FileAdapter", FakeAdapter)

    api_obj, admin_obj, dl_obj = object(), object(), object()
    created = {"api": None, "admin": None, "dl": None}

    monkeypatch.setattr(toolbox, "APIClient", lambda ctx: created.__setitem__("api", ctx) or api_obj)
    monkeypatch.setattr(toolbox, "AdminClient", lambda ctx: created.__setitem__("admin", ctx) or admin_obj)
    monkeypatch.setattr(toolbox, "Downloader", lambda ctx: created.__setitem__("dl", ctx) or dl_obj)

    ctx = Namespace()
    sess = FakeSession()
    out = toolbox.init_clients(ctx, session=sess)

    assert out is ctx
    assert ctx.session is sess

    # Mounts file:// adapter and sets UA header
    assert any(prefix == "file://" for prefix, _a in sess.mounted)
    assert isinstance([a for p, a in sess.mounted if p == "file://"][0], FakeAdapter)
    assert sess.headers.get("User-Agent") == toolbox.useragent

    # Attaches client helpers (constructor internals are not tested)
    assert ctx.api_client is api_obj and created["api"] is ctx
    assert ctx.admin_client is admin_obj and created["admin"] is ctx
    assert ctx.downloader is dl_obj and created["dl"] is ctx

    # url_ok(): 200/206 => True, other codes => False; head args are fixed
    assert ctx.url_ok("http://ok.example") is True
    assert ctx.url_ok("http://partial.example") is True
    assert ctx.url_ok("http://nope.example") is False
    assert sess.head_calls[0] == ("http://ok.example", True, 30)


def test_init_clients_creates_session_when_none(monkeypatch):
    """The `init_clients` function should create a `requests.Session` when session
    is None and store it on `context.session`.
    """
    class FakeSession:
        def __init__(self):
            self.mounted = []
            self.headers = {}

        def mount(self, prefix, adapter):
            self.mounted.append((prefix, adapter))

        def head(self, url, allow_redirects=True, timeout=30):
            class R:
                status_code = 200

                def __enter__(self): return self
                def __exit__(self, *a): return False
            return R()

    monkeypatch.setattr(toolbox.requests, "Session", FakeSession)
    monkeypatch.setattr(toolbox, "FileAdapter", lambda: object())
    monkeypatch.setattr(toolbox, "APIClient", lambda ctx: object())
    monkeypatch.setattr(toolbox, "AdminClient", lambda ctx: object())
    monkeypatch.setattr(toolbox, "Downloader", lambda ctx: object())

    ctx = Namespace()
    toolbox.init_clients(ctx, session=None)

    assert isinstance(ctx.session, FakeSession)


def test_main_parses_initializes_and_dispatches_mode(monkeypatch):
    """The `main` function should parse argv, initialize context and clients,
    and dispatch the selected mode exactly once using modes()[args.mode].
    """
    called = {"parse": 0, "ctx": 0, "init": 0, "mode": 0}
    args_obj = Namespace(mode="download_files", verbose=False)

    def fake_parse(argv):
        called["parse"] += 1
        assert argv == ["download_files"]
        return args_obj

    ctx_obj = Namespace()

    def fake_init_context(args):
        called["ctx"] += 1
        assert args is args_obj
        return ctx_obj

    class FakeSess:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_init_clients(context, session):
        called["init"] += 1
        assert context is ctx_obj
        assert session is not None
        context.inited = True
        return context

    def mode_fn(context):
        assert context is ctx_obj
        assert getattr(context, "inited", False) is True
        called["mode"] += 1

    monkeypatch.setattr(toolbox, "parse_args", fake_parse)
    monkeypatch.setattr(toolbox, "init_context", fake_init_context)
    monkeypatch.setattr(toolbox.requests, "Session", lambda: FakeSess())
    monkeypatch.setattr(toolbox, "init_clients", fake_init_clients)
    monkeypatch.setattr(toolbox, "modes", lambda: {"download_files": mode_fn})

    toolbox.main(["download_files"])
    assert called == {"parse": 1, "ctx": 1, "init": 1, "mode": 1}


# -------------------------
# CLI modes
# -------------------------

def test_mode_download_files_auth_gate(context_tmp, capsys):
    """The `mode_download_files` function should refuse to proceed when api
    authentication/health checks fail and emit a user-visible message.
    """

    class FakeApi:
        def check_api_auth(self): return False

    context_tmp.api_client = FakeApi()

    toolbox.mode_download_files(context_tmp)

    out = capsys.readouterr().out
    assert "API is inaccessible" in out


def test_mode_download_files_happy_path_calls_pipeline(context_tmp, monkeypatch):
    """The `mode_download_files` function should orchestrate export/api ingestion,
    file extraction, downloads, and summarization in the expected order when the
    api is accessible.
    """

    class FakeApi:
        def check_api_auth(self): return True

    context_tmp.api_client = FakeApi()

    calls = []
    monkeypatch.setattr(toolbox, "log", lambda args: calls.append("log"))
    monkeypatch.setattr(toolbox, "posts_from_export", lambda args: calls.append("export") or {"1": {"date": "0", "image_urls": []}})
    monkeypatch.setattr(toolbox, "posts_from_api", lambda args, posts: calls.append("api") or posts)
    monkeypatch.setattr(toolbox, "files_from_posts", lambda args, posts: calls.append("files_from_posts") or {})
    monkeypatch.setattr(toolbox, "download_files", lambda args, files: calls.append("download_files") or files)
    monkeypatch.setattr(toolbox, "summarize", lambda args, files: calls.append("summarize"))

    toolbox.mode_download_files(context_tmp)

    assert calls == ["log", "export", "api", "files_from_posts", "download_files", "summarize"]


def test_mode_download_links_overrides_config_and_runs(context_tmp, monkeypatch):
    """The `mode_download_links` function should force link-only semantics
    (no thumbs, skip_days=0) and run the export/api/files/summarize pipeline.
    """

    class FakeApi:
        def check_api_auth(self): return True

    context_tmp.api_client = FakeApi()
    context_tmp.config.old_url_thumb = "https://old.example.com/thumb/"
    context_tmp.config.skip_days = "99"

    called = []
    monkeypatch.setattr(toolbox, "log", lambda args: called.append("log"))
    monkeypatch.setattr(toolbox, "posts_from_export", lambda args: called.append("export") or {})
    monkeypatch.setattr(toolbox, "posts_from_api", lambda args, posts: called.append("api") or posts)
    monkeypatch.setattr(toolbox, "files_from_posts", lambda args, posts: called.append("files_from_posts") or {})
    monkeypatch.setattr(toolbox, "summarize", lambda args, files: called.append("summarize"))

    toolbox.mode_download_links(context_tmp)

    assert context_tmp.config.old_url_thumb is None
    assert int(context_tmp.config.skip_days) == 0
    assert called == ["log", "export", "api", "files_from_posts", "summarize"]


def test_mode_update_posts_and_delete_files_auth_gate(context_tmp, capsys):
    """The `mode_update_posts` and `mode_delete_files` functions should enforce
    their respective auth checks and emit a failure message instead of performing
    destructive operations.
    """

    class BadApi:
        def check_api_auth(self): return False

    context_tmp.api_client = BadApi()

    toolbox.mode_update_posts(context_tmp)
    assert "API is inaccessible" in capsys.readouterr().out

    class BadAdmin:
        def check_admin_auth(self): return False

    context_tmp.admin_client = BadAdmin()

    toolbox.mode_delete_files(context_tmp)
    assert "Admin UI is inaccessible" in capsys.readouterr().out


def test_mode_update_legacy_links_calls_expected(context_tmp, monkeypatch):
    """The `mode_update_legacy_links` function should run the legacy-link update
    pipeline (export->files_from_export->summarize->update_posts) with legacy=True
    and reset skip_days to 0.
    """

    class GoodApi:
        def check_api_auth(self): return True

    context_tmp.api_client = GoodApi()

    context_tmp.config.skip_days = "5"

    calls = []
    monkeypatch.setattr(toolbox, "posts_from_export", lambda args, legacy=False: calls.append(("export", legacy)) or {})
    monkeypatch.setattr(toolbox, "files_from_export", lambda args, posts: calls.append("files_from_export") or {})
    monkeypatch.setattr(toolbox, "summarize", lambda args, files, legacy=False: calls.append(("summarize", legacy)))
    monkeypatch.setattr(toolbox, "update_posts", lambda args, legacy=False: calls.append(("update_posts", legacy)))

    toolbox.mode_update_legacy_links(context_tmp)

    assert int(context_tmp.config.skip_days) == 0
    assert calls == [("export", True), "files_from_export", ("summarize", True), ("update_posts", True)]
