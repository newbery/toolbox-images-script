import json

from toolbox import cleanup, models


def test_check_new_urls_respects_skip_and_generates_file_scheme(
    ctx, tmp_path, monkeypatch, write_csv
):
    """The `check_new_urls` function should check only non-skipped files and,
    in test-run mode with a local new_url root, generate file:// urls for
    validation via `url_ok`.
    """
    # Use local directory as "new_url" and dry-run=True to enable file:// prefix
    new_root = tmp_path / "new"
    new_root.mkdir()
    ctx.config.new_url = str(new_root)  # not http(s)
    ctx.dry_run = True

    # posts.csv with two urls, one skipped, one checked
    write_csv(
        ctx.path.posts,
        ["pid", "date", "image_urls", "message"],
        [
            ["1", "0", "['https://old.example.com/1.jpg', 'https://old.example.com/2.jpg']", "x"],
        ],
    )
    files = {
        "https://old.example.com/1.jpg": models.ForumFile(
            fileid="1",
            url="https://old.example.com/1.jpg",
            result=models.FileResult.skipped,
        ),
        "https://old.example.com/2.jpg": models.ForumFile(
            fileid="2",
            url="https://old.example.com/2.jpg",
            result=models.FileResult.downloaded,
        ),
    }

    def url_ok(url: str) -> bool:
        # It should be a file:// url pointing into new_root
        assert url.startswith("file://")
        return True

    ctx.url_ok = url_ok

    assert cleanup.check_new_urls(ctx, files) is True


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
    )
    out = cleanup.grep_urls_in_file(updates, ["https://b.example.com/y.jpg", ""])
    assert out.split() == ["3"]


def test_check_old_urls_detects_in_updated_or_nonupdated(ctx, tmp_path, write_csv):
    """The `check_old_urls` function should return False when legacy urls
    (or legacy file references) still appear in either updated content or in
    posts that were never updated.
    """
    # Create updates.csv (updated posts content)
    write_csv(
        ctx.path.updates,
        ["pid", "result", "content"],
        [
            ["10", "success", "contains https://old.example.com/123/a.jpg"],
            ["11", "success", "ok"],
        ],
    )
    # Create non-updated posts csvs that still contain a fileid pattern
    write_csv(
        ctx.path.posts_from_export,
        ["pid", "date", "image_urls", "message"],
        [
            ["20", "0", "[]", "legacy =123 somewhere"],
        ],
    )
    write_csv(
        ctx.path.posts_from_api,
        ["pid", "date", "image_urls", "message"],
        [
            ["21", "0", "[]", "nope"],
        ],
    )
    files_to_check = [
        models.ForumFile(fileid="123", url="https://old.example.com/123/a.jpg"),
    ]

    ok = cleanup.check_old_urls(ctx, files_to_check, legacy=False)
    assert ok is False


def test_check_old_urls_detects_url_file_in_updated_post(ctx, write_csv):
    """The final check should detect surviving /file?id=... references."""
    write_csv(
        ctx.path.updates,
        ["pid", "result", "content"],
        [["10", "success", "contains /file?id=123"]],
    )
    write_csv(
        ctx.path.posts_from_export,
        ["pid", "date", "image_urls", "message"],
        [["20", "0", "[]", "no legacy reference"]],
    )
    write_csv(
        ctx.path.posts_from_api,
        ["pid", "date", "image_urls", "message"],
        [["21", "0", "[]", "no legacy reference"]],
    )
    files_to_check = [
        models.ForumFile(
            fileid="123",
            url="https://old.example.com/123/a.jpg",
            url_file="/file?id=123",
        ),
    ]

    assert cleanup.check_old_urls(ctx, files_to_check, legacy=False) is False


def test_check_old_urls_escapes_fileids_used_as_regex(ctx, write_csv):
    """Regex metacharacters in file IDs should be matched literally."""
    write_csv(
        ctx.path.updates,
        ["pid", "result", "content"],
        [["10", "success", "no legacy reference"]],
    )
    write_csv(
        ctx.path.posts_from_export,
        ["pid", "date", "image_urls", "message"],
        [["20", "0", "[]", "similar but different =12x34 reference"]],
    )
    write_csv(
        ctx.path.posts_from_api,
        ["pid", "date", "image_urls", "message"],
        [["21", "0", "[]", "no legacy reference"]],
    )
    files_to_check = [
        models.ForumFile(
            fileid="12.34",
            url="https://old.example.com/12.34/a.jpg",
        ),
    ]

    assert cleanup.check_old_urls(ctx, files_to_check, legacy=False) is True


def test_delete_files_batches_and_calls_client(ctx, monkeypatch):
    """The `delete_files` function should load `fileids_to_delete.json` and invoke
    the admin client's `delete_files` in batches of 100 fileids.
    """
    ctx.path.fileids_to_delete.write_text(
        json.dumps([str(i) for i in range(1, 205)])
    )
    monkeypatch.setattr(cleanup.time, "sleep", lambda *_a, **_k: None)

    calls = []

    class FakeAdmin:
        def check_admin_auth(self):
            return True

        def delete_files(self, fileids):
            calls.append(list(fileids))

    ctx.admin_client = FakeAdmin()

    # Run in apply mode so the admin client is invoked.
    ctx.dry_run = False
    ctx.args = models.CliArgs(mode="delete_files", apply=True, yes=True)

    cleanup.delete_files(ctx)
    # Should batch at 100
    assert len(calls) == 3
    assert len(calls[0]) == 100
    assert len(calls[1]) == 100
    assert len(calls[2]) == 4
