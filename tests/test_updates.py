import json

from toolbox import io, models, updates


def test_update_posts_updates_content_and_writes_outputs(context_tmp, monkeypatch, write_csv):
    """The `update_posts` function should rewrite legacy urls in post content to
    the new host, record per-post update results, and emit the set of fileids
    eligible for deletion.
    """
    # Avoid sleeping
    monkeypatch.setattr(updates.time, "sleep", lambda *_args, **_kwargs: None)
    # Prepare posts.csv to update
    msg = (
        "<p>"
        "<img src='https://old.example.com/123/a.jpg'/>"
        "<img src='https://old.example.com/thumb/123/a.jpg'/>"
        "<a href='/file?id=123'>file</a>"
        "</p>"
    )
    write_csv(
        context_tmp.path.posts,
        ["pid", "date", "image_urls", "message"],
        [
            [
                "1",
                "0",
                "['https://old.example.com/123/a.jpg', 'https://old.example.com/thumb/123/a.jpg']",
                msg,
            ],
            # This one should be unchanged (skipped file)
            [
                "2",
                "0",
                "['https://old.example.com/555/a.jpg']",
                "<img src='https://old.example.com/555/a.jpg'/>",
            ],
        ],
    )
    # Prepare files.csv input
    write_csv(
        context_tmp.path.files,
        ["fileid", "pids", "url", "url_thumb", "url_file", "new_url", "result"],
        [
            [
                "123",
                "{'1'}",
                "https://old.example.com/123/a.jpg",
                "https://old.example.com/thumb/123/a.jpg",
                "/file?id=123",
                "",
                str(models.FileResult.downloaded.value),
            ],
            [
                "555",
                "{'2'}",
                "https://old.example.com/555/a.jpg",
                "",
                "",
                "",
                str(models.FileResult.skipped.value),
            ],
        ],
    )
    # Patch check_new_urls/check_old_urls
    monkeypatch.setattr(updates, "check_new_urls", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(updates, "check_old_urls", lambda *_args, **_kwargs: True)

    # Fake API client: should NOT be called in dry-run mode.
    class FakeClient:
        def update_post(self, _pid, _message):
            raise AssertionError("update_post should not be called in dry-run mode")

    context_tmp.api_client = FakeClient()
    updates.update_posts(context_tmp, legacy=False)

    # updates.csv should include a dry-run result with rewritten content for pid 1
    updates_rows = list(io.read_csv(context_tmp.path.updates))
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
