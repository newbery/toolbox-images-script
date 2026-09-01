from toolbox import commands


def test_mode_download_files_auth_gate(context_tmp, capsys):
    """The `mode_download_files` function should refuse to proceed when api
    authentication/health checks fail and emit a user-visible message.
    """

    class FakeApi:
        def check_api_auth(self):
            return False

    context_tmp.api_client = FakeApi()

    commands.mode_download_files(context_tmp)
    out = capsys.readouterr().out
    assert "API is inaccessible" in out


def test_mode_download_files_happy_path_calls_pipeline(context_tmp, monkeypatch):
    """The `mode_download_files` function should orchestrate export/api ingestion,
    file extraction, downloads, and summarization in the expected order when the
    api is accessible.
    """

    class FakeApi:
        def check_api_auth(self):
            return True

    context_tmp.api_client = FakeApi()
    calls = []
    monkeypatch.setattr(commands, "log", lambda args: calls.append("log"))
    monkeypatch.setattr(
        commands,
        "posts_from_export",
        lambda args: calls.append("export") or {"1": {"date": "0", "image_urls": []}},
    )
    monkeypatch.setattr(
        commands, "posts_from_api", lambda args, posts: calls.append("api") or posts
    )
    monkeypatch.setattr(
        commands,
        "files_from_posts",
        lambda args, posts: calls.append("files_from_posts") or {},
    )
    monkeypatch.setattr(
        commands,
        "download_files",
        lambda args, files: calls.append("download_files") or files,
    )
    monkeypatch.setattr(commands, "summarize", lambda args, files: calls.append("summarize"))
    commands.mode_download_files(context_tmp)

    assert calls == ["log", "export", "api", "files_from_posts", "download_files", "summarize"]


def test_mode_download_links_overrides_config_and_runs(context_tmp, monkeypatch):
    """The `mode_download_links` function should force link-only semantics
    (no thumbs, skip_days=0) and run the export/api/files/summarize pipeline.
    """

    class FakeApi:
        def check_api_auth(self):
            return True

    context_tmp.api_client = FakeApi()
    context_tmp.config.old_url_thumb = "https://old.example.com/thumb/"
    context_tmp.config.skip_days = 99
    called = []
    monkeypatch.setattr(commands, "log", lambda args: called.append("log"))
    monkeypatch.setattr(commands, "posts_from_export", lambda args: called.append("export") or {})
    monkeypatch.setattr(
        commands, "posts_from_api", lambda args, posts: called.append("api") or posts
    )
    monkeypatch.setattr(
        commands,
        "files_from_posts",
        lambda args, posts: called.append("files_from_posts") or {},
    )
    monkeypatch.setattr(commands, "summarize", lambda args, files: called.append("summarize"))
    commands.mode_download_links(context_tmp)

    assert context_tmp.config.old_url_thumb is None
    assert context_tmp.config.skip_days == 0
    assert called == ["log", "export", "api", "files_from_posts", "summarize"]


def test_mode_update_posts_and_delete_files_auth_gate(context_tmp, capsys):
    """The `mode_update_posts` and `mode_delete_files` functions should enforce
    their respective auth checks and emit a failure message instead of performing
    destructive operations.
    """

    class BadApi:
        def check_api_auth(self):
            return False

    context_tmp.api_client = BadApi()

    commands.mode_update_posts(context_tmp)
    assert "API is inaccessible" in capsys.readouterr().out

    class BadAdmin:
        def check_admin_auth(self):
            return False

    context_tmp.admin_client = BadAdmin()

    commands.mode_delete_files(context_tmp)
    assert "Admin UI is inaccessible" in capsys.readouterr().out


def test_mode_update_legacy_links_calls_expected(context_tmp, monkeypatch):
    """The `mode_update_legacy_links` function should run the legacy-link update
    pipeline (export->files_from_export->summarize->update_posts) with legacy=True
    and reset skip_days to 0.
    """

    class GoodApi:
        def check_api_auth(self):
            return True

    context_tmp.api_client = GoodApi()

    context_tmp.config.skip_days = 5
    calls = []
    monkeypatch.setattr(
        commands,
        "posts_from_export",
        lambda args, legacy=False: calls.append(("export", legacy)) or {},
    )
    monkeypatch.setattr(
        commands,
        "files_from_export",
        lambda args, posts: calls.append("files_from_export") or {},
    )
    monkeypatch.setattr(
        commands,
        "summarize",
        lambda args, files, legacy=False: calls.append(("summarize", legacy)),
    )
    monkeypatch.setattr(
        commands,
        "update_posts",
        lambda args, legacy=False: calls.append(("update_posts", legacy)),
    )
    commands.mode_update_legacy_links(context_tmp)

    assert context_tmp.config.skip_days == 0
    assert calls == [
        ("export", True),
        "files_from_export",
        ("summarize", True),
        ("update_posts", True),
    ]
