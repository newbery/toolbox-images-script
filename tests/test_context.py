from pathlib import Path

from toolbox import context, models


def test_config_merges_dotenv_and_env(monkeypatch):
    """Config loading should merge sources and coerce values to typed fields."""

    def fake_dotenv_values(filename):
        if filename == ".env":
            return {
                "API_URL": "https://api.example.com",
                "ADMIN_URL": "https://admin.example.com",
                "EXPORT_DIR": "csv",
                "DOWNLOAD_DIR": "downloads",
                "OUTPUT_DIR": "output",
                "OLD_URL": "https://old.example.com/",
                "OLD_URL_THUMB": "",
                "NEW_URL": "https://new.example.com/",
                "SKIP_DAYS": "30",
                "TEST_POST_ID": "",
                "DRY_RUN": "true",
                "API_USERNAME": "from-env-file",
            }
        if filename == ".env.secrets":
            return {
                "API_KEY": "secret-key",
                "API_USERNAME": "secret-user",
                "ADMIN_COOKIE": "secret-cookie",
            }
        return {}

    monkeypatch.setattr(context, "dotenv_values", fake_dotenv_values)
    monkeypatch.setenv("TOOLBOX_SKIP_DAYS", "7")
    monkeypatch.setenv("TOOLBOX_DRY_RUN", "false")
    monkeypatch.setenv("OTHER", "x")

    cfg = context.config()
    assert cfg.api_username == "secret-user"
    assert cfg.api_key == "secret-key"
    assert cfg.skip_days == 7
    assert cfg.dry_run is False
    assert cfg.export_dir == Path("csv")
    assert cfg.old_url_thumb is None
    assert cfg.test_post_id is None
    assert not hasattr(cfg, "other")


def test_paths_builds_expected_paths(tmp_path, config_for):
    """The `paths` function should build the derived filesystem paths
    (exports/downloads/output and expected filenames) from the config directories.
    """
    cfg = config_for(tmp_path)
    paths = context.paths(cfg)
    assert paths.export_dir.name == "export"
    assert paths.posts.name == "posts.csv"
    assert paths.fileids_to_delete.name == "fileids_to_delete.json"


def test_init_context_populates_typed_context_and_config_dry_run_false(
    tmp_path, monkeypatch, capsys, config_for
):
    """The typed context should use the configured dry-run value by default."""
    cfg = config_for(tmp_path, dry_run=False)
    monkeypatch.setattr(context, "config", lambda: cfg)

    args = models.CliArgs(mode="download_files")
    ctx = context.init_context(args)
    assert isinstance(ctx, context.Context)
    assert ctx.args is args
    assert ctx.config is cfg
    assert isinstance(ctx.path, context.Paths)
    assert ctx.dry_run is False
    assert capsys.readouterr().out == ""


def test_init_context_sets_dry_run_true_and_prints_banner(
    tmp_path, monkeypatch, capsys, config_for
):
    """Configured dry-run should produce a dry-run context and banner."""
    cfg = config_for(tmp_path, dry_run=True)
    monkeypatch.setattr(context, "config", lambda: cfg)

    ctx = context.init_context(models.CliArgs(mode="download_files"))

    assert ctx.dry_run is True
    out = capsys.readouterr().out
    assert "---- Dry Run (no remote changes) ----" in out


def test_init_context_cli_apply_overrides_config(tmp_path, monkeypatch, config_for):
    """An explicit --apply option should override configured dry-run mode."""
    cfg = config_for(tmp_path, dry_run=True)
    monkeypatch.setattr(context, "config", lambda: cfg)

    ctx = context.init_context(models.CliArgs(mode="download_files", apply=True))

    assert ctx.dry_run is False


def test_init_clients_configures_session_clients_and_url_ok(ctx, monkeypatch):
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

    monkeypatch.setattr(context, "FileAdapter", FakeAdapter)

    api_obj, admin_obj, dl_obj = object(), object(), object()
    created = {"api": None, "admin": None, "dl": None}
    monkeypatch.setattr(
        context, "APIClient", lambda ctx: created.__setitem__("api", ctx) or api_obj
    )
    monkeypatch.setattr(
        context, "AdminClient", lambda ctx: created.__setitem__("admin", ctx) or admin_obj
    )
    monkeypatch.setattr(context, "Downloader", lambda ctx: created.__setitem__("dl", ctx) or dl_obj)

    sess = FakeSession()
    out = context.init_clients(ctx, session=sess)
    assert out is ctx
    assert ctx.session is sess

    # Mounts file:// adapter and sets UA header
    assert any(prefix == "file://" for prefix, _a in sess.mounted)
    assert isinstance([a for p, a in sess.mounted if p == "file://"][0], FakeAdapter)
    assert sess.headers.get("User-Agent") == context.USER_AGENT

    # Attaches client helpers (constructor internals are not tested)
    assert ctx.api_client is api_obj and created["api"] is ctx
    assert ctx.admin_client is admin_obj and created["admin"] is ctx
    assert ctx.downloader is dl_obj and created["dl"] is ctx

    # url_ok(): 200/206 => True, other codes => False; head args are fixed
    assert ctx.url_ok("http://ok.example") is True
    assert ctx.url_ok("http://partial.example") is True
    assert ctx.url_ok("http://nope.example") is False
    assert sess.head_calls[0] == ("http://ok.example", True, 30)


def test_init_clients_creates_session_when_none(ctx, monkeypatch):
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

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            return R()

    monkeypatch.setattr(context.requests, "Session", FakeSession)
    monkeypatch.setattr(context, "FileAdapter", lambda: object())
    monkeypatch.setattr(context, "APIClient", lambda ctx: object())
    monkeypatch.setattr(context, "AdminClient", lambda ctx: object())
    monkeypatch.setattr(context, "Downloader", lambda ctx: object())

    context.init_clients(ctx, session=None)

    assert isinstance(ctx.session, FakeSession)
