"""
Application configuration, runtime context, and shared process settings.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import requests
from alive_progress import alive_bar as _alive_bar
from alive_progress import config_handler
from dotenv import dotenv_values
from requests_file import FileAdapter

from .clients import AdminClient, APIClient, Downloader
from .models import CliArgs

USER_AGENT = "toolbox-images-script/1.0"

config_handler.set_global(bar="smooth", spinner="classic", receipt=False)
alive_bar = _alive_bar


@dataclass(slots=True)
class Config:
    """Typed runtime configuration loaded from dotenv files and the environment."""

    api_url: str
    admin_url: str
    export_dir: Path
    download_dir: Path
    output_dir: Path
    old_url: str
    old_url_thumb: str | None
    new_url: str
    skip_days: int
    test_post_id: str | None
    dry_run: bool
    api_key: str
    api_username: str
    admin_cookie: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, str | None]) -> Self:
        """Build configuration from merged, lower-case string settings."""

        def value(name: str, default: str = "") -> str:
            raw = values.get(name, default)
            return default if raw is None else raw

        old_url_thumb = value("old_url_thumb") or None
        test_post_id = value("test_post_id") or None

        return cls(
            api_url=value("api_url"),
            admin_url=value("admin_url"),
            export_dir=Path(value("export_dir")),
            download_dir=Path(value("download_dir")),
            output_dir=Path(value("output_dir")),
            old_url=value("old_url"),
            old_url_thumb=old_url_thumb,
            new_url=value("new_url"),
            skip_days=int(value("skip_days", "0")),
            test_post_id=test_post_id,
            dry_run=parse_bool(values.get("dry_run"), default=True),
            api_key=value("api_key"),
            api_username=value("api_username"),
            admin_cookie=value("admin_cookie"),
        )


@dataclass(frozen=True, slots=True)
class Paths:
    """Local input, working, and output paths derived from configuration."""

    export_dir: Path
    download_dir: Path
    output_dir: Path
    posts_from_export: Path
    posts_from_api: Path
    posts: Path
    files: Path
    updates: Path
    fileids_to_delete: Path
    fileids_to_delete_dry_run: Path
    log: Path


@dataclass(slots=True)
class Context:
    """Runtime state shared by the migration pipeline and service clients."""

    args: CliArgs
    config: Config
    path: Paths
    dry_run: bool
    session: requests.Session = field(init=False, repr=False)
    api_client: "APIClient" = field(init=False, repr=False)
    admin_client: "AdminClient" = field(init=False, repr=False)
    downloader: "Downloader" = field(init=False, repr=False)
    url_ok: Callable[[str], bool] = field(init=False, repr=False)


def parse_bool(val: str | bool | None, default: bool = True) -> bool:
    """Parse a permissive boolean string, falling back when no value is supplied."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    value = val.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {val!r}")


def config() -> Config:
    """Collect typed config from dotenv files and TOOLBOX_* environment variables.

    Values from `.env.secrets` override `.env`, and process environment values
    prefixed with `TOOLBOX_` override both files.
    """
    env = {k.lower(): v for k, v in dotenv_values(".env").items()}
    env_secrets = {k.lower(): v for k, v in dotenv_values(".env.secrets").items()}

    prefix = "TOOLBOX_"
    length = len(prefix)
    environ = {k[length:].lower(): v for k, v in os.environ.items() if k.startswith(prefix)}

    return Config.from_mapping({**env, **env_secrets, **environ})


def paths(config: Config) -> Paths:
    """Generate the local and derived paths used by the migration."""
    dirs = (config.export_dir, config.download_dir, config.output_dir)

    # Don't create parents: a typo should not create an unexpected directory tree.
    for path in dirs:
        path.mkdir(exist_ok=True)

    output_dir = config.output_dir
    return Paths(
        export_dir=config.export_dir,
        download_dir=config.download_dir,
        output_dir=output_dir,
        posts_from_export=output_dir / "posts_from_export.csv",
        posts_from_api=output_dir / "posts_from_api.csv",
        posts=output_dir / "posts.csv",
        files=output_dir / "files.csv",
        updates=output_dir / "updates.csv",
        fileids_to_delete=output_dir / "fileids_to_delete.json",
        fileids_to_delete_dry_run=output_dir / "fileids_to_delete.dry_run.json",
        log=output_dir / "log.txt",
    )


def init_context(args: CliArgs) -> Context:
    """Initialize the typed runtime context."""
    cfg = config()
    path = paths(cfg)

    # Precedence: explicit CLI flags > config/env (default: dry-run).
    if args.apply:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        dry_run = cfg.dry_run

    if dry_run:
        print("---- Dry Run (no remote changes) ----")

    return Context(args=args, config=cfg, path=path, dry_run=dry_run)


def init_clients(context: Context, session: requests.Session | None = None) -> Context:
    """Initialize service clients and add them to the context."""
    session = session or requests.Session()
    session.mount("file://", FileAdapter())
    session.headers.update({"User-Agent": USER_AGENT})

    context.session = session
    context.api_client = APIClient(context)
    context.admin_client = AdminClient(context)
    context.downloader = Downloader(context)

    def url_ok(url: str) -> bool:
        with session.head(url, allow_redirects=True, timeout=30) as resp:
            return resp.status_code in (200, 206)

    context.url_ok = url_ok
    return context
