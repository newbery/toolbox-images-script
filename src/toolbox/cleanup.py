"""
Verify migrated URLs and remove obsolete Website Toolbox files.
"""

import json
import shutil
import tempfile
import time
from ast import literal_eval
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote

from plumbum.cmd import cut, grep

from .context import Context, alive_bar
from .io import batched, confirm, linecount, read_csv
from .models import FileMap, FileResult, ForumFile
from .urls import get_new_url_func


def delete_files(context: Context) -> None:
    """Delete Toolbox images given by set of fileids.

    There is no API endpoint for deleting files so this instead uses the
    Admin UI by simulating the Delete Files form submission.

    This of course won't work with files not hosted by Toolbox so it will
    throw an error if an attempt to made to do that.
    """
    client = context.admin_client
    deletes_path = context.path.fileids_to_delete
    fileids_to_delete = json.loads(deletes_path.read_text())

    if context.dry_run:
        print("---- Dry Run: would delete the following fileids (no changes made) ----")
        if not fileids_to_delete:
            print("(none)")
        else:
            for fid in fileids_to_delete:
                print(" ", fid)
        return

    if not fileids_to_delete:
        print("Delete files: no fileids listed; nothing to do.")
        return

    if not context.args.yes:
        preview = ", ".join(str(x) for x in fileids_to_delete[:10])
        more = "" if len(fileids_to_delete) <= 10 else f"... (+{len(fileids_to_delete) - 10} more)"
        print(f"About to permanently delete {len(fileids_to_delete)} files from Toolbox.")
        print(f"First 10 fileids: {preview} {more}")

    # A final interactive confirmation helps avoid catastrophic deletes.
    if not confirm(context, "Type DELETE to confirm: ", "DELETE"):
        return

    successes: list[str] = []
    count = len(fileids_to_delete)

    try:
        with alive_bar(count, title="Delete files") as bar:
            for fileids in batched(fileids_to_delete, 100):
                if not fileids:
                    continue
                client.delete_files(fileids)
                successes.extend(fileids)
                bar(len(fileids))
                time.sleep(1.5)
    finally:
        print(f"Successfully deleted: {successes}")

    print(f"Delete files: {count} deleted")


def check_new_urls(context: Context, files: FileMap) -> bool:
    """Check new urls for file/image urls given by posts in `posts.csv` output
    from last `output_results` run. If any are inaccessible then return False.

    This is checked before 'update_posts'.
    """
    dry_run = context.dry_run
    old_prefix = context.config.old_url
    thumb_prefix = context.config.old_url_thumb
    new_prefix = context.config.new_url
    posts_path = context.path.posts
    url_ok = context.url_ok

    # Set this to False to generate a list of failing urls.
    # Make this an environment setting?
    stop_fast = False

    # The proxy we're using throttles at 2500 req per 10 min.
    # Make this sleep interval an environment setting?
    sleep = 0.001 if dry_run else 0.25

    # If new_url is a local path then generate a local 'file://' url.
    # This only works in DRY_RUN since real post updates need public urls.
    if dry_run and not new_prefix.lower().startswith(("https://", "http://")):
        new_prefix = f"file://{Path(new_prefix).resolve()!s}/"

    new_url_func = get_new_url_func(old_prefix, thumb_prefix, new_prefix)

    # Confirm all old urls are accessible at new location except
    # for those files that were skipped or failed during download.
    seen = set()
    images_errors = set()
    count = max(0, linecount(posts_path) - 1)
    with alive_bar(count, title="Check new urls") as bar:
        for row in read_csv(posts_path):
            for url in literal_eval(row["image_urls"]):
                file = files.get(url)
                result = file.result if file else FileResult.default
                if url in seen or result in (FileResult.skipped, FileResult.error):
                    continue
                seen.add(url)
                new_url = file.new_url if file and file.new_url else new_url_func(url)
                if not url_ok(new_url):
                    images_errors.add(new_url)
                    if stop_fast:
                        raise Exception("Image not found:", new_url)
                time.sleep(sleep)
            bar()

    if images_errors:
        if stop_fast:
            # breakpoint()
            raise Exception
        print("Check new urls: !!! Errors attempting to access the following images:")
        for url in images_errors:
            print(" ", url)
    else:
        print("Check new urls: Passed; All images are accessible at new urls")

    return not images_errors


def check_urls_in_old_folder(context: Context) -> None:
    """This function is just a helpful diagnostic to confirm that all images
    in the '{context.path.download_dir}/_old_/' folder can be found in the new
    image host location.

    Assumption: the relative paths under '_old_/' already match the path portion expected
    under the new host prefix (context.config.new_url).

    Any that are not found are copied to '{context.path.download_dir}/_notfound_/'
    so they can be inspected manually.
    """
    download_dir = Path(context.path.download_dir)
    old_dir = download_dir / "_old_"
    notfound_dir = download_dir / "_notfound_"
    url_ok = context.url_ok

    if not old_dir.exists():
        print(f"Check urls in old folder: '{old_dir}' not found; nothing to do")
        return

    # The proxy we're using throttles at 2500 req per 10 min.
    # Make this sleep interval an environment setting?
    sleep = 0.25

    # Normalize new_url prefix
    new_prefix = str(context.config.new_url)
    if not new_prefix.endswith("/"):
        new_prefix += "/"

    def double_quote(path: str) -> str:
        return quote(quote(path))

    # If new_prefix is a proxy URL with query params, filenames may need
    # to be quoted twice to protect special characters through the proxy.
    fixpath = double_quote if "?" in new_prefix else (lambda x: x)

    def iter_old_files() -> Iterator[tuple[Path, str]]:
        for p in old_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(old_dir).as_posix()

            # Skip hidden/housekeeping paths under _old_
            if rel.startswith(".") or "/." in rel:
                continue
            if rel.split("/", 1)[0].startswith("_"):
                continue

            yield p, rel

    # Pre-count without holding all paths in memory
    total = sum(1 for _ in iter_old_files())
    if total == 0:
        print(f"Check urls in old folder: No files under {old_dir}")
        return

    missing = 0
    checked = 0
    first_few_missing: list[str] = []

    with alive_bar(total, title="Check old downloads at new host") as bar:
        for src_path, rel in iter_old_files():
            new_url = new_prefix + fixpath(rel)
            checked += 1

            if not url_ok(new_url):
                missing += 1
                dst = notfound_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst)
                if len(first_few_missing) < 20:
                    first_few_missing.append(new_url)

                breakpoint()

            time.sleep(sleep)
            bar()

    if missing:
        print(f"Check urls in old folder: {missing}/{checked} missing; copied to: {notfound_dir}")
        if first_few_missing:
            print("First missing urls:")
            for url in first_few_missing:
                print(" ", url)
    else:
        print(f"Check urls in old folder: Passed; {checked} files found at new host")


def grep_urls_in_file(updates_path: Path, urls: list[str]) -> str:
    """Given a CSV file `updates_path` and a list of URLs, return the matching
    post IDs (first CSV field) for rows that contain any of the URLs.

    Equivalent intent to the original:
        result = (grep["-E", _urls, updates_path] | cut["-d,", "-f1"])(retcode=None)

    but uses fixed-string grep (no regex interpretation), via:
        grep -F -f <patterns_file> updates.csv | cut -d, -f1
    """
    # Drop empties and de-dup
    seen: set[str] = set()
    patterns = [u for u in urls if u and not (u in seen or seen.add(u))]
    if not patterns:
        return ""

    pattern_path: Path | None = None
    try:
        # Write patterns one-per-line for grep -f
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            for u in patterns:
                tf.write(u)
                tf.write("\n")
            pattern_path = Path(tf.name)

        # grep -F: fixed strings, -f: read patterns from file
        # Pipe to cut to extract first CSV column (post id)
        # retcode=None allows grep exit 1 (no matches) without raising
        return (grep["-F", "-f", str(pattern_path), str(updates_path)] | cut["-d,", "-f1"])(
            retcode=None
        )

    finally:
        if pattern_path is not None:
            try:
                pattern_path.unlink()
            except FileNotFoundError:
                pass


def check_old_urls(
    context: Context, files_to_check: Iterable[ForumFile], legacy: bool = False
) -> bool:
    """Check if any old_urls are still found in updated posts (and in posts
    not updated).

    1) Search 'updates.csv' for any 'url', 'url_thumb', or 'url_file'.
    2) Search a subset of 'posts_from_export.csv' and 'posts_from_api.csv'
    that includes only the non-updated posts.
    3) If any matches are found, print out the list and return False, otherwise
    return True.

    This is checked after 'update_posts'
    """
    updates_path = Path(context.path.updates)
    from_export_path = context.path.posts_from_export
    from_api_path = context.path.posts_from_api
    posts_paths = [from_export_path] if legacy else [from_export_path, from_api_path]

    urls = set()
    fileids = set()
    for f in files_to_check:
        urls.update([f.url, f.url_thumb])
        fileids.update([rf"={f.fileid}", rf"/{f.fileid}/"])
    urls.discard("")

    found_in_updated = []
    found_in_nonupdated = []
    count = len(urls) + len(fileids)

    with alive_bar(count, title="Check old urls") as bar:
        for batch in batched(urls, 100):
            result = grep_urls_in_file(updates_path, batch)
            found_in_updated += result.split()
            bar(len(batch))

        for batch in batched(fileids, 10):
            fileids_ = "|".join(batch)
            result = (grep["-Eh", fileids_, *posts_paths] | cut["-d,", "-f1"])(retcode=None)
            found_in_nonupdated += result.split()
            bar(len(batch))
        found_in_nonupdated = set(found_in_nonupdated) - set(found_in_updated)

    if found_in_updated:
        print("Check old urls: !!! Old urls found in these updated posts:")
        for pid in found_in_updated:
            print(f"  {pid}")

    if found_in_nonupdated:
        print("Check old urls: !!! Old fileids found in these non-updated posts:")
        for pid in sorted(found_in_nonupdated):
            print(f"  {pid}")

    return not bool(found_in_updated or found_in_nonupdated)
