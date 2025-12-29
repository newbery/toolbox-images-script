#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from argparse import Namespace
from ast import literal_eval
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import cache, cached_property, partial
from itertools import chain, islice
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
from alive_progress import alive_bar, config_handler
from bs4 import BeautifulSoup, Comment, MarkupResemblesLocatorWarning
from dotenv import dotenv_values
from plumbum.cmd import cut, grep, wc
from requests_file import FileAdapter

htmlparser = partial(BeautifulSoup, features="html.parser")
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
config_handler.set_global(bar="smooth", spinner="classic", receipt=False)

useragent = "toolbox-images-script/1.0"

DESCRIPTION = """
This script is a utility to help manage a forum hosted on the Website Toolbox
service that is hitting the server space limits. It does this by downloading
Toolbox-hosted images found in post messages and then updating the image links
in these posts to point to a new image host.

Images and files in post 'attachments' are not currently updated with this script.
Support for post attachments may be added in a later version.

Also, images and files in private messages and in avatar images and user galleries
are not managed by this script. Support for these files will probably not be added.
"""

USAGE = """
"""


@cache
def modes() -> dict[str, Callable[[Namespace], None]]:
    """Command line modes"""
    return {
        "download_files": mode_download_files,
        "download_links": mode_download_links,
        "update_posts": mode_update_posts,
        "delete_files": mode_delete_files,
        "update_legacy_links": mode_update_legacy_links,
    }


def main(argv: list | None = None) -> None:
    """Main command line entrypoint"""
    argv = argv or sys.argv[1:]
    args = parse_args(argv)
    context = init_context(args)
    with requests.Session() as session:
        context = init_clients(context, session)
        modes()[args.mode](context)


def parse_args(argv: list) -> Namespace:
    """Parse command line args"""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    
    # Safety controls
    #
    # Default behavior is "dry-run" unless explicitly overridden, either by:
    #   * the config env var TOOLBOX_DRY_RUN, or
    #   * the explicit CLI flags below.
    #
    # Destructive operations are additionally guarded at the service-client
    # layer (see BaseClient._require_apply()).

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform remote updates/deletes. Without this, the script runs in dry-run mode.",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run (no remote updates/deletes), regardless of config.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmations for destructive actions (only relevant with --apply).",
    )
    
    parser.add_argument("mode", choices=list(modes()))
    return parser.parse_args(argv)


def init_context(args: Namespace, context: Namespace | None = None) -> Namespace:
    """Initialize context"""
    context = context or Namespace()
    context.args = args
    context.config = config()
    context.path = paths(context.config)
    
    # Determine dry-run mode.
    # Precedence: explicit CLI flags > config/env (default: dry-run).
    if getattr(args, "apply", False):
        context.dry_run = False
    elif getattr(args, "dry_run", False):
        context.dry_run = True
    else:
        context.dry_run = parse_bool(getattr(context.config, "dry_run", None), default=True)

    if context.dry_run:
        print("---- Dry Run (no remote changes) ----")

    return context


def init_clients(context: Namespace, session: requests.Session | None = None) -> Namespace:
    """Initialize service clients and add to context"""
    session = session or requests.Session()
    session.mount('file://', FileAdapter())
    session.headers.update({"User-Agent": useragent})
    
    context.session = session
    context.api_client = APIClient(context)
    context.admin_client = AdminClient(context)
    context.downloader = Downloader(context)
    
    def url_ok(url: str) -> bool:
        with session.head(url, allow_redirects=True, timeout=30) as resp:
            return resp.status_code in (200, 206)
    
    context.url_ok = url_ok
    return context


def mode_download_files(context: Namespace) -> None:
    """Process posts, download images, and then generate a list of downloaded
    images and a list of posts to update.
    
    If a `posts.csv` content export from the Toolbox site is found then that
    export is processed first. This export from Toolbox is optional but it
    helps reduce the number of API requests in the second step. API requests
    are throttled but they still count against the site views allotment.
    
    After the content export is processed, the Toolbox API is queried and any
    remaining posts are then processed.
    
    Images are then downloaded and a final list of downloaded images and posts
    to be updated is generated.
    """
    
    # Confirm that we have access to api
    if not context.api_client.check_api_auth():
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    # Keep results from the last 10 runs for debugging purposes
    # rotate_output_archive(context)
    log(context)
    
    # Process the data sources
    posts = posts_from_export(context)
    posts = posts_from_api(context, posts)
    files = files_from_posts(context, posts)
    files = download_files(context, files)
    
    # Generate summary
    summarize(context, files)
    
    print("Done")


def mode_download_links(context: Namespace) -> None:
    """Collect links from posts and then generate a list of posts to update.
    
    This is very similar to 'mode_download_files' except no files are downloaded.
    
    In this case, the two config settings, OLD_URL and NEW_URL, are repurposed
    slightly but the effect is largely the same. One difference to note is that
    we do not treat 'thumb' image urls any differently than 'full' image urls.
    That distinction is only important with the original Toolbox-hosted links
    which have a different url construction for the 'thumb' case.
    
    This mode also ignores the SKIP_DAYS config so that all posts with these links
    are updated without a date filter. This might take a while to complete.
    """
    
    # Confirm that we have access to api
    if not context.api_client.check_api_auth():
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    # Override configs
    context.config.old_url_thumb = None
    context.config.skip_days = 0
    
    # Keep results from the last 10 runs for debugging purposes
    # rotate_output_archive(args)
    log(context)
    
    # Process the data sources
    posts = posts_from_export(context)
    posts = posts_from_api(context, posts)
    files = files_from_posts(context, posts)
    
    # Generate summary
    summarize(context, files)
    
    print("Done")


def mode_update_posts(context: Namespace) -> None:
    """Process the `posts.csv` result from the last `download_*` run and update
    the posts with image links updated to point to the new image host.
    
    This update attempts to be cautious about updates by confirming that all
    new image urls are reachable before the update. Otherwise, we assume
    the list of posts to update has already been filtered appropriately by
    the logic in the `download` mode.
    """
    
    # Confirm that we have access to api
    if not context.api_client.check_api_auth():
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    log(context)
    update_posts(context)
    
    print("Done")


def mode_delete_files(context: Namespace) -> None:
    """Process the `posts.csv` result from the last `download` run and update
    the posts with image links updated to point to the new image host.
    
    This mode attempts to be cautious by loading the list of files-to-be-deleted
    from a successful run of 'update_posts'.
    """
    
    # Confirm that we have access to Admin UI
    if not context.admin_client.check_admin_auth():
        print("Admin UI is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    log(context)
    delete_files(context)
    
    print("Done")


def mode_update_legacy_links(context: Namespace) -> None:
    """Clean up legacy urls in a Toolbox forum.
    
    There are two types of legacy urls: "/file?=" and "files.websitetoolbox.com".
    To make migration easier, this mode will find all of these and update them
    to the urls used by the Cloudfront CDN for Toolbox-hosted files.
    
    Legacy links are all old so we'll just parse the content export for the data.
    """
    
    # Confirm that we have access to api
    if not context.api_client.check_api_auth():
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    # Keep results from the last 10 runs for debugging purposes
    # rotate_output_archive(args)
    log(context)
    
    # Override configs
    context.config.old_url_thumb = None
    context.config.skip_days = 0
    
    # Process the data sources
    posts = posts_from_export(context, legacy=True)
    files = files_from_export(context, posts)
    
    # Generate summary
    summarize(context, files, legacy=True)
    
    update_posts(context, legacy=True)

    print("Done")


def posts_from_export(context: Namespace, legacy: bool = False) -> dict:
    """Process the posts listed in the `posts.csv` file from the Toolbox content
    export, collecting a list of image urls in the message text for any images
    hosted by the Toolbox server.
    """
    old_url: str = context.config.old_url
    old_url_thumb: str = context.config.old_url_thumb
    posts_input_path = context.path.export_dir / "posts.csv"
    posts_output_path = context.path.posts_from_export
    
    prefix: str | tuple[str, str] = (old_url, old_url_thumb) if old_url_thumb else old_url
    find_urls = find_legacy_urls if legacy else find_urls_func(prefix)
    
    posts = {}
    count = max(0, linecount(posts_input_path) - 1)
    found = 0
    
    with alive_bar(count, title="From export") as bar:
        
        with posts_output_path.open("w", newline="") as f:
            fieldnames = ["pid", "date", "image_urls", "message"]
            posts_output = csv.writer(f)
            posts_output.writerow(fieldnames)
            
            for row in read_csv(posts_input_path):
                pid = row["pid"]
                date = row["date"]
                message = row["message"]
                image_urls = find_urls(message)
                if image_urls:
                    found += 1
                posts[pid] = {"date": date, "image_urls": image_urls}
                posts_output.writerow([pid, date, image_urls, message])
                bar()
    
    print(f"From export: Processed {len(posts)} posts; Found {found} with image links")
    return posts


def posts_from_api(context: Namespace, posts: dict) -> dict:
    """Process the posts collected via the List Posts API, collecting a list of
    image urls in the message text for any images hosted by the Toolbox server.
    
    The most recent posts are returned first so once we reach a post that we've
    previously processed (via the content export processing), we can skip the rest.
    """
    dry_run = context.dry_run
    client = context.api_client
    old_url = context.config.old_url
    old_url_thumb = context.config.old_url_thumb
    posts_output_path = context.path.posts_from_api
    
    prefix = (old_url, old_url_thumb) if old_url_thumb else old_url
    find_urls = find_urls_func(prefix)
    
    count = 0
    found = 0
    
    with alive_bar(title="From api") as bar:
        
        with posts_output_path.open("w", newline="") as f:
            fieldnames = ["pid", "date", "image_urls", "message"]
            posts_output = csv.writer(f)
            posts_output.writerow(fieldnames)
            
            stop = False
            page_count = 0
            api_requests = client.list_posts()
            for page in api_requests:
                page_count += 1
                for row in page["data"]:
                    pid = str(row["postId"])
                    if pid in posts:
                        # this is processed already so exit early
                        bar()
                        stop = True
                        api_requests.close()
                        break
                    count += 1
                    date = row["postTimestamp"]
                    message = row["message"]
                    image_urls = find_urls(message)
                    if image_urls:
                        found += 1
                    posts[pid] = {"date": date, "image_urls": image_urls}
                    posts_output.writerow([pid, date, image_urls, message])
                    bar()
                    
                if stop or (dry_run and page_count > 3):
                    break
    
    print(f"From api: Processed {count} posts; Found {found} with image links")
    return posts


def files_from_posts(context: Namespace, posts: dict) -> dict:
    """Collect the file info for the urls found in the posts and tag the
    ones that should be excluded.
    
    Files/images referenced in recent posts (given by SKIP_DAYS config) will
    be excluded in the theory that recent posts may still be edited and recent
    posts are most likely to benefit from the Toolbox CDN so moving them is
    probably better postponed.
    """
    test_post_id = context.config.test_post_id
    utc = timezone.utc
    last_date = datetime.now(utc) - timedelta(days=int(context.config.skip_days))
    prefix = context.config.old_url
    prefix_thumb = context.config.old_url_thumb
    
    # This is not 100% reliable. It will be wrong if a non-Toolbox file host
    # provider is also using cloudfront.net. But it's good enough for us.
    toolbox = ".cloudfront.net/" in prefix
    
    # Generate map of files/images to posts and set of files_to_exclude
    files = {}
    files_to_exclude = set()
    for pid, post in posts.items():
        urls = post["image_urls"]        
        
        fileids: list[str] = []
        pairs: list[tuple[str, str]] = []
        
        if toolbox:
            for url in urls:
                if fileid := fileid_from_url(url):
                    fileids.append(fileid)
                    pairs.append((fileid, url))
        else:
            # For the non-toolbox case, let's reuse the url as a fileid
            fileids = urls[:]
            pairs = [(url, url) for url in urls]
        
        if test_post_id and test_post_id != pid:
            files_to_exclude.update(fileids)
        else:
            try:
                ts = int(post["date"])
            except Exception:
                print("Bad date")
                continue
            if datetime.fromtimestamp(ts, utc) > last_date:
                files_to_exclude.update(fileids)
        
        for fileid, url in pairs:
            if fileid in files:
                files[fileid]["pids"].add(pid)
                if not files[fileid]["url_thumb"]:
                    if prefix_thumb and url.startswith(prefix_thumb):
                        files[fileid]["url_thumb"] = url
            else:
                thumb = ""
                if prefix_thumb and url.startswith(prefix_thumb):
                    thumb = url
                    url = url.replace(prefix_thumb, prefix)
                files[fileid] = {
                    "url": url,
                    "url_thumb": thumb,
                    "url_file": f"/file?id={fileid}" if toolbox else "",
                    "path": unquote(url[len(prefix):]),
                    "pids": {pid},
                    "result": "",
                }
    
    # Tag files to be skipped
    for fileid in files_to_exclude:
        files[fileid]["result"] = File.skipped
    
    return files


def files_from_export(context: Namespace, posts: dict) -> dict:
    """This is a special mode for updating legacy links. In this case, we
    need to collect the image data from the export in order to construct
    the updated urls.
    """
    old_url = context.config.old_url
    files_input_path = context.path.export_dir / "attachment.csv"
    
    # Generate map of files/images to posts
    files = {}
    for pid, post in posts.items():
        urls = post["image_urls"]
        
        pairs: list[tuple[str, str]] = []
        for url in urls:
            if fileid := fileid_from_url(url):
                pairs.append((fileid, url))
        
        for fileid, url in pairs:
            if fileid in files:
                files[fileid]["pids"].add(pid)
            else:
                files[fileid] = {
                    "url": url,
                    "url_thumb": "",
                    "url_file": "",
                    "path": "",
                    "pids": {pid},
                    "result": "",
                }
    
    # Generate new_url from 'attachments.csv' export
    count = 0
    filecount = len(files)
    seen = set()
    for row in read_csv(files_input_path):
        fileid = row["fileid"]
        if fileid in files:
            seen.add(fileid)
            files[fileid]["new_url"] = old_url + f"{fileid}/{row['filename']}"
            count += 1
            if count == filecount:
                break
    
    if count != len(files):
        # This should not happen. So stop and figure it out.
        missing = set(files) - seen
        files_ = {k: v for k, v in files.items() if k in missing}  # noqa
        # breakpoint()
        raise Exception
    
    return files


def download_files(context: Namespace, files: dict) -> dict:
    """Download files to be moved to the new image host"""
    dry_run = context.dry_run
    download_dir = context.path.download_dir
    download = context.downloader.download
    
    def download_file(url, path):
        """Download a single file"""
        path_old = download_dir / path
        path_new = download_dir / "_new_" / path
        
        if path_old.exists():
            size = path_old.stat().st_size
        elif path_new.exists():
            size = path_new.stat().st_size
        else:
            size = download(url, path_new)
        return size
    
    skipped = 0
    downloaded = 0
    errors = set()
    
    # Download images, skipping recent images and problem downloads
    size = 0
    count = len(files)
    with alive_bar(count, title="Downloads") as bar:
        
        for fileid, file in files.items():
            if file["result"] == File.skipped:
                skipped += 1
                continue
            
            # Full image/file
            _size = download_file(file["url"], file["path"])
            if _size:
                size += _size
                downloaded += 1
                file["result"] = File.downloaded
            else:
                errors.add(fileid)
                file["result"] = File.error
            
            # Thumb image
            if file["url_thumb"] and file["result"] is File.downloaded:
                _size = download_file(file["url_thumb"], f"thumb/{file['path']}")
                if _size:
                    size += _size
                else:
                    errors.add(fileid)
                    file["result"] = File.error
            
            bar(1)
            
            if dry_run and downloaded > 11:
                skipped = len(files) - len(errors) - downloaded
                break

    if errors:
        print("Downloads: ! Errors (probably old deleted images):")
        for fileid in sorted(errors):
            print(f" {files[fileid]['pids']} {files[fileid]['url']}")
    
    print(f"Skipped {skipped} images/files and downloaded {downloaded} ({friendly_size(size)})")
    
    return files


def summarize(context: Namespace, files: dict, legacy: bool = False) -> None:
    """Generate final output results from the merge of the results from processing
    the content export and the list_posts API.
    """
    from_export_path = context.path.posts_from_export
    from_api_path = context.path.posts_from_api
    posts_output_path = context.path.posts
    files_output_path = context.path.files
    
    # Collect set of all post ids that should be skipped
    posts_to_skip = set()
    for file in files.values():
        if file["result"] is File.skipped:
            posts_to_skip.update(file["pids"])
    
    # Generate reverse map of post_ids to fileids but exclude posts that
    # are in set of posts_to_skip
    posts_to_process = defaultdict(set)
    for fileid, file in files.items():
        for pid in file["pids"]:
            if pid not in posts_to_skip:
                posts_to_process[pid].add(fileid)
    postcount = len(posts_to_process)
    
    # Generate total count of non-skipped or downloaded files
    files_to_process = set()
    for fileids in posts_to_process.values():
        files_to_process.update(fileids)
    filecount = len(files_to_process)
    
    with alive_bar(title="Summarize") as bar:
        
        # Generate final `posts.csv` containing posts to be updated.
        with posts_output_path.open("w", newline="") as f:
            fieldnames = ["pid", "date", "image_urls", "message"]
            posts_output = csv.writer(f)
            posts_output.writerow(fieldnames)
            
            if legacy:
                posts_data = read_csv(from_export_path)
            else:
                posts_data = chain(read_csv(from_export_path), read_csv(from_api_path))
            
            for row in posts_data:
                pid = row["pid"]
                if pid not in posts_to_process:
                    bar()
                    continue
                date = row["date"]
                message = row["message"]
                image_urls = row["image_urls"]
                posts_output.writerow([pid, date, image_urls, message])
                bar()
        
        # Generate `files.csv` with final data about all files found.
        # This includes skipped files since it's useful for diagnosis.
        with files_output_path.open("w", newline="") as f:
            fieldnames = ["fileid", "pids", "url", "url_thumb", "url_file", "new_url", "result"]
            files_output = csv.writer(f)
            files_output.writerow(fieldnames)
            for fileid, file in files.items():
                pids = file["pids"]
                url = file["url"]
                url_thumb = file["url_thumb"]
                url_file = file["url_file"]
                new_url = file.get("new_url", "")  # for legacy link updates
                result = file["result"].value if file["result"] else 0
                files_output.writerow([fileid, pids, url, url_thumb, url_file, new_url, result])
                bar()
    
    print(f"Summarize: {postcount} posts and {filecount} files/images")


def rewrite_post_content(
    *,
    message: str,
    image_urls: list[str],
    files: dict[str, dict],
    legacy: bool,
    new_url_func: Callable[[str], str],
) -> tuple[str, set[str]]:
    """Rewrite a post message and return (new_message, touched_urls).

    `touched_urls` are the original URLs that were replaced or de-linked. This is
    later used to compute safe delete candidates.
    """
    new_message = message
    touched_urls: set[str] = set()

    for url in image_urls:
        try:
            file = files[url]
        except KeyError as e:
            raise KeyError(f"URL referenced in posts.csv not found in files.csv: {url}") from e

        # Updating legacy link
        if legacy:
            if file.get("new_url"):
                new_message = new_message.replace(url, file["new_url"])
                touched_urls.add(url)
            continue

        # De-link missing file (discovered during download)
        if file["result"] is File.error:
            new_message = remove_bad_url(new_message, url)
            touched_urls.add(url)
            continue

        # Skip files that were skipped during download
        if file["result"] is File.skipped:
            continue

        # Replace file url with new url
        new_url = new_url_func(url)
        new_message = new_message.replace(url, new_url)
        touched_urls.add(url)

        # Full image links often accompany thumb images
        if "/thumb/" in url:
            full_url = url.replace("thumb/", "")
            if full_url in files:
                new_full_url = new_url_func(full_url)
                new_message = new_message.replace(full_url, new_full_url)
                touched_urls.add(full_url)

        # Toolbox sometimes uses a special "/file?id=" link
        if file["url_file"]:
            new_full_url = new_url_func(file["url"])
            new_message = new_message.replace(file["url_file"], new_full_url)
            touched_urls.add(file["url"])

    return new_message, touched_urls


def build_update_plan(
    *,
    posts_path: Path,
    files: dict[str, dict],
    legacy: bool,
    new_url_func: Callable[[str], str],
) -> tuple[Path, list[str], set[str]]:
    """Build an on-disk plan of posts that would change.

    Returns:
        (plan_path, sample_pids, urls_touched)
    """
    sample_pids: list[str] = []
    urls_touched: set[str] = set()
    temp = tempfile.NamedTemporaryFile
    count = max(0, linecount(posts_path) - 1)

    with temp(mode="w", encoding="utf-8", newline="\n", delete=False) as plan_file:
        plan_path = Path(plan_file.name)

        with alive_bar(count, title="Plan post updates") as bar:
            for row in read_csv(posts_path):
                pid = row["pid"]
                image_urls = literal_eval(row["image_urls"])

                new_message, touched_urls = rewrite_post_content(
                    message=row["message"],
                    image_urls=image_urls,
                    files=files,
                    legacy=legacy,
                    new_url_func=new_url_func,
                )

                if new_message != row["message"]:
                    if len(sample_pids) < 10:
                        sample_pids.append(pid)
                    urls_touched.update(touched_urls)

                    plan_file.write(
                        json.dumps(
                            {
                                "pid": pid,
                                "content": new_message,
                                "touched_urls": sorted(touched_urls),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                bar()

    return plan_path, sample_pids, urls_touched


def apply_update_plan(*, context: Namespace, plan_path: Path) -> tuple[int, int, set[str], set[str], set[str]]:
    """Apply (or simulate) the planned updates, streaming the plan from disk.

    Returns:
        (posts_updated, posts_would_update, urls_to_delete, urls_to_keep, posts_errors)
    """
    dry_run = context.dry_run
    client = context.api_client
    updates_output_path = context.path.updates
    
    # The plan file is JSONL with no header row.
    count = max(0, linecount(plan_path))
    
    posts_updated = 0
    posts_would_update = 0
    posts_errors: set[str] = set()

    # Track only the urls we actually touched (replaced/de-linked), so we don't
    # accidentally propose deleting skipped/untouched files.
    urls_to_delete: set[str] = set()  # urls safe (or would be safe) to delete
    urls_to_keep: set[str] = set()    # urls not safe to delete

    with updates_output_path.open("w", newline="") as f:
        fieldnames = ["pid", "result", "content"]
        updates_output = csv.writer(f)
        updates_output.writerow(fieldnames)

        with alive_bar(count, title="Update posts") as bar:
            with plan_path.open("r", encoding="utf-8") as plan_in:
                for line in plan_in:
                    item = json.loads(line)
                    pid = item["pid"]
                    new_message = item["content"]
                    touched_urls = set(item.get("touched_urls", []))

                    if dry_run:
                        posts_would_update += 1
                        urls_to_delete.update(touched_urls)
                        result = "dry_run"
                    else:
                        if client.update_post(pid, new_message):
                            urls_to_delete.update(touched_urls)
                            posts_updated += 1
                            result = "success"
                        else:
                            urls_to_keep.update(touched_urls)
                            posts_errors.add(pid)
                            result = "fail"
                        time.sleep(1)  # Throttle API requests

                    updates_output.writerow([pid, result, new_message])
                    bar()

    return posts_updated, posts_would_update, urls_to_delete, urls_to_keep, posts_errors


def update_posts(context: Namespace, legacy: bool = False) -> None:
    """Update posts given by `posts.csv` output from last `download` run"""
    dry_run = context.dry_run
    old_prefix = context.config.old_url
    thumb_prefix = context.config.old_url_thumb
    new_prefix = context.config.new_url
    posts_path = context.path.posts
    files_path = context.path.files
    
    updates_output_path = context.path.updates
    deletes_output_path = context.path.fileids_to_delete
    
    new_url_func = get_new_url_func(old_prefix, thumb_prefix, new_prefix)
    
    # Load file data from output_results, keyed by any urls found in posts
    files = {}
    for row in read_csv(files_path):
        row["pids"] = literal_eval(row["pids"])
        row["result"] = File(int(row["result"]))
        files[row["url"]] = row
        if row["url_thumb"]:
            files[row["url_thumb"]] = row
    
    # If new_urls don't work, abort
    if not check_new_urls(context, files):
        return
 
    # Initialize these in case we get an exception in the 'try' block below
    urls_to_delete: set[str] = set()
    urls_to_keep: set[str] = set()
    posts_errors: set[str] = set()
    posts_updated = 0
    posts_would_update = 0
    posts_errors: set[str] = set()
    plan_path: Path | None = None

    try:
        plan_path, sample_pids, urls_touched = build_update_plan(
            posts_path=posts_path,
            files=files,
            legacy=legacy,
            new_url_func=new_url_func,
        )

        # Final interactive confirmation in APPLY mode (remote changes).
        total_posts = max(0, linecount(posts_path) - 1)
        posts_to_update = max(0, linecount(plan_path))
        if not dry_run and posts_to_update and not getattr(context.args, "yes", False):
            action = "update legacy links in posts" if legacy else "update posts"
            
            print("---- APPLY MODE: REMOTE CHANGES ----")
            print(f"About to {action} via the Toolbox API (remote changes).")
            print(f"Input posts: {posts_path}")
            print(f"Input files: {files_path}")
            print(f"Will write: {updates_output_path} (OVERWRITES)")
            print(f"Will write: {deletes_output_path} (OVERWRITES)")
            
            print("URL rewrite:")
            print(f"  old: {old_prefix}")
            print(f"  new: {new_prefix}")
            
            print("Preflight:")
            print(f"  posts to update: {posts_to_update} of {total_posts}")
            if sample_pids:
                print(f"  sample pids: {', '.join(sample_pids)}")
            print(f"  unique URLs touched: {len(urls_touched)}")

            if not legacy:
                est_fileids = len({files[url]["fileid"] for url in urls_touched if url in files})
                print(f"  estimated delete candidates (fileids): {est_fileids}")

            if not confirm(context, "Type UPDATE to confirm: ", "UPDATE"):
                return

        # Apply (or simulate) the plan, streaming from disk and writing updates.csv results.
        posts_updated, posts_would_update, urls_to_delete, urls_to_keep, posts_errors = apply_update_plan(
            context=context, plan_path=plan_path,
        )

    finally:
        if plan_path is not None:
            try:
                plan_path.unlink()
            except FileNotFoundError:
                pass
    
    # Adjusting list of images that are now safe to delete.
    #
    # NOTE: in dry-run, we still compute the fileids we *would* delete, but we
    # write them to a separate file so a subsequent delete run won't
    # accidentally use them.
    urls_to_delete_final = set() if legacy else (urls_to_delete - urls_to_keep)
    files_to_delete = [files[url] for url in urls_to_delete_final if url in files]
    fileids_to_delete = {file["fileid"] for file in files_to_delete}
    
    if dry_run or legacy:
        deletes_output_path.write_text(json.dumps([]))
        context.path.fileids_to_delete_dry_run.write_text(json.dumps(sorted(fileids_to_delete)))
    else:
        deletes_output_path.write_text(json.dumps(sorted(fileids_to_delete)))
    
    if posts_errors:
        print("! Errors attempting to update the following posts:")
        for pid in posts_errors:
            print(" ", pid)
    
    if dry_run:
        print(f"Update posts: would update {posts_would_update} posts (dry-run)")
        print(f"Dry-run delete candidates written to: {context.path.fileids_to_delete_dry_run}")
    else:
        print(f"Update posts: {posts_updated} updated")
    
    # If old_urls still exist, warn the user (only relevant after a real update).
    if not dry_run and not legacy:
        if not check_old_urls(context, files_to_delete):
            print("! WARNING: At least one old_url or fileid was found in the posts")
            # raise an exception so it's obvious something is amiss
            raise Exception()


def delete_files(context: Namespace) -> None:
    """Delete Toolbox images given by set of fileids.
    
    There is no API endpoint for deleting files so this instead uses the
    Admin UI by simulating the Delete Files form submission.
    
    This of course won't work with files not hosted by Toolbox so it will
    throw an error if an attempt to made to do that.
    """
    dry_run = context.dry_run
    client = context.admin_client
    deletes_path = context.path.fileids_to_delete
    fileids_to_delete = json.loads(deletes_path.read_text())
    
    if dry_run:
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

    if not getattr(context.args, "yes", False):
        preview = ", ".join(str(x) for x in fileids_to_delete[:10])
        more = "" if len(fileids_to_delete) <= 10 else f"... (+{len(fileids_to_delete) - 10} more)"
        print(f"About to permanently delete {len(fileids_to_delete)} files from Toolbox.")
        print(f"First 10 fileids: {preview} {more}")
    
    # A final interactive confirmation helps avoid catastrophic deletes.
    if not confirm(context, "Type DELETE to confirm: ", "DELETE"):
        return
    
    successes: list = []
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


def check_new_urls(context: Namespace, files: dict) -> bool:
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
    sleep = .001 if dry_run else .25
    
    # If new_url is a local path then generate a local 'file://' url.
    # This only works in DRY_RUN since real post updates need public urls.
    if dry_run and not new_prefix.lower().startswith(("https://", "http://")):
        new_prefix = f"file://{str(Path(new_prefix).resolve())}/"
    
    new_url_func = get_new_url_func(old_prefix, thumb_prefix, new_prefix)
    
    # Confirm all old urls are accessible at new location except
    # for those files that were skipped or failed during download.
    seen = set()
    images_errors = set()
    count = max(0, linecount(posts_path) - 1)
    with alive_bar(count, title="Check new urls") as bar:
        for row in read_csv(posts_path):
            for url in literal_eval(row["image_urls"]):
                urldata = files.get(url, {})
                result = urldata.get("result")
                if url in seen or result in (File.skipped, File.error):
                    continue
                seen.add(url)
                new_url = urldata.get("new_url") or new_url_func(url)
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


def check_old_urls(context: Namespace, files_to_check: list, legacy: bool = False) -> bool:
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
        urls.update([f["url"], f["url_thumb"]])
        fileids.update([fr"={f['fileid']}", fr"/{f['fileid']}/"])
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
            _fileids = "|".join(batch)
            result = (grep["-Eh", _fileids, *posts_paths] | cut["-d,", "-f1"])(retcode=None)
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


def get_new_url_func(old_prefix: str, thumb_prefix: str, new_prefix: str):
    """Return 'new_url_func' function with the appropriate 'fixpath' change to the
    path for the case where the old url or new url contains a parameter string.
    In this case, the parameter string may need to be quoted (or unquoted) to
    escape special characters (or unescape) that aren't expected in a parameter.
    """
    
    def is_special(path):
        return '#' in path or '?' in path
    
    def safe_quote(path):
        """Unquote before quoting... to catch cases where the path is already quoted
        but if the unquoted string contains a '#', let's double-quote it, otherwise
        the server will interpret this as a fragment. This is done to retain the quoted
        '#' through the proxy we're using which otherwise exposes the '#' character
        in the filename too early.
        
        This may not be ideal (and may not be robust for alternative proxy
        configurations) but it works for the current setup.
        """
        unquoted_path = unquote(path)
        path = path if is_special(unquoted_path) else unquoted_path
        return quote(path)
    
    old_has_param = "?" in old_prefix
    new_has_param = "?" in new_prefix
    both_match = old_has_param is new_has_param
    
    if both_match:
        fixpath = lambda x: x  # noqa
    elif old_has_param:
        fixpath = unquote
    else:
        fixpath = safe_quote
    
    def new_url_func(url: str) -> str:
        if thumb_prefix and url.startswith(thumb_prefix):
            thumb = "thumb/"
            prefix = len(thumb_prefix)
        else:
            thumb = ""
            prefix = len(old_prefix)
        return new_prefix + thumb + fixpath(url[prefix:])
    
    return new_url_func


def find_urls_func(prefix: str | tuple[str, str]):
    """Return 'find_urls' function that returns a list of image urls found in a
    string that starts with any of the expected url prefixes.
    
    Later this will be extended to include support for other types of urls.
    """

    def find_urls(text: str) -> list[str]:
        urls: set[str] = set()
        
        for img in htmlparser(text).find_all("img"):
            src = img.get("src")
            if isinstance(src, str) and src.startswith(prefix):
                urls.add(src)
        
        return sorted(urls)
    
    return find_urls


def find_legacy_urls(text: str) -> list[str]:
    """Return legacy urls found in given html string"""
    html = htmlparser(text)
    prefixes: tuple[str, ...] = (
        "/file?id=",
        "https://s3.amazonaws.com/files.websitetoolbox.com/",
        "http://files.websitetoolbox.com/",
    )
    
    urls: set[str] = set()
    
    for img in html.find_all("img"):
        src = img.get("src")
        if isinstance(src, str) and src.startswith(prefixes):
            urls.add(src)
    
    for a in html.find_all("a"):
        href = a.get("href")
        if isinstance(href, str) and href.startswith(prefixes):
            urls.add(href)
    
    return sorted(urls)


def fileid_from_url(url: str) -> str | None:
    """Extract a fileid from any of the url formats we currently see.
    
    Supports:
      1) (legacy url) .../file?id=<fileid>  
      2) (legacy url) .../files.websitetoolbox.com/<toolid>/<fileid>/<filename>
         including:
           https://s3.amazonaws.com/files.websitetoolbox.com/<toolid>/<fileid>/<filename>
      3) Non-legacy urls and other urls/paths where fileid is the segment before a filename. 
         (with a best-effort fallback to last numeric segment)
    
    Returns None if no plausible fileid can be found.
    """
    try:
        p = urlparse(url)
    except Exception:
        return None
    
    path_segments = [s for s in (p.path or "").split("/") if s]
    
    # Case #1: /file?id=<fileid>
    if path_segments and path_segments[-1] == "file" and p.query:
        qs = parse_qs(p.query, keep_blank_values=True)
        vals = qs.get("id")
        if vals and vals[0]:
            return vals[0]
    
    # The other cases are all parsed the same way
    if len(path_segments) >= 2:
        last = unquote(path_segments[-1])
        second_to_last = path_segments[-2]
        
        # If last looks like a filename, second_to_last is probably a fileid
        if "." in last and second_to_last.isdigit():
            return second_to_last
        
        # Otherwise fallback to taking the last numeric segment
        for seg in reversed(path_segments):
            if seg.isdigit():
                return seg
    
    return None


def remove_bad_url(text: str, bad_url: str) -> str:
    """De-link a bad image and add "missing image" text.
    
    This is image-specific. Later this will be extended to include support
    for other types of files.
    """
    html = htmlparser(text)
    for img in html.find_all("img"):
        if img.get("src", "") == bad_url:
            notice = html.new_tag("span", attrs={"class": "missing-image"})
            notice.append("(missing image)")
            link = img.find_parent("a")
            badstuff = link or img
            badstuff.insert_after(" ", notice, Comment(f" Bad URL: {bad_url.replace('https://', '')} "))
            if link:
                link.attrs.pop("href", None)
            img.attrs.pop("src", None)
    return html.decode(formatter="html")


def friendly_size(size: int) -> str:
    """Return a friendly string representing the byte size with units"""
    unit = "bytes"
    s = float(size)
    for u in ("kb", "MB", "GB"):
        if s <= 1024:
            break
        unit = u
        s = s / 1024
    return f"{int(s)} {unit}"


@cache
def linecount(path: Path) -> int:
    """A quick way to count lines in a file. Defaults to 0 if file not found."""
    if not path.is_file():
        return 0
    out = wc("-l", str(path))
    return int(out.split()[0])


def parse_bool(val: str | bool | None, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    v = val.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {val!r}")


def batched(iterable, n):
    "Batch iterable into lists of length n. The last batch may be shorter."
    if n < 1:
        yield []
        return
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch


def read_csv(path: Path) -> Iterator:
    """A generator that returns the rows of a csv file, one row at a time.
    Each row is cast as a dictionary with keys corresponding to the column names
    in the first row.
    """
    if not path.exists():
        return
    
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def rotate_output_archive(context: Namespace, count: int = 10) -> None:
    """Archive the output folder and rotate the archives, keeping the last
    'count' archives.
    """
    output_dir = context.path.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir.with_name(output_dir.name + ".archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(sep="-").replace(':', "-")
    if list(output_dir.iterdir()):
        new_archive = archive_dir / timestamp
        output_dir.rename(new_archive)
        output_dir.mkdir(parents=True, exist_ok=True)
        archives = [str(i) for i in archive_dir.iterdir() if i.is_dir()]
        if len(archives) > count:
            for path in sorted(archives, reverse=True)[count:]:
                shutil.rmtree(path)


def confirm(context: Namespace, prompt: str, token: str) -> bool:
    """Require an interactive confirmation unless --yes was provided."""
    if getattr(context.args, "yes", False):
        return True
    try:
        typed = input(prompt).strip()
    except EOFError:
        print("No confirmation received (EOF). Aborting.")
        return False
    if typed != token:
        print("Confirmation not received. Aborting.")
        return False
    return True


def log(context: Namespace, text: str | None = None) -> None:
    """Write text to log file. Defaults to just logging the current command."""
    now = datetime.now().isoformat(sep=" ")
    txt = text if text else " ".join(sys.argv)
    with context.path.log.open("a") as log:
        log.write(f"{now} {txt}\n")


def config() -> Namespace:
    """Collect the config from environment variables.
    
    This leverages the `dotenv` library to collect default values from .env
    files in the current working directory which can then be overridden by
    the values collected from the environment. The `.env` file can be safely
    checked into source control. The `.env.secrets` file should probably not
    be checked into source control.
    
    Variable names in the environment should be prefixed by 'TOOLBOX_'.
    This prefix is stripped and the names then lowercased before being
    merged with the default values collected from .env files.
    """
    env: dict[str, str | None] = {k.lower(): v for k, v in dotenv_values(".env").items()}
    env_secrets: dict[str, str | None] = {k.lower(): v for k, v in dotenv_values(".env.secrets").items()}
    
    prefix = "TOOLBOX_"
    length = len(prefix)
    environ = {k[length:].lower(): v for k, v in os.environ.items() if k.startswith(prefix)}
    
    return Namespace(**{**env, **env_secrets, **environ})


def paths(config: Namespace) -> Namespace:
    """Generate pathlib Paths for all paths specified by config"""
    dirs = ("export_dir", "download_dir", "output_dir")
    output_files = ("posts_from_export", "posts_from_api", "posts", "files", "updates")
    
    paths = {name: Path(getattr(config, name)) for name in dirs}
    for name in output_files:
        paths[name] = paths["output_dir"] / f"{name}.csv"
    paths["fileids_to_delete"] = paths["output_dir"] / "fileids_to_delete.json"
    
    # In dry-run, we record the fileids that *would* be deleted here, without
    # enabling a subsequent accidental delete run to use them.
    paths["fileids_to_delete_dry_run"] = paths["output_dir"] / "fileids_to_delete.dry_run.json"
    
    paths["log"] = paths["output_dir"] / "log.txt"
    
    return Namespace(**paths)


class File(Enum):
    default = 0
    skipped = 1
    downloaded = 2
    error = 3


class BaseClient:
    
    def __init__(self, context: Namespace):
        self.context = context
        self.session = context.session
        self.dry_run = context.dry_run
        
    def _require_apply(self, action: str) -> None:
        """Refuse to run destructive operations unless explicitly applied."""
        if self.dry_run:
            raise RuntimeError(
                f"Refusing destructive action in dry-run: {action}. "
                "Re-run with --apply (or set TOOLBOX_DRY_RUN=false) to execute." 
            )


class Downloader(BaseClient):
    
    def download(self, url: str, path: Path) -> int:
        with self.session.get(url, stream=True, timeout=60) as resp:
            if resp.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as f:
                    for chunk in resp.iter_content(1024):
                        if chunk:
                            f.write(chunk)
                cl = resp.headers.get("Content-Length")
                size = int(cl) if cl is not None else path.stat().st_size
            else:
                size = 0
        return size


class AdminClient(BaseClient):
    
    def __init__(self, context: Namespace):
        super().__init__(context)
        admin_url = context.config.admin_url.rstrip("/")
        self.dashboard_endpoint = f"{admin_url}/dashboard"
        self.delete_endpoint = f"{admin_url}/mb/uploading"
        self.files_endpoint = f"{admin_url}/mb/uploading/files"
        self.headers = {
            "Cookie": context.config.admin_cookie,
            "Referer": self.files_endpoint,
        }
    
    def check_admin_auth(self) -> bool:
        """Check that the Admin cookie in config is valid. If not, return False."""
        url = self.dashboard_endpoint
        with self.session.get(url, headers=self.headers, timeout=30) as resp:
            return resp.ok
    
    @cached_property
    def hidden_defaults(self) -> dict:
        """Pull hidden defaults from the real page (trail/sort/reverse/loadedUsername)"""
        url = self.files_endpoint
        with self.session.get(url, headers=self.headers, timeout=30) as resp:
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            form = soup.find("form", {"id": "frmFiles"})
            hidden = {}
            if form:
                for i in form.select('input[type="hidden"][name]'):
                    hidden[i["name"]] = i.get("value", "")
            return hidden
    
    def delete_files(self, fileids) -> bool:
        self._require_apply(f"delete_files count={len(fileids)}")
        url = self.delete_endpoint
        defaults = list(self.hidden_defaults.items()) + [("action", "deleteFiles")]
        data = defaults + [("deleteimg", fileid) for fileid in fileids]
        with self.session.post(url, data=data, headers=self.headers, timeout=30) as resp:
            resp.raise_for_status()
            return resp.ok


class APIClient(BaseClient):
    
    def __init__(self, context: Namespace):
        super().__init__(context)
        api_url = context.config.api_url
        self.posts_endpoint = f"{api_url}/api/posts"
        self.headers = {
            "Accept": "application/json",
            "x-api-key": context.config.api_key,
            "x-api-username": context.config.api_username,
        }
    
    def check_api_auth(self) -> bool:
        """Check that the API key/username in config are valid. If not, return False."""
        url = self.posts_endpoint
        params = {"limit": 1}
        with self.session.get(url, params=params, headers=self.headers, timeout=30) as resp:
            return resp.ok
    
    def list_posts(self):
        """A generator that returns the results of the "List Posts" API call
        one response 'page' at a time. Each page contains up to 100 posts.
        
        This API call returns the most recent posts first. We can stop once we
        reach a post that has already been processed (via `posts_from_export`).
        Once this condition is reached, call 'close()' on the iterator returned
        by this generator and continue to next iteration which will end it.
        """
        params = {"limit": 100, "page": 1}
        get = self.session.get
        url = self.posts_endpoint
        
        with get(url, params=params, headers=self.headers, timeout=30) as resp:
            resp.raise_for_status()
            response = resp.json()
            yield response
        
        while response["has_more"]:
            params["page"] += 1
            with get(url, params=params, headers=self.headers, timeout=30) as resp:
                resp.raise_for_status()
                response = resp.json()
                yield response
            
            # Throttle the requests
            # 125719 posts / (100 posts/page) --> 1257 seconds or 21 minutes
            time.sleep(1)
            
            # Each request counts as a page view
            # so just do 3 requests (300 posts) during test runs
            if self.dry_run and params["page"] > 2:
                return
    
    def update_post(self, pid: str, message: str):
        """Call the "Update Post" API endpoint to update a post message."""
        self._require_apply(f"update_post pid={pid}")
        url = f"{self.posts_endpoint}/{pid}"
        body = {"content": message}
        with self.session.post(url, json=body, headers=self.headers, timeout=30) as resp:
            resp.raise_for_status()
            return resp.ok


if __name__ == "__main__":
    main()
