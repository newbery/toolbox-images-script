#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import warnings
from argparse import Namespace
from ast import literal_eval
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, unquote

import requests
from alive_progress import alive_bar, config_handler
from dotenv import dotenv_values
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from requests_file import FileAdapter

htmlparser = partial(BeautifulSoup, features="html.parser")
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
config_handler.set_global(bar="smooth", spinner="classic", receipt=False)


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


def main(argv: list = None) -> None:
    """Main command line entrypoint"""
    argv = argv or sys.argv[1:]
    args = parse_args(argv)
    args.func(args)


def parse_args(argv: list) -> Namespace:
    """Parse command line args"""
    modes = {"download": mode_download, "update": mode_update}

    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("mode", choices=list(modes))

    args = parser.parse_args(argv)

    args.config = config()
    args.path = paths(args)
    args.test_run = args.config.test_run != "false"
    args.func = modes[args.mode]
    
    if args.test_run:
        print("---- Test Run ----")

    return args


def mode_download(args: Namespace) -> None:
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
    
    # Confirm that we have access to api and admin UI
    if not check_authentication_settings(args):
        print("API or Admin UI is accessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    # Keep results from the last 5 runs for debugging purposes
    rotate_output_archive(args)

    # Process the data sources
    posts = posts_from_export(args)
    posts = posts_from_api(args, posts)
    images = images_from_posts(args, posts)

    # Generate final result
    output_download_results(args, images)
    
    print("Done")


def mode_upload(args: Namespace) -> None:
    """A placeholder for a possible future feature.
    
    Currently, this script assumes that the images are manually uploaded into
    the image host after the `download` mode but before the `update` mode.
    It would be nice for this script to be able to upload the images directly.
    """
    pass


def mode_update(args: Namespace) -> None:
    """Process the `posts.csv` result from the last `download` run and update
    the posts with image links updated to point to the new image host.
    
    This update attempts to be cautious about updates by confirming that all
    new image urls are reachable before the update. Otherwise, we assume
    the list of posts to update has already been filtered appropriately by
    the logic in the `download` mode.
    """
    
    # Confirm all images are accessible at new location
    if not check_new_image_urls(args):
        print("Aborting before updating posts!")
        return

    # Point of no return...
    images = update_posts(args)
    delete_images(args, images)

    print("Done")


def posts_from_export(args: Namespace) -> dict:
    """Process the posts listed in the `posts.csv` file from the Toolbox content
    export, collecting a list of image urls in the message text for any images
    hosted by the Toolbox server.
    """
    prefix = args.config.old_url
    posts_input_path = args.path.content_export_dir / "posts.csv"
    posts_output_path = args.path.posts_from_export
    posts = {}
    
    count = linecount(posts_input_path) - 1
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
                image_urls = {
                    img["src"] for img in htmlparser(message).find_all("img")
                    if img.get("src", "").startswith(prefix)
                }
                if image_urls:
                    found += 1
                posts[pid] = {"date": date, "image_urls": image_urls}
                posts_output.writerow([pid, date, image_urls, message])
                bar()
    
    print(f"From export: Processed {len(posts)} posts; Found {found} with image links")
    return posts


def posts_from_api(args: Namespace, posts: dict) -> dict:
    """Process the posts collected via the List Posts API, collecting a list of
    image urls in the message text for any images hosted by the Toolbox server.
    
    The most recent posts are returned first so once we reach a post that we've
    previously processed (via the content export processing), we can skip the rest.
    """
    test_run = args.test_run
    prefix = args.config.old_url
    posts_output_path = args.path.posts_from_api

    count = 0
    found = 0

    with alive_bar(title="From api") as bar:

        with posts_output_path.open("w", newline="") as f:
            fieldnames = ["pid", "date", "image_urls", "message"]
            posts_output = csv.writer(f)
            posts_output.writerow(fieldnames)

            page_count = 0
            api_requests = list_posts(args)
            for page in api_requests:
                page_count += 1
                for row in page["data"]:
                    pid = str(row["postId"])
                    if pid in posts:
                        # this is processed already so exit early
                        bar()
                        api_requests.close()
                        continue
                    count += 1
                    date = row["postTimestamp"]
                    message = row["message"]
                    image_urls = {
                        img["src"] for img in htmlparser(message).find_all("img")
                        if img.get("src", "").startswith(prefix)
                    }
                    if image_urls:
                        found += 1
                    posts[pid] = {"date": date, "image_urls": image_urls}
                    posts_output.writerow([pid, date, image_urls, message])
                    bar()
                
                if test_run and page_count > 3:
                    break

    print(f"From api: Processed {count} posts; Found {found} with image links")
    return posts


def images_from_posts(args: Namespace, posts: dict) -> dict:
    """Download images to be moved to the new image host.
    
    Images referenced in recent posts (given by SKIP_DAYS config) are skipped
    in the theory that recent posts may still be edited and recent posts are
    most likely to benefit from the Toolbox CDN so moving them is probably
    better postponed.
    """
    test_run = args.test_run
    test_post_id = args.config.test_post_id
    utc = timezone.utc
    last_date = datetime.now(utc) - timedelta(days=int(args.config.skip_days))
    prefix = args.config.old_url
    prefix_thumb = args.config.old_url_thumb
    img_dir = args.path.image_download_dir
    session = requests.Session()
    
    def download_image(url, path):
        if path.exists():
            size = path.stat().st_size
        else:
            resp = session.get(url, stream=True)
            if resp.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as f:
                    for chunk in resp.iter_content(1024):
                        f.write(chunk)
                size = int(resp.headers['content-length'])
            else:
                size = 0
        return size

    skipped = 0
    downloaded = 0
    errors = set()

    # Generate map of images to posts and set of images_to_exclude
    images = defaultdict(lambda: {"result": None, "pids": set(), "thumb": ""})
    images_to_exclude = set()
    for pid, post in posts.items():
        urls = post["image_urls"]
        if test_post_id and test_post_id != pid:
            images_to_exclude.update(urls)
        if datetime.fromtimestamp(int(post["date"]), utc) > last_date:
            images_to_exclude.update(urls)
        for url in urls:
            images[url]["pids"].add(pid)
            images[url]["thumb"] = url.replace(prefix, prefix_thumb)

    # Download images, skipping recent images and problem downloads
    size = 0
    count = 2 * (len(images) - len(images_to_exclude))
    with alive_bar(count, title="Downloads") as bar:

        for url, image in images.items():
            if url in images_to_exclude:
                skipped += 2
                image["result"] = Image.skipped
                continue

            thumb = image["thumb"]
            path = unquote(url[len(prefix):])
            path_full = img_dir / path
            path_thumb = img_dir / "thumb" / path
                
            # Full image
            _size = download_image(url, path_full)
            if _size:
                size += _size
                downloaded += 1
                image["result"] = Image.downloaded
            else:
                errors.add(url)
                image["result"] = Image.error
            
            # Thumb image
            # We don't need to report thumb errors
            # since thumbs are only used in attachments
            # and older posts may not have thumbs.
            _size = download_image(thumb, path_thumb)
            if _size:
                size += _size
                downloaded += 1
           
            bar(2)

            if test_run and downloaded > 22:
                skipped = 2 * len(images) - len(errors) - downloaded
                break

    if errors:
        print(f"Downloads: ! Errors (probably old deleted images):")
        for url in sorted(errors):
            print(f" {images[url]['pids']} {url}")

    print(f"Skipped {skipped} images and downloaded {downloaded} ({friendly_size(size)})")

    return images


def output_download_results(args: Namespace, images: dict) -> None:
    """Generate final output results from the merge of the results from processing
    the content export and the list_posts API.
    """
    from_export_path = args.path.posts_from_export
    from_api_path = args.path.posts_from_api
    posts_output_path = args.path.posts
    images_output_path = args.path.images

    # Collect set of all post ids that correspond to downloaded images
    # and then remove any post ids that correspond to non-downloaded images.
    # A post may contain both types but we don't want posts with non-downloaded images
    downloaded = 0
    pids = set()
    pids_to_remove = set()
    for image in images.values():
        if image["result"] is Image.downloaded:
            downloaded += 2
            pids.update(image["pids"])
        else:
            pids_to_remove.update(image["pids"])
    pids = pids - pids_to_remove

    count = linecount(from_export_path) + linecount(from_api_path) + len(images) - 2
    with alive_bar(count, title="Summarize") as bar:

        # Generate final `posts.csv` containing posts to be updated.
        with posts_output_path.open("w", newline="") as f:
            fieldnames = ["pid", "date", "image_urls", "message"]
            posts_output = csv.writer(f)
            posts_output.writerow(fieldnames)
            posts_data = chain(read_csv(from_export_path), read_csv(from_api_path))
            for row in posts_data:
                pid = row["pid"]
                if pid not in pids:
                    bar()
                    continue
                date = row["date"]
                message = row["message"]
                image_urls = row["image_urls"]
                posts_output.writerow([pid, date, image_urls, message])
                bar()

        # Generate `image.csv` with final data about processed images.
        with images_output_path.open("w", newline="") as f:
            fieldnames = ["url", "result", "pids"]
            images_output = csv.writer(f)
            images_output.writerow(fieldnames)
            for url, image in images.items():
                _pids = image["pids"]
                _result = image["result"]
                images_output.writerow([url, _result, _pids])
                bar()
    
    print(f"Summarize: {downloaded} images and {len(pids)} posts to update")
    

def check_authentication_settings(args: Namespace) -> bool:
    """Check that the authentication setting in config are valid. If not, return False."""
    
    api_url = args.config.api_url
    api_list_posts = f"{api_url}/api/posts"
    api_headers = {
        "Accept": "application/json",
        "x-api-key": args.config.api_key,
        "x-api-username": args.config.api_username,
    }
    api_params = {"limit": 1}
    
    admin_url = args.config.admin_url
    admin_dashboard = f"{admin_url}/dashboard"
    admin_headers = {"Cookie": args.config.admin_cookie}

    api_ok = requests.get(api_list_posts, params=api_params, headers=api_headers).ok
    admin_ok = requests.get(admin_dashboard, headers=admin_headers).ok
    
    return api_ok and admin_ok
   
    
def check_new_image_urls(args: Namespace) -> bool:
    """Check new urls for image urls given by posts in `posts.csv` output
    from last `download` run. If any are inaccessible then return False.
    """
    test_run = args.test_run
    old_prefix = args.config.old_url
    new_prefix = args.config.new_url
    posts_path = args.path.posts
    session = requests.Session()
    stop_fast = True  # Make this an environment setting?

    # If new_url is a local path then generate a local 'file://' url.
    # This only works in TEST_RUN since real post updates need public urls.
    if test_run and not new_prefix.lower().startswith(("https://", "http://")):
        new_prefix = f"file://{str(Path(new_prefix).resolve())}/"
        session.mount('file://', FileAdapter())

    new_url_func = get_new_url_func(old_prefix, new_prefix)

    # Confirm all full images are accessible at new location.
    # We don't check thumbs here since they don't always exist.
    images_errors = set()
    count = linecount(posts_path) - 1
    with alive_bar(count, title="Check images") as bar:
        for row in read_csv(posts_path):
            for old_url in literal_eval(row["image_urls"]):
                new_url = new_url_func(old_url)
                if session.get(new_url).status_code != 200:
                    images_errors.add(new_url)
                    if stop_fast:
                        raise Exception("Image not found:", new_url)

                # The proxy we're using throttles at 2500 req per 10 min.
                # Make this an environment setting?
                time.sleep(.25)

            bar()

    if images_errors:
        print("Check images: ! Errors attempting to access the following images:")
        for url in images_errors:
            print(" ", url)
    else:
        print("Check images: Passed; All images are accessible at new urls")

    return not images_errors


def update_posts(args: Namespace) -> None:
    """Update posts given by `posts.csv` output from last `download` run"""
    api_url = args.config.api_url
    old_prefix = args.config.old_url
    new_prefix = args.config.new_url
    posts_path = args.path.posts
    update_posts = f"{api_url}/api/posts/"
    headers = {
        "Accept": "application/json",
        "x-api-key": args.config.api_key,
        "x-api-username": args.config.api_username
    }

    request = requestor(args, "post")
    new_url_func = get_new_url_func(old_prefix, new_prefix)

    images_to_delete = set()  # images ready to be deleted from Toolbox
    images_to_keep = set()    # images not ready to be deleted
    
    posts_updated = 0
    posts_errors = set()
    count = linecount(posts_path) - 1
    with alive_bar(count, title="Update posts") as bar:
        for row in read_csv(posts_path):
            pid = row["pid"]
            image_urls = literal_eval(row["image_urls"])
            new_message = row["message"]
            for old_url in image_urls:
                new_url = new_url_func(old_url)
                new_message = new_message.replace(old_url, new_url)
            url = update_posts + pid
            body = {"content": new_message}  # why is 'message' named 'content' here?
            resp = request(url, json=body, headers=headers)
            if resp.status_code == 200:
                images_to_delete.update(image_urls)
                posts_updated += 1
            else:
                images_to_keep.update(image_urls)
                posts_errors.add(pid)
            time.sleep(1)  # Throttle API requests
            bar()

    # Adjusting list of images that are now safe to delete
    images_to_delete = images_to_delete - images_to_keep

    if posts_errors:
        print("! Errors attempting to update the following posts:")
        for pid in posts_errors:
            print(" ", pid)
    print(f"Update posts: {posts_updated} updated")
    
    return images_to_delete


def delete_images(args: Namespace, images: set) -> None:
    """Delete Toolbox images given by set of image urls"""

    admin_url = args.config.admin_url
    delete_images = f"{admin_url}/mb/uploading"  # strange name :)
    headers = {"Cookie": args.config.admin_cookie}
    request = requestor(args, "post")
    successes = []

    count = len(images)
    with alive_bar(count, title="Delete images") as bar:
        for batch in batched(images, 100):
            if not batch:
                continue
            image_ids = [url.split("/")[-2] for url in batch]
            params = {"action": "deleteFiles", "deleteimg": ",".join(image_ids)}
            resp = request(delete_images, headers=headers, params=params)
            if resp.ok:
                successes.extend(batch)
            else:
                print("Image delete failed with the following request params:", params)
                print("These were successfully deleted:")
                for url in successes:
                    print(" ", url)
                print("Aborting any more deletes!")

            length = len(batch)
            half = int(length / 2)
            bar(half)

            # Throttle these requests. These are not API calls so this is likely
            # not enforced but we still don't want to anger the Toolbox gods.
            time.sleep(.5)

            bar(length - half)
    
    print(f"Delete images: {count} deleted")


def get_new_url_func(old_prefix: str, new_prefix: str):
    """Return 'new_url_func' function with the appropriate 'fixpath' change to the
    path for the case where the old url or new url contains a parameter string.
    In this case, the parameter string may need to be quoted (or unquoted) to
    escape special characters (or unescape) that aren't expected in a parameter.
    """

    fixpath = None
    if "?" in old_prefix + new_prefix:
        if "?" not in old_prefix:
            fixpath = quote
        elif "?" not in new_prefix:
            fixpath = unquote
    
    # fixpath is a no-op when the old and new are both parameters or both not.
    if not fixpath:
        fixpath = lambda x: x  # noqa

    def new_url_func(url: str) -> str:
        return new_prefix + fixpath(url[len(old_prefix):])

    return new_url_func


def requestor(args: Namespace, method: str = "get"):
    """Just a shim to make it easy to switch to a fake requestor for testing"""
    if args.test_run:
        class FakeResponse:
            status_code = 200
            ok = True
        return lambda *a, **b: FakeResponse()
    
    session = requests.Session()
    return getattr(session, method)


def friendly_size(size: int) -> str:
    """Return a friendly string representing the byte size with units"""
    unit = "bytes"
    for u in ("kb", "MB", "GB"):
        if size <= 1024:
            break
        unit = u
        size = size / 1024
    return f"{int(size)} {unit}"


def linecount(path: Path) -> int:
    """A quick way to count lines in a file. Defaults to 0 if file not found."""
    result = subprocess.run(["wc", "-l", str(path)], capture_output=True)
    output = "0" if result.returncode else result.stdout
    return int(output.split()[0])


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


def list_posts(args: Namespace) -> Iterator:
    """A generator that returns the results of the "List Posts" API call
    one response 'page' at a time. Each page contains up to 100 posts.
    
    This API call returns the most recent posts first. We can stop once we
    reach a post that has already been processed (via `posts_from_export`).
    """
    test_run = args.test_run
    api_url = args.config.api_url
    list_posts = f"{api_url}/api/posts"
    headers = {
        "Accept": "application/json",
        "x-api-key": args.config.api_key,
        "x-api-username": args.config.api_username,
    }
    params = {"limit": 100, "page": 1}

    session = requests.Session()
    response = session.get(list_posts, params=params, headers=headers).json()
    yield response

    while response["has_more"]:
        params["page"] += 1
        response = session.get(list_posts, params=params, headers=headers).json()
        yield response
        
        # Throttle the requests
        # 125719 posts / (100 posts/page) --> 1257 seconds or 21 minutes
        time.sleep(1)

        # Each request counts as a page view
        # so just do 3 requests (300 posts) during test runs
        if test_run and params["page"] > 2:
            return


def rotate_output_archive(args: Namespace):
    """Archive the output folder and rotate the archives, keeping the last
    five archives.
    """
    output_dir = args.path.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir.with_name(output_dir.name + ".archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(sep="-").replace(':', "-")
    if list(output_dir.iterdir()):
        new_archive = archive_dir / timestamp
        output_dir.rename(new_archive)
        output_dir.mkdir(parents=True, exist_ok=True)
        archives = [str(i) for i in archive_dir.iterdir() if i.is_dir()]
        if len(archives) > 5:
            for path in sorted(archives, reverse=True)[5:]:
                shutil.rmtree(path)


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
    env = {k.lower(): v for k, v in dotenv_values(".env").items()}
    env_secrets = {k.lower(): v for k, v in dotenv_values(".env.secrets").items()}

    prefix = "TOOLBOX_"
    length = len(prefix)
    environ = {k[length:].lower(): v for k, v in os.environ.items() if k.startswith(prefix)}

    return Namespace(**{**env, **env_secrets, **environ})


def paths(args: Namespace) -> Namespace:
    """Generate pathlib Paths for all paths specified by config"""
    dirs = ("content_export_dir", "image_download_dir", "output_dir")
    output_files = ("posts_from_export", "posts_from_api", "posts", "images")
    
    paths = {name: Path(getattr(args.config, name)) for name in dirs}
    for name in output_files:
        paths[name] = paths["output_dir"] / f"{name}.csv"
    
    return Namespace(**paths)


class Image(Enum):
    skipped = 0
    error = 1
    downloaded = 2


if __name__ == "__main__":
    main()
