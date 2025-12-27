#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
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


def main(argv: list | None = None) -> None:
    """Main command line entrypoint"""
    argv = argv or sys.argv[1:]
    args = parse_args(argv)
    args.func(args)


def parse_args(argv: list) -> Namespace:
    """Parse command line args"""
    modes = {
        "download_files": mode_download_files,
        "download_links": mode_download_links,
        "update_posts": mode_update_posts,
        "delete_files": mode_delete_files,
        "update_legacy_links": mode_update_legacy_links,
    }

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


def mode_download_files(args: Namespace) -> None:
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
    if not check_api_auth_settings(args):
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    # Keep results from the last 10 runs for debugging purposes
    # rotate_output_archive(args)
    log(args)

    # Process the data sources
    posts = posts_from_export(args)
    posts = posts_from_api(args, posts)
    files = files_from_posts(args, posts)
    files = download_files(args, files)

    # Generate summary
    summarize(args, files)
    
    print("Done")


def mode_download_links(args: Namespace) -> None:
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
    if not check_api_auth_settings(args):
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return
    
    # Override configs
    args.config.old_url_thumb = None
    args.config.skip_days = 0

    # Keep results from the last 10 runs for debugging purposes
    # rotate_output_archive(args)
    log(args)

    # Process the data sources
    posts = posts_from_export(args)
    posts = posts_from_api(args, posts)
    files = files_from_posts(args, posts)

    # Generate summary
    summarize(args, files)
    
    print("Done")


def mode_update_posts(args: Namespace) -> None:
    """Process the `posts.csv` result from the last `download_*` run and update
    the posts with image links updated to point to the new image host.
    
    This update attempts to be cautious about updates by confirming that all
    new image urls are reachable before the update. Otherwise, we assume
    the list of posts to update has already been filtered appropriately by
    the logic in the `download` mode.
    """
    
    # Confirm that we have access to api
    if not check_api_auth_settings(args):
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return

    log(args)
    update_posts(args)

    print("Done")


def mode_delete_files(args: Namespace) -> None:
    """Process the `posts.csv` result from the last `download` run and update
    the posts with image links updated to point to the new image host.
    
    This mode attempts to be cautious by loading the list of files-to-be-deleted
    from a successful run of 'update_posts'.
    """
    
    # Confirm that we have access to Admin UI
    if not check_admin_auth_settings(args):
        print("Admin UI is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return

    log(args)
    delete_files(args)

    print("Done")


def mode_update_legacy_links(args: Namespace) -> None:
    """Clean up legacy urls in a Toolbox forum.
    
    There are two types of legacy urls: "/file?=" and "files.websitetoolbox.com".
    To make migration easier, this mode will find all of these and update them
    to the urls used by the Cloudfront CDN for Toolbox-hosted files.
    
    Legacy links are all old so we'll just parse the content export for the data.
    """
    
    # Confirm that we have access to api
    if not check_api_auth_settings(args):
        print("API is inaccessible!")
        print("Maybe the authentication config is invalid?")
        print("Aborting")
        return

    # Keep results from the last 10 runs for debugging purposes
    # rotate_output_archive(args)
    log(args)

    # Override configs
    args.config.old_url_thumb = None
    args.config.skip_days = 0

    # Process the data sources
    posts = posts_from_export(args, legacy=True)
    files = files_from_export(args, posts)
    
    # Generate summary
    summarize(args, files, legacy=True)

    update_posts(args, legacy=True)

    print("Done")


def posts_from_export(args: Namespace, legacy: bool = False) -> dict:
    """Process the posts listed in the `posts.csv` file from the Toolbox content
    export, collecting a list of image urls in the message text for any images
    hosted by the Toolbox server.
    """
    old_url: str = args.config.old_url
    old_url_thumb: str = args.config.old_url_thumb
    posts_input_path = args.path.export_dir / "posts.csv"
    posts_output_path = args.path.posts_from_export
    
    prefix: str | tuple[str, str] = (old_url, old_url_thumb) if old_url_thumb else old_url
    find_urls = find_legacy_urls if legacy else find_urls_func(prefix)

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
                image_urls = find_urls(message)
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
    old_url = args.config.old_url
    old_url_thumb = args.config.old_url_thumb
    posts_output_path = args.path.posts_from_api
    
    prefix = (old_url, old_url_thumb) if old_url_thumb else old_url
    find_urls = find_urls_func(prefix)

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
                        # api_requests.close()
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
                
                if test_run and page_count > 3:
                    break

    print(f"From api: Processed {count} posts; Found {found} with image links")
    return posts


def files_from_posts(args: Namespace, posts: dict) -> dict:
    """Collect the file info for the urls found in the posts and tag the
    ones that should be excluded.
    
    Files/images referenced in recent posts (given by SKIP_DAYS config) will
    be excluded in the theory that recent posts may still be edited and recent
    posts are most likely to benefit from the Toolbox CDN so moving them is
    probably better postponed.
    """
    test_post_id = args.config.test_post_id
    utc = timezone.utc
    last_date = datetime.now(utc) - timedelta(days=int(args.config.skip_days))
    prefix = args.config.old_url
    prefix_thumb = args.config.old_url_thumb
    
    # This is not 100% reliable. It will be wrong if a non-Toolbox file host
    # provider is also using cloudfront.net. But it's good enough for us.
    toolbox = ".cloudfront.net/" in prefix

    # Generate map of files/images to posts and set of files_to_exclude
    files = {}
    files_to_exclude = set()
    for pid, post in posts.items():
        urls = post["image_urls"]
        
        # For the non-toolbox case, let's reuse the url as a fileid
        fileids = [url.split("/")[-2] for url in urls] if toolbox else urls

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

        for fileid, url in zip(fileids, urls, strict=True):
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


def files_from_export(args: Namespace, posts: dict) -> dict:
    """This is a special mode for updating legacy links. In this case, we
    need to collect the image data from the export in order to construct
    the updated urls.
    """
    old_url = args.config.old_url
    files_input_path = args.path.export_dir / "attachment.csv"

    # Generate map of files/images to posts
    files = {}
    for pid, post in posts.items():
        urls = post["image_urls"]
        fileids = [url.replace("=", "/").split("/")[-1] for url in urls]

        for fileid, url in zip(fileids, urls, strict=True):
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
        breakpoint()
        raise Exception

    return files


def download_files(args: Namespace, files: dict) -> dict:
    """Download files to be moved to the new image host"""
    test_run = args.test_run
    download_dir = args.path.download_dir
    session = requests.Session()

    def download_file(url, path):
        """Download a single file"""
        path_old = download_dir / path
        path_new = download_dir / "_new_" / path
        
        if path_old.exists():
            size = path_old.stat().st_size
        elif path_new.exists():
            size = path_new.stat().st_size
        else:
            resp = session.get(url, stream=True)
            if resp.status_code == 200:
                path_new.parent.mkdir(parents=True, exist_ok=True)
                with path_new.open("wb") as f:
                    for chunk in resp.iter_content(1024):
                        f.write(chunk)
                size = int(resp.headers['content-length'])
            else:
                size = 0
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

            if test_run and downloaded > 11:
                skipped = len(files) - len(errors) - downloaded
                break

    if errors:
        print("Downloads: ! Errors (probably old deleted images):")
        for fileid in sorted(errors):
            print(f" {files[fileid]['pids']} {files[fileid]['url']}")

    print(f"Skipped {skipped} images/files and downloaded {downloaded} ({friendly_size(size)})")

    return files


def summarize(args: Namespace, files: dict, legacy: bool = False) -> None:
    """Generate final output results from the merge of the results from processing
    the content export and the list_posts API.
    """
    from_export_path = args.path.posts_from_export
    from_api_path = args.path.posts_from_api
    posts_output_path = args.path.posts
    files_output_path = args.path.files

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


def update_posts(args: Namespace, legacy: bool = False) -> None:
    """Update posts given by `posts.csv` output from last `download` run"""
    test_run = args.test_run
    api_url = args.config.api_url
    old_prefix = args.config.old_url
    thumb_prefix = args.config.old_url_thumb
    new_prefix = args.config.new_url
    posts_path = args.path.posts
    files_path = args.path.files
    
    updates_output_path = args.path.updates
    deletes_output_path = args.path.fileids_to_delete

    update_posts = f"{api_url}/api/posts/"
    headers = {
        "Accept": "application/json",
        "x-api-key": args.config.api_key,
        "x-api-username": args.config.api_username
    }

    request = requestor(args, "post")
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
    if not check_new_urls(args, files):
        return
    
    urls_to_delete = set()  # files ready to be deleted from Toolbox
    urls_to_keep = set()    # files not ready to be deleted
    
    posts_updated = 0
    posts_errors = set()
    count = linecount(posts_path) - 1
    with alive_bar(count, title="Update posts") as bar:

        with updates_output_path.open("w", newline="") as f:
            fieldnames = ["pid", "result", "content"]
            updates_output = csv.writer(f)
            updates_output.writerow(fieldnames)
            
            for row in read_csv(posts_path):
                pid = row["pid"]
                image_urls = literal_eval(row["image_urls"])
                new_message = row["message"]
                for url in image_urls:
                    if url == "https://d28lcup14p4e72.cloudfront.net/197085/7976316/IMG_6454.jpgsss":
                        continue

                    file = files[url]

                    # Updating legacy link
                    if legacy:
                        new_message = new_message.replace(url, file["new_url"])
                        continue

                    # De-link missing file (discovered during download)
                    if file["result"] is File.error:
                        new_message = remove_bad_url(new_message, url)
                        continue

                    # Skip files that were skipped during download
                    if file["result"] is File.skipped:
                        continue

                    # Replace file url with new url
                    new_url = new_url_func(url)
                    new_message = new_message.replace(url, new_url)
                    
                    # Full image links often accompany thumb images
                    if "/thumb/" in url:
                        full_url = url.replace("thumb/", "")
                        new_full_url = new_url_func(full_url)
                        new_message = new_message.replace(full_url, new_full_url)

                    # Toolbox sometimes uses a special "/file?id=" link
                    if file["url_file"]:
                        new_full_url = new_url_func(file["url"])
                        new_message = new_message.replace(file["url_file"], new_full_url)

                if new_message == row["message"]:
                    # No changes
                    bar()
                    continue

                url = update_posts + pid
                body = {"content": new_message}  # why is 'message' named 'content' here?
                resp = request(url, json=body, headers=headers)
                if resp.status_code == 200:
                    urls_to_delete.update(image_urls)
                    posts_updated += 1
                    result = "success"
                else:
                    urls_to_keep.update(image_urls)
                    posts_errors.add(pid)
                    result = "fail"

                updates_output.writerow([pid, result, new_message])

                if not test_run:
                    time.sleep(1)  # Throttle API requests

                bar()

    # Adjusting list of images that are now safe to delete
    urls_to_delete = set() if legacy else (urls_to_delete - urls_to_keep)
    files_to_delete = [files[url] for url in urls_to_delete]
    fileids_to_delete = {file["fileid"] for file in files_to_delete}
    deletes_output_path.write_text(json.dumps(sorted(fileids_to_delete)))

    if posts_errors:
        print("! Errors attempting to update the following posts:")
        for pid in posts_errors:
            print(" ", pid)
    print(f"Update posts: {posts_updated} updated")

    # If old_urls still exist, warn the user
    if not check_old_urls(args, files_to_delete):
        print("! WARNING: At least one old_url or fileid was found in the posts")
        # raise an exception so it's obvious something is amiss
        raise Exception()


def delete_files(args: Namespace) -> None:
    """Delete Toolbox images given by set of fileids.
    
    There is no API endpoint for deleting files so this instead uses the
    Admin UI by simulating the Delete Files form submission.
    
    This of course won't work with files not hosted by Toolbox so it will
    throw an error if an attempt to made to do that.
    """
    test_run = args.test_run
    admin_url = args.config.admin_url.rstrip("/")
    deletes_path = args.path.fileids_to_delete
    fileids_to_delete = json.loads(deletes_path.read_text())

    delete_endpoint = f"{admin_url}/mb/uploading"
    files_page = f"{admin_url}/mb/uploading/files"

    session = requests.Session()
    session.headers.update({
        "User-Agent": useragent,
        "Cookie": args.config.admin_cookie,
        "Referer": files_page,
    })

    # Pull hidden defaults from the real page (trail/sort/reverse/loadedUsername)
    r = session.get(files_page)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form", {"id": "frmFiles"})
    hidden = {}
    if form:
        for i in form.select('input[type="hidden"][name]'):
            hidden[i["name"]] = i.get("value", "")

    # Force the action we want
    hidden["action"] = "deleteFiles"

    successes = []
    failed = {}
    count = len(fileids_to_delete)

    with alive_bar(count, title="Delete files") as bar:
        for fileids in batched(fileids_to_delete, 100):
            if not fileids:
                continue

            # This has changed recently (2025/12) to repeated deleteimg fields (like checkboxes), not CSV
            data = list(hidden.items()) + [("deleteimg", fid) for fid in fileids]

            if test_run:
                # Don’t actually delete in test runs
                resp = "TEST RUN"
                resp_ok = True
            else:
                resp = session.post(delete_endpoint, data=data)
                resp_ok = resp.ok

            if resp_ok:
                successes.extend(fileids)
                print(f"Deleted {len(successes)}")
            else:
                failed = {"data": data, "successes": successes, "response": resp}
                break

            bar(len(fileids))

            if not test_run:
                time.sleep(1.5)

    if failed:
        print("File delete failed; last payload had", len(failed["data"]), "fields")
        breakpoint()
        raise Exception

    print(f"Delete files: {count} deleted")


# def delete_files(args: Namespace) -> None:
#     """Delete Toolbox images given by set of fileids.
    
#     There is no API endpoint for deleting files so this instead uses the
#     Admin UI by simulating the Delete Files form submission.
    
#     This of course won't work with files not hosted by Toolbox so it will
#     throw an error if an attempt to made to do that.
#     """
#     test_run = args.test_run
#     admin_url = args.config.admin_url
#     deletes_path = args.path.fileids_to_delete
#     fileids_to_delete = json.loads(deletes_path.read_text())
    
#     delete_files = f"{admin_url}/mb/uploading"  # strange name :)
#     headers = {"Cookie": args.config.admin_cookie}
#     request = requestor(args, "post")
#     successes = []

#     failed = {}
#     count = len(fileids_to_delete)
#     with alive_bar(count, title="Delete files") as bar:
#         for fileids in batched(fileids_to_delete, 100):
#             if not fileids:
#                 continue
#             params = {"action": "deleteFiles", "deleteimg": ",".join(fileids)}
#             resp = request(delete_files, headers=headers, params=params)
#             if resp.ok:
#                 successes.extend(fileids)
#                 print(f"Deleted {len(successes)}")
#             else:
#                 failed = {
#                     "params": params,
#                     "successes": successes,
#                     "response": resp,
#                 }
#                 break
#                 # print("File delete failed with the following request params:", params)
#                 # if successes:
#                 #     print("These were successfully deleted:")
#                 #     for fileid in successes:
#                 #         print(" ", fileid)
#                 # print("Aborting any more deletes!")
#                 # print(resp.)
#                 # raise Exception

#             length = len(fileids)
#             half = int(length / 2)
#             bar(half)

#             # Throttle these requests. The rate limit for these requests is not
#             # advertised so this was determined from trial and error.
#             if not test_run:
#                 time.sleep(1.5)

#             bar(length - half)
            
#             # break
    
#     # breakpoint()

#     if failed:
#         print("File delete failed with the following request params:", failed["params"])
#         if failed["successes"]:
#             print("These were successfully deleted:")
#             for fileid in failed["successes"]:
#                 print(" ", fileid)
#         print("Aborting any more deletes!")
#         breakpoint()
#         raise Exception

#     print(f"Delete files: {count} deleted")


def check_new_urls(args: Namespace, files: dict) -> bool:
    """Check new urls for file/image urls given by posts in `posts.csv` output
    from last `output_results` run. If any are inaccessible then return False.
    
    This is checked before 'update_posts'.
    """
    test_run = args.test_run
    old_prefix = args.config.old_url
    thumb_prefix = args.config.old_url_thumb
    new_prefix = args.config.new_url
    posts_path = args.path.posts
    
    # Set this to False to generate a list of failing urls.
    # Make this an environment setting?
    stop_fast = False
    
    # The proxy we're using throttles at 2500 req per 10 min.
    # Make this sleep interval an environment setting?
    sleep = .001 if test_run else .25

    # If new_url is a local path then generate a local 'file://' url.
    # This only works in TEST_RUN since real post updates need public urls.
    if test_run and not new_prefix.lower().startswith(("https://", "http://")):
        new_prefix = f"file://{str(Path(new_prefix).resolve())}/"
        session = requests.Session()
        session.mount('file://', FileAdapter())
        head = session.head
    else:
        head = requestor(args, "head")

    new_url_func = get_new_url_func(old_prefix, thumb_prefix, new_prefix)

    # Confirm all old urls are accessible at new location except
    # for those files that were skipped or failed during download.
    seen = set()
    images_errors = set()
    count = linecount(posts_path) - 1
    with alive_bar(count, title="Check new urls") as bar:
        for row in read_csv(posts_path):
            for url in literal_eval(row["image_urls"]):
                urldata = files.get(url, {})
                result = urldata.get("result")
                if url in seen or result in (File.skipped, File.error):
                    continue
                seen.add(url)
                new_url = urldata.get("new_url") or new_url_func(url)
                with head(new_url) as response:
                    if response.status_code not in (200, 206):
                        images_errors.add((new_url, url))
                        if stop_fast:
                            raise Exception("Image not found:", new_url, url)
                time.sleep(sleep)
            bar()

    if images_errors:
        if stop_fast:
            breakpoint()
        print("Check new urls: !!! Errors attempting to access the following images:")
        for url in images_errors:
            print(" ", url)
    else:
        print("Check new urls: Passed; All images are accessible at new urls")

    return not images_errors


def check_old_urls(args: Namespace, files_to_check: list, legacy: bool = False) -> bool:
    """Check if any old_urls are still found in updated posts (and in posts
    not updated).
    
    1) Search 'updates.csv' for any 'url', 'url_thumb', or 'url_file'.
    2) Search a subset of 'posts_from_export.csv' and 'posts_from_api.csv'
    that includes only the non-updated posts.
    3) If any matches are found, print out the list and return False, otherwise
    return True.
    
    This is checked after 'update_posts'
    """
    updates_path = args.path.updates
    from_export_path = args.path.posts_from_export
    from_api_path = args.path.posts_from_api
    posts_paths = str(from_export_path) if legacy else f"{from_export_path} {from_api_path}"
    
    urls = set()
    fileids = set()
    for f in files_to_check:
        urls.update([f["url"], f["url_thumb"]])
        fileids.update([fr"={f['fileid']}", fr"/{f['fileid']}/"])
    urls.discard("")

    found_in_updated = []
    found_in_nonupdated = []
    
    count = len(urls) + len(fileids)
    batch_count = (int(count / 100)) or 1

    with alive_bar(count, title="Check old urls") as bar:

        for batch in batched(urls, batch_count):
            _urls = "|".join(batch)
            result = (grep["-E", _urls, updates_path] | cut["-d,", "-f1"])(retcode=None)
            found_in_updated += result.split()
            bar(len(batch))

        for batch in batched(fileids, batch_count):
            _fileids = "|".join(batch)
            result = (grep["-Eh", _fileids, posts_paths] | cut["-d,", "-f1"])(retcode=None)
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
    """Return legacy urls found in given string"""
    html = htmlparser(text)
    prefixes: tuple[str, ...] = (
        "/file?id=",
        "https://s3.amazonaws.com/files.websitetoolbox.com/",
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
                del link["href"]
            del img["src"]
    return html.decode(formatter="html")


def check_api_auth_settings(args: Namespace) -> bool:
    """Check that the API key/username in config are valid. If not, return False."""
    api_url = args.config.api_url
    api_list_posts = f"{api_url}/api/posts"
    api_headers = {
        "User-Agent": useragent,
        "Accept": "application/json",
        "x-api-key": args.config.api_key,
        "x-api-username": args.config.api_username,
    }
    api_params = {"limit": 1}
    response = requests.get(api_list_posts, params=api_params, headers=api_headers)  # noqa
    if response.ok:
        return True
    else:
        breakpoint()
        return False
    

def check_admin_auth_settings(args: Namespace) -> bool:
    """Check that the Admin cookie in config is valid. If not, return False."""
    admin_url = args.config.admin_url
    admin_dashboard = f"{admin_url}/dashboard"
    admin_headers = {
        "User-Agent": useragent,
        "Cookie": args.config.admin_cookie,
    }
    return requests.get(admin_dashboard, headers=admin_headers).ok  # noqa


def requestor(args: Namespace, method: str = "get", headers: dict | None = None):
    """Just a shim to make it easy to switch to a fake requestor for testing"""
    if args.test_run:

        class FakeResponse:
            status_code = 200
            ok = True

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False
                
        return lambda *a, **b: FakeResponse()
    
    session = requests.Session()
    session.headers.update({"User-Agent": useragent} | (headers or {}))
    return getattr(session, method)


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


def linecount(path: Path) -> int:
    """A quick way to count lines in a file. Defaults to 0 if file not found."""
    if not path.is_file():
        return 0
    out = wc("-l", str(path))
    return int(out.split()[0])


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
    Once this condition is reached, call 'close()' on the iterator returned
    by this generator and continue to next iteration which will end it.
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


def rotate_output_archive(args: Namespace, count: int = 10) -> None:
    """Archive the output folder and rotate the archives, keeping the last
    'count' archives.
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
        if len(archives) > count:
            for path in sorted(archives, reverse=True)[count:]:
                shutil.rmtree(path)


def log(args: Namespace, text: str | None = None) -> None:
    """Write text to log file. Defaults to just logging the current command."""
    now = datetime.now().isoformat(sep=" ")
    txt = text if text else " ".join(sys.argv)
    with args.path.log.open("a") as log:
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


def paths(args: Namespace) -> Namespace:
    """Generate pathlib Paths for all paths specified by config"""
    dirs = ("export_dir", "download_dir", "output_dir")
    output_files = ("posts_from_export", "posts_from_api", "posts", "files", "updates")
    
    paths = {name: Path(getattr(args.config, name)) for name in dirs}
    for name in output_files:
        paths[name] = paths["output_dir"] / f"{name}.csv"
    paths["fileids_to_delete"] = paths["output_dir"] / "fileids_to_delete.json"
    paths["log"] = paths["output_dir"] / "log.txt"
    
    return Namespace(**paths)


class File(Enum):
    default = 0
    skipped = 1
    downloaded = 2
    error = 3


if __name__ == "__main__":
    main()
