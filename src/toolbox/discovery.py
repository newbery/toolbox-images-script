"""
Discover posts and Website Toolbox-hosted files.
"""

import csv
from datetime import UTC, datetime, timedelta
from urllib.parse import unquote

from .context import Context, alive_bar
from .io import linecount, read_csv
from .models import FileMap, FileResult, ForumFile, Post, PostMap
from .urls import fileid_from_url, find_legacy_urls, find_urls_func


def posts_from_export(context: Context, legacy: bool = False) -> PostMap:
    """Process the posts listed in the `posts.csv` file from the Toolbox content
    export, collecting a list of image urls in the message text for any images
    hosted by the Toolbox server.
    """
    old_url: str = context.config.old_url
    old_url_thumb = context.config.old_url_thumb
    posts_input_path = context.path.export_dir / "posts.csv"
    posts_output_path = context.path.posts_from_export

    prefix: str | tuple[str, str] = (old_url, old_url_thumb) if old_url_thumb else old_url
    find_urls = find_legacy_urls if legacy else find_urls_func(prefix)

    posts: PostMap = {}
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
                posts[pid] = Post(date=date, image_urls=image_urls)
                posts_output.writerow([pid, date, image_urls, message])
                bar()

    print(f"From export: Processed {len(posts)} posts; Found {found} with image links")
    return posts


def posts_from_api(context: Context, posts: PostMap) -> PostMap:
    """Process the posts collected via the List Posts API, collecting a list of
    image urls in the message text for any images hosted by the Toolbox server.

    The most recent posts are returned first so once we reach a post that we've
    previously processed (via the content export processing), we can skip the rest.
    """
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
                    posts[pid] = Post(date=date, image_urls=image_urls)
                    posts_output.writerow([pid, date, image_urls, message])
                    bar()

                if stop or (context.dry_run and page_count > 3):
                    break

    print(f"From api: Processed {count} posts; Found {found} with image links")
    return posts


def files_from_posts(context: Context, posts: PostMap) -> FileMap:
    """Collect the file info for the urls found in the posts and tag the
    ones that should be excluded.

    Files/images referenced in recent posts (given by SKIP_DAYS config) will
    be excluded in the theory that recent posts may still be edited and recent
    posts are most likely to benefit from the Toolbox CDN so moving them is
    probably better postponed.
    """
    test_post_id = context.config.test_post_id
    last_date = datetime.now(UTC) - timedelta(days=context.config.skip_days)
    prefix = context.config.old_url
    prefix_thumb = context.config.old_url_thumb

    # This is not 100% reliable. It will be wrong if a non-Toolbox file host
    # provider is also using cloudfront.net. But it's good enough for us.
    toolbox = ".cloudfront.net/" in prefix

    # Generate map of files/images to posts and set of files_to_exclude
    files: FileMap = {}
    files_to_exclude: set[str] = set()
    for pid, post in posts.items():
        urls = post.image_urls

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
                ts = int(post.date)
            except Exception:
                print("Bad date")
                continue
            if datetime.fromtimestamp(ts, UTC) > last_date:
                files_to_exclude.update(fileids)

        for fileid, url in pairs:
            if fileid in files:
                files[fileid].pids.add(pid)
                if not files[fileid].url_thumb:
                    if prefix_thumb and url.startswith(prefix_thumb):
                        files[fileid].url_thumb = url
            else:
                thumb = ""
                if prefix_thumb and url.startswith(prefix_thumb):
                    thumb = url
                    url = url.replace(prefix_thumb, prefix)
                files[fileid] = ForumFile(
                    fileid=fileid,
                    url=url,
                    url_thumb=thumb,
                    url_file=f"/file?id={fileid}" if toolbox else "",
                    path=unquote(url[len(prefix) :]),
                    pids={pid},
                )

    # Tag files to be skipped
    for fileid in files_to_exclude:
        files[fileid].result = FileResult.skipped

    return files


def files_from_export(context: Context, posts: PostMap) -> FileMap:
    """This is a special mode for updating legacy links. In this case, we
    need to collect the image data from the export in order to construct
    the updated urls.
    """
    old_url = context.config.old_url
    files_input_path = context.path.export_dir / "attachment.csv"

    # Generate map of files/images to posts
    files: FileMap = {}
    for pid, post in posts.items():
        urls = post.image_urls

        pairs: list[tuple[str, str]] = []
        for url in urls:
            if fileid := fileid_from_url(url):
                pairs.append((fileid, url))

        for fileid, url in pairs:
            if fileid in files:
                files[fileid].pids.add(pid)
            else:
                files[fileid] = ForumFile(
                    fileid=fileid,
                    url=url,
                    pids={pid},
                )

    # Generate new_url from 'attachments.csv' export
    count = 0
    filecount = len(files)
    seen = set()
    for row in read_csv(files_input_path):
        fileid = row["fileid"]
        if fileid in files:
            seen.add(fileid)
            files[fileid].new_url = old_url + f"{fileid}/{row['filename']}"
            count += 1
            if count == filecount:
                break

    if count != len(files):
        # This should not happen. So stop and figure it out.
        raise Exception
        # missing = set(files) - seen
        # files_ = {k: v for k, v in files.items() if k in missing}
        # breakpoint()

    return files
