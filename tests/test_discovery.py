from datetime import datetime, timezone

from toolbox import discovery, io, models


def test_posts_from_export_collects_urls(context_tmp):
    """The `posts_from_export` function should parse the export posts csv,
    extract image urls, and write a normalized `posts_from_export` csv with
    per-post url lists.
    """
    # write export/posts.csv
    posts_csv = context_tmp.path.export_dir / "posts.csv"
    posts_csv.write_text(
        "pid,date,message\n"
        '1,100,<p><img src="https://old.example.com/123/a.jpg"/></p>\n'
        "2,101,<p>no image</p>\n",
        encoding="utf-8",
    )

    posts = discovery.posts_from_export(context_tmp)
    assert set(posts) == {"1", "2"}
    assert posts["1"].image_urls == ["https://old.example.com/123/a.jpg"]
    assert posts["2"].image_urls == []
    out_rows = list(io.read_csv(context_tmp.path.posts_from_export))
    assert out_rows[0]["pid"] == "1"


def test_posts_from_api_stops_when_pid_already_seen(context_tmp):
    """The `posts_from_api` function should paginate api results into the posts
    dict, extract image urls, and stop early once it encounters a postId already
    present in the seed posts map.
    """

    class FakeApiRequests:
        def __init__(self, pages):
            self.pages = pages
            self.closed = False

        def __iter__(self):
            yield from self.pages

        def close(self):
            self.closed = True

    class FakeClient:
        def __init__(self, pages):
            self._pages = pages

        def list_posts(self):
            return FakeApiRequests(self._pages)

    # Existing post from export already processed
    posts = {"1": models.Post(date="100", image_urls=[])}
    pages = [
        {
            "data": [
                {
                    "postId": 2,
                    "postTimestamp": "200",
                    "message": '<img src="https://old.example.com/2.jpg"/>',
                }
            ]
        },
        {"data": [{"postId": 1, "postTimestamp": "199", "message": "stop here"}]},
        {"data": [{"postId": 3, "postTimestamp": "198", "message": "should not be reached"}]},
    ]
    context_tmp.api_client = FakeClient(pages)
    out = discovery.posts_from_api(context_tmp, posts)
    assert "2" in out
    assert out["2"].image_urls == ["https://old.example.com/2.jpg"]

    # "3" should not be processed due to early stop
    assert "3" not in out


def test_files_from_posts_toolbox_parses_fileids_and_thumb(context_tmp):
    """The `files_from_posts` function should group image urls by fileid,
    detect thumb urls, record canonical paths, and accumulate the set of
    post ids referencing each file.
    """
    # Make it look like a toolbox/cloudfront url so toolbox=True
    context_tmp.config.old_url = "https://abc.cloudfront.net/"
    context_tmp.config.old_url_thumb = "https://abc.cloudfront.net/thumb/"
    context_tmp.config.skip_days = 0
    posts = {
        "1": models.Post(date="0", image_urls=["https://abc.cloudfront.net/999/123/a.jpg"]),
        "2": models.Post(date="0", image_urls=["https://abc.cloudfront.net/thumb/999/123/a.jpg"]),
    }
    files = discovery.files_from_posts(context_tmp, posts)
    assert "123" in files
    f = files["123"]
    assert f.url.endswith("/999/123/a.jpg")
    assert f.url_thumb.endswith("/thumb/999/123/a.jpg")
    assert f.pids == {"1", "2"}
    assert f.path == "999/123/a.jpg"


def test_files_from_posts_skips_recent_or_nonmatching_test_post(context_tmp):
    """The `files_from_posts` function should mark files as skipped when the
    containing post is newer than the configured skip_days threshold.
    """
    context_tmp.config.old_url = "https://abc.cloudfront.net/"
    context_tmp.config.old_url_thumb = ""
    context_tmp.config.skip_days = 1  # skip anything newer than 1 day ago
    now_ts = int(datetime.now(timezone.utc).timestamp())
    posts = {
        "1": models.Post(date=str(now_ts), image_urls=["https://abc.cloudfront.net/1/111/a.jpg"])
    }

    files = discovery.files_from_posts(context_tmp, posts)
    assert files["111"].result == models.FileResult.skipped


def test_files_from_export_builds_new_url(context_tmp):
    """The `files_from_export` function should resolve legacy '/file?id=' links
    using attachment metadata to produce a concrete legacy file url for later
    checking/updating.
    """
    # Posts with legacy /file?id= urls; attachments.csv supplies filename
    context_tmp.config.old_url = "https://abc.cloudfront.net/"
    posts = {"1": models.Post(date="0", image_urls=["/file?id=123"])}
    attach = context_tmp.path.export_dir / "attachment.csv"
    attach.write_text("fileid,filename\n123,a.jpg\n", encoding="utf-8")

    files = discovery.files_from_export(context_tmp, posts)
    assert files["123"].new_url == "https://abc.cloudfront.net/123/a.jpg"
