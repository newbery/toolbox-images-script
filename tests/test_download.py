from toolbox import download, io, models


def test_download_files_marks_downloaded_and_errors(ctx, monkeypatch):
    """The `download_files` function should download files marked missing,
    update per-file result state to downloaded or error based on the download
    outcome, and preserve skipped entries.
    """

    # Create a downloader that writes dummy files and returns size
    class FakeDownloader:
        def __init__(self):
            self.calls = []

        def download(self, url, path_new):
            self.calls.append((url, str(path_new)))
            path_new.parent.mkdir(parents=True, exist_ok=True)
            path_new.write_bytes(b"abc")
            return 3

    ctx.downloader = FakeDownloader()
    files = {
        "1": models.ForumFile(fileid="1", url="https://x/1.jpg", path="1.jpg", pids={"p1"}),
        "2": models.ForumFile(
            fileid="2",
            url="https://x/2.jpg",
            url_thumb="https://x/t2.jpg",
            path="2.jpg",
            pids={"p2"},
        ),
        "3": models.ForumFile(
            fileid="3",
            url="https://x/3.jpg",
            path="3.jpg",
            pids={"p3"},
            result=models.FileResult.skipped,
        ),
    }
    # Make download fail for one file
    download_ = ctx.downloader.download

    def fake_download(url, path_new):
        if str(url).endswith("1.jpg"):
            return 0
        return download_(url, path_new)

    ctx.downloader.download = fake_download  # type: ignore
    out = download.download_files(ctx, files)
    assert out["1"].result == models.FileResult.error
    assert out["2"].result == models.FileResult.downloaded
    assert out["3"].result == models.FileResult.skipped


def test_summarize_writes_posts_and_files(ctx, write_csv):
    """The `summarize` function should write consolidated posts.csv and files.csv,
    excluding posts whose only referenced files are skipped/error according to
    the computed file results.
    """
    # posts_from_export and posts_from_api inputs
    write_csv(
        ctx.path.posts_from_export,
        ["pid", "date", "image_urls", "message"],
        [
            ["1", "0", "[]", "<p>no</p>"],
            [
                "2",
                "0",
                "['https://old.example.com/123/a.jpg']",
                "<img src='https://old.example.com/123/a.jpg'/>",
            ],
        ],
    )
    write_csv(
        ctx.path.posts_from_api,
        ["pid", "date", "image_urls", "message"],
        [
            [
                "3",
                "0",
                "['https://old.example.com/123/a.jpg']",
                "<img src='https://old.example.com/123/a.jpg'/>",
            ],
        ],
    )
    files = {
        "123": models.ForumFile(
            fileid="123",
            url="https://old.example.com/123/a.jpg",
            path="123/a.jpg",
            pids={"2", "3"},
            result=models.FileResult.downloaded,
        ),
        "999": models.ForumFile(
            fileid="999",
            url="https://old.example.com/999/missing.jpg",
            path="999/missing.jpg",
            pids={"1"},
            result=models.FileResult.skipped,
        ),
    }
    download.summarize(ctx, files)

    posts_out = list(io.read_csv(ctx.path.posts))

    # pid 1 should be skipped due to skipped file
    assert [r["pid"] for r in posts_out] == ["2", "3"]

    files_out = list(io.read_csv(ctx.path.files))
    assert any(r["fileid"] == "123" for r in files_out)
