import pytest

from toolbox import clients


def test_downloader_replaces_target_only_after_download_completes(ctx, tmp_path):
    target = tmp_path / "nested" / "image.jpg"
    part = target.with_name(f"{target.name}.part")

    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": "6"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, chunk_size):
            assert chunk_size == 1024
            yield b"abc"
            assert not target.exists()
            assert part.exists()
            yield b"def"

    class FakeSession:
        def get(self, url, stream, timeout):
            assert url == "https://example.com/image.jpg"
            assert stream is True
            assert timeout == 60
            return FakeResponse()

    ctx.session = FakeSession()

    size = clients.Downloader(ctx).download("https://example.com/image.jpg", target)

    assert size == 6
    assert target.read_bytes() == b"abcdef"
    assert not part.exists()


def test_downloader_removes_partial_file_and_preserves_target_on_error(ctx, tmp_path):
    target = tmp_path / "image.jpg"
    target.write_bytes(b"existing")
    part = target.with_name(f"{target.name}.part")

    class FakeResponse:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, chunk_size):
            assert chunk_size == 1024
            yield b"partial"
            raise RuntimeError("connection lost")

    class FakeSession:
        def get(self, url, stream, timeout):
            return FakeResponse()

    ctx.session = FakeSession()

    with pytest.raises(RuntimeError, match="connection lost"):
        clients.Downloader(ctx).download("https://example.com/image.jpg", target)

    assert target.read_bytes() == b"existing"
    assert not part.exists()
