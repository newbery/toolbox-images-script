import pytest

from toolbox import urls


def test_find_urls_func_filters_and_sorts():
    """The `find_urls_func` function should return a sorted, de-duplicated
    list of urls that match the configured legacy prefix, excluding
    non-matching hosts.
    """
    find_urls = urls.find_urls_func("https://old.example.com/")
    html = (
        "<p>"
        '<img src="https://old.example.com/a.jpg"/>'
        '<img src="https://old.example.com/b.jpg"/>'
        '<img src="https://other.example.com/c.jpg"/>'
        '<img src="https://old.example.com/a.jpg"/>'
        "</p>"
    )
    assert find_urls(html) == ["https://old.example.com/a.jpg", "https://old.example.com/b.jpg"]


def test_find_legacy_urls_in_img_and_link():
    """The `find_legacy_urls` function should extract legacy attachment references
    from both <a href='/file?id=...'> links and legacy hosted <img src='...'> urls.
    """
    html = (
        '<a href="/file?id=123">x</a>'
        '<img src="http://files.websitetoolbox.com/999/123/a.jpg"/>'
        '<img src="https://example.com/ignore.jpg"/>'
    )
    assert urls.find_legacy_urls(html) == [
        "/file?id=123",
        "http://files.websitetoolbox.com/999/123/a.jpg",
    ]


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://example.com/file?id=123", "123"),
        ("https://s3.amazonaws.com/files.websitetoolbox.com/999/123/a.jpg", "123"),
        ("https://cdn.example.com/999/123/a.jpg", "123"),
        ("https://cdn.example.com/x/123", "123"),
        ("not a url", None),
    ],
)
def test_fileid_from_url(url, expected):
    """The `fileid_from_url` function should extract the fileid component from
    supported url shapes and and should return None when the input is not a
    recognized file url.
    """
    assert urls.fileid_from_url(url) == expected


def test_remove_bad_url_de_links_image_and_adds_notice():
    """The `remove_bad_url` function should remove src/href references to a
    known-bad url and insert a visible '(missing image)' marker plus an HTML
    comment for traceability.
    """
    bad = "https://old.example.com/999/missing.jpg"
    html = f'<p><a href="{bad}"><img src="{bad}"/></a> hello</p>'
    out = urls.remove_bad_url(html, bad)

    # src and href should be removed
    assert 'src="' not in out
    assert 'href="' not in out

    # notice inserted
    assert "missing-image" in out
    assert "(missing image)" in out

    # comment includes Bad URL marker
    assert "Bad URL:" in out


def test_get_new_url_func_basic_and_thumb():
    """The `get_new_url_func` function should rewrite old urls to the new prefix,
    preserving the '/thumb/' variant when present.
    """
    f = urls.get_new_url_func(
        old_prefix="https://old.example.com/",
        thumb_prefix="https://old.example.com/thumb/",
        new_prefix="https://new.example.com/",
    )
    assert f("https://old.example.com/123/a.jpg") == "https://new.example.com/123/a.jpg"
    assert f("https://old.example.com/thumb/123/a.jpg") == "https://new.example.com/thumb/123/a.jpg"


def test_get_new_url_func_handles_param_quote_unquote():
    """The `get_new_url_func` function should safely quote urls when embedding
    them into query parameters, and should unquote them when converting from
    param-encoded to path-style urls.
    """
    # Old has no param, new has param -> safe_quote should be used
    f = urls.get_new_url_func(
        old_prefix="https://old.example.com/",
        thumb_prefix="",
        new_prefix="https://new.example.com/?url=",
    )

    # '#' should remain quoted so it doesn't become a fragment
    out = f("https://old.example.com/a#b.jpg")
    assert out.startswith("https://new.example.com/?url=")
    assert "%23" in out  # # is quoted

    # Old has param but new does not -> unquote should be used
    g = urls.get_new_url_func(
        old_prefix="https://old.example.com/?url=",
        thumb_prefix="",
        new_prefix="https://new.example.com/",
    )
    out2 = g("https://old.example.com/?url=a%23b.jpg")
    assert out2 == "https://new.example.com/a#b.jpg"
