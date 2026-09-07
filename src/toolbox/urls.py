"""
URL discovery, rewriting, and HTML cleanup helpers.
"""

import warnings
from collections.abc import Callable
from functools import partial
from urllib.parse import parse_qs, quote, unquote, urlparse

from bs4 import BeautifulSoup, Comment, MarkupResemblesLocatorWarning

htmlparser = partial(BeautifulSoup, features="html.parser")
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def get_new_url_func(
    old_prefix: str, thumb_prefix: str | None, new_prefix: str
) -> Callable[[str], str]:
    """Return 'new_url_func' function with the appropriate 'fixpath' change to the
    path for the case where the old url or new url contains a parameter string.
    In this case, the parameter string may need to be quoted (or unquoted) to
    escape special characters (or unescape) that aren't expected in a parameter.
    """

    def is_special(path: str) -> bool:
        return "#" in path or "?" in path

    def safe_quote(path: str) -> str:
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
        fixpath = noop
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


def find_urls_func(prefix: str | tuple[str, str]) -> Callable[[str], list[str]]:
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
            badstuff.insert_after(
                " ", notice, Comment(f" Bad URL: {bad_url.replace('https://', '')} ")
            )
            if link and link.get("href") == bad_url:
                link.attrs.pop("href", None)
            img.attrs.pop("src", None)
    return html.decode(formatter="html")


def noop(x):
    return x
