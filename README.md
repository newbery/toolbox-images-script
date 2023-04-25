
toolbox.py
----------

A utility script to move images from posts on a forum hosted with the Website Toolbox service
(https://www.websitetoolbox.com)

This is a simple one-trick pony. It's only meant to help with a very specific
case where a forum is hitting the server space limits. This script handles this
by downloading any Toolbox-hosted images found in post messages and then updating
the image links within these posts to point to a new image host.

The Website Toolbox API documentation is here:  
https://www.websitetoolbox.com/api/#introduction

Each API request is counted toward the page view usage allotment for the account
so it's best to take advantage of WS Toolbox's Forum Content Export feature. 
This script will process the exported `posts.csv` if it's available before attempting
to collect any remaining posts via the API. A content export is optional but this
can be considerably faster than collecting all post data from the API and may help
to avoid chewing through the page view allotment for the forum.

Note that there is no way to optimize the API call needed to update each individual
post, or the final calls to delete the old images, so these last two steps may still
generate a lot of page views depending on how many posts are updated.


Prerequisites
-------------

This is a Python script with a few third-party libraries as requirements.
As is usual in such cases, it's strongly recommended to install the requirements
in a Python virtualenv (or a Docker image). There are several ways to do this but
I'll just describe my preferred mechanism using Poetry (https://python-poetry.org/)

First, make sure you have Python 3.8 or greater available on your system.
If you need to install different versions of Python for any reason, I suggest
using `pyenv` (https://github.com/pyenv/pyenv)

Not directly related to this script but to install other python scripts with
dependencies, I generally recommend using `pipx` (https://pypa.github.io/pipx/)

Poetry can be installed using pipx: `pipx install poetry` 


Quick Start
-----------

```bash
# (1) Clone this repository
git clone https://github.com/newbery/toolbox-images-script.git
cd toolbox-images-script

# (2) Install the virtualenv (with script dependencies)
poetry install

# (3) Activate the virtualenv
poetry shell

# (4) Download and unzip the forum content export files (as csv)
# into a subdirectory of this directory (this step is manual and optional)

# (5) Update the script configuration (see below)

# (6) Run the script to process the posts data and download images
./toolbox.py download

# (7) Copy the downloaded images to the new image host (this step is manual)

# (8) Run the script to process the 'download' result and update the posts.
./toolbox.py update

```


Configuration
-------------

The script needs a little bit of configuration before it's ready for use.
It looks for two files in the working directory; `.env` and `.env.secrets`.
The second file does not exist so you have to create it. Just copy the template
file `.env.secrets.template` into a new file named `.env.secrets`.

These are "dot" files which in some systems are hidden from the directory
listing. If you don't see any dot files, just google how to make these
hidden files visible in your system.

Both files are reasonably documented and mostly self-explanatory but the
authentication settings in `.env.secrets` may need a little explanation.

The `API_KEY` is found in the forum Admin UI under `Integrate > API`.

The `API_USERNAME` is any username with "administrator" privileges.

The `ADMIN_COOKIE` is just the browser cookie you get when you navigate to
the forum Admin UI via "https://your-forum-domain/admin". Just copy this
cookie from whatever settings page your web browser offers for this purpose.

Technically, we don't need the entire cookie for our purposes but feel free
to copy the entire string as we will just ignore the unneeded bits. The minimum
cookie values you need are "username", "wtsession", and "forumuserid". The
full string should look something like this (all on one line):

`ADMIN_COOKIE = "username=aaa; wtsession=123456789abcdefghij; forumuserid=123456"`


What does toolbox.py do?
------------------------

The 'download' action:
- checks that the authentication values in `.env` files are valid (if not, abort)
- archives the results from a previous run (archive contains last 5 results)
- collects post data from `posts.csv` content export file
- collects post data from "List Posts" API
- downloads forum-hosted images from all posts older than 30 days (configurable)
- generates a final list of images-to-move (from all images successfully downloaded)
- generates a final list of posts-to-update (excluding posts with problem images)

The 'update' action:
- checks if all images-to-move are accessible from new host (if not, abort)
- updates all posts-to-update with updated image links in their messages
- deletes all images-to-move from old host

All API requests are throttled to 1 request per second since a 3 reqs/sec limit
is enforced by the server. If a lot of posts need to be updated, this can take
a while. You can marvel at the progress bars as you wait, or go get some coffee.


Supported images / files
------------------------

Only those forum-hosted images linked directly from within post message text
are managed with this script. This is most of the images in the forum for which
this script is written so that's good enough for our purposes.

The following images and files are NOT managed:
- images/files included in posts as "attachments"
- private message images/files
- album images/files
- event images/files
- profile picture
- profile avatar


Caveats
-------

Miscellaneous notes about issues encountered with this script.

- There were a couple of edge cases where images were linked directly to the backend
url instead of to the Cloudfront CDN. I discovered this late during manual inspection
of the results. Since this was just a handful of posts, it was easier to fix these
by manually downloading the images and manually updating the posts. It's probably
not worth fixing the script to account for these edge cases. If starting a migration
from scratch, maybe do these corrections first. Search for the offending links
in the content export csv ("https://s3.amazonaws.com/files.websitetoolbox.com/...)"

