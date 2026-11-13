# Copyright (C) 2026 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT
"""Put the documentation web page together from the HTML of each version.

The web page has a folder per version: ``main``, built against DOLFINx nightly, and one per minor
release from ``oldest`` in .github/docs-versions.toml (e.g. ``v1.5``), built from its newest tag.
The root of the web page, and every URL from before the web page had versions, leads to ``main``.
The versions are listed in ``versions.json``, which the version switcher reads.

Each version is built once, and its HTML is kept as ``docs-<tag>.tar.gz`` (``docs-main.tar.gz``
for main) in the ``docs-archive`` release, so that a push to main only builds main. main is
archived too, so that a failed build of it falls back to the last one that built.

Subcommands, in the order the deploy workflow uses them:

- ``plan``: what to build, from the tags and what is in the archive.
- ``assemble``: unpack each version, write ``versions.json`` and the redirects to ``main``.
- ``inject``: add the version switcher to every page.
- ``used``: the archived tarballs the web page uses, so the others can be removed.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / ".github" / "docs-versions.toml"
SWITCHER = REPO / ".github" / "docs" / "version-switcher.js"

ARCHIVE = os.environ.get("ARCHIVE", "docs-archive")
MAIN = "main"
# Release tags, e.g. v1.5.1 or v1.5.1.post0
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?$")

# Where the version switcher is put in the web page, and how a page refers to it
SWITCHER_PATH = "_versions/version-switcher.js"
MARKER = b"<!-- io4dolfinx-version-switcher -->"
# Hides the second search bar that sphinx-book-theme >= 1.3 shows in builds from before
# `navbar_persistent: []` was set in _config.yml (v1.5.1 and older), as a style so it doesn't flash
FIXES = b"<style>#pst-header .navbar-persistent--container{display:none}</style>"
# The page's own <head> comes first; later ones are inside embedded documents, such as the
# escaped viewer page in the widget state of a pyvista plot
HEAD = re.compile(rb"<head(?:\s[^>]*)?>", re.IGNORECASE)


def asset(tag: str) -> str:
    """The name of the archived tarball of a version."""
    return f"docs-{tag}.tar.gz"


def tags() -> list[str]:
    """The tags of the repository on GitHub."""
    out = subprocess.run(
        ["git", "ls-remote", "--tags", "origin"], check=True, capture_output=True, text=True
    ).stdout
    names = (line.split()[1].removeprefix("refs/tags/") for line in out.splitlines())
    return sorted({name.removesuffix("^{}") for name in names})


def version(tag: str) -> tuple[int, int, int, int] | None:
    """``(major, minor, patch, post)`` of a release tag, with ``post`` -1 if there is none."""
    match = TAG.match(tag)
    if match is None:
        return None
    major, minor, patch = (int(g) for g in match.groups()[:3])
    return (major, minor, patch, -1 if match.group(4) is None else int(match.group(4)))


def versions() -> list[dict[str, str]]:
    """All published versions, main first, then one per minor release, newest first."""
    config = tomllib.loads(CONFIG.read_text())
    oldest = tuple(int(i) for i in config["oldest"].split("."))
    newest: dict[tuple[int, int], tuple[tuple[int, int, int, int], str]] = {}
    for tag in tags():
        v = version(tag)
        if v is not None and v[:2] >= oldest and v > newest.get(v[:2], ((-1,) * 4, ""))[0]:
            newest[v[:2]] = (v, tag)

    main = {"label": MAIN, "name": "main (nightly)", "ref": MAIN, "image": config["nightly_image"]}
    releases = []
    for (major, minor), (_, tag) in sorted(newest.items(), reverse=True):
        image = config.get("images", {}).get(f"{major}.{minor}", config["image"])
        releases.append({"label": f"v{major}.{minor}", "name": tag, "ref": tag, "image": image})
    return [main, *releases]


def plan(args: argparse.Namespace):
    """Write the build matrix to $GITHUB_OUTPUT: main, and the releases not yet archived."""
    archived = set(args.archived.read_text().split())
    # A new tag doesn't change main, so a run for a tag only builds main if it isn't archived
    for_tag = os.environ.get("GITHUB_REF_TYPE") == "tag"
    jobs = [
        v | {"asset": asset(v["ref"])}
        for v in versions()
        if asset(v["ref"]) not in archived or (v["label"] == MAIN and not for_tag)
    ]
    for job in jobs:
        print(f"Building {job['label']} ({job['name']}) in {job['image']}")
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as f:
            f.write(f"matrix={json.dumps({'include': jobs})}\n")


def fetch(tarball: Path) -> bool:
    """Download a tarball from the archive, if it is there."""
    download = ["gh", "release", "download", ARCHIVE, "--pattern", tarball.name]
    download += ["--dir", str(tarball.parent), "--clobber"]
    return subprocess.run(download, check=False).returncode == 0


def redirect(page: Path, target: str):
    """A page leading to ``target`` (relative to the page), keeping the query and anchor."""
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8"><title>Redirecting</title>\n'
        f'<link rel="canonical" href="{target}">\n'
        f'<script>location.replace("{target}" + location.search + location.hash)</script>\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        f'</head><body><a href="{target}">Redirecting to {target}</a></body></html>\n'
    )


def assemble(args: argparse.Namespace):
    """Unpack every version into the web page, taking those not built in this run from the
    archive, and write ``versions.json`` and the redirects to main."""
    args.site.mkdir(parents=True, exist_ok=True)
    entries = []
    for v in versions():
        tarball = args.fresh / asset(v["ref"])
        if not tarball.exists() and not fetch(tarball):
            if v["label"] == MAIN:
                raise SystemExit("main is neither built nor archived")
            print(f"::warning::No build of {v['name']}, leaving {v['label']} out")
            continue
        with tarfile.open(tarball) as t:
            t.extractall(args.site / v["label"], filter="data")
        entries.append({"name": v["name"], "version": v["label"]})
    (args.site / "versions.json").write_text(json.dumps(entries, indent=2) + "\n")

    # The root, and the pages from before the web page had versions, lead to main:
    # <page>.html -> main/<page>.html
    folders = {e["version"] for e in entries}
    for page in (args.site / MAIN).rglob("*.html"):
        rel = page.relative_to(args.site / MAIN)
        if rel.parts[0].startswith("_") or rel.parts[0] in folders:
            continue
        redirect(args.site / rel, "../" * (len(rel.parts) - 1) + f"{MAIN}/{rel.as_posix()}")

    for e in entries:
        print(f"{e['version']}: {e['name']}")


def inject(args: argparse.Namespace):
    """Add the version switcher to every page of every version."""
    target = args.site / SWITCHER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SWITCHER, target)

    entries = json.loads((args.site / "versions.json").read_text())
    count = 0
    for e in entries:
        for page in (args.site / e["version"]).rglob("*.html"):
            html = page.read_bytes()
            head = HEAD.search(html)
            if MARKER in html or head is None:
                continue
            depth = len(page.relative_to(args.site).parts) - 1
            src = "../" * depth + SWITCHER_PATH
            tag = MARKER + FIXES + f'<script defer src="{src}"></script>'.encode()
            page.write_bytes(html[: head.end()] + tag + html[head.end() :])
            count += 1
    print(f"Added the version switcher to {count} pages")


def used(args: argparse.Namespace):
    """Print the archived tarballs the web page uses."""
    print(*(asset(v["ref"]) for v in versions()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)

    p = commands.add_parser("plan", help=plan.__doc__)
    p.add_argument("archived", type=Path, help="File listing the names of the archived tarballs")
    p.set_defaults(func=plan)

    p = commands.add_parser("assemble", help=assemble.__doc__)
    p.add_argument("fresh", type=Path, help="Folder with the tarballs built in this run")
    p.add_argument("site", type=Path, help="Where to put the web page")
    p.set_defaults(func=assemble)

    p = commands.add_parser("inject", help=inject.__doc__)
    p.add_argument("site", type=Path)
    p.set_defaults(func=inject)

    p = commands.add_parser("used", help=used.__doc__)
    p.set_defaults(func=used)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
