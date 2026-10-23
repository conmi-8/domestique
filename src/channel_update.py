"""Branch-channel updates for fork builds (docs/fork-updates.md).

A fork's CI (.github/workflows/fork-channel.yml) builds every branch it is
pushed to, installs and smoke-tests the Windows installer, and publishes it as
that branch's channel: a rolling prerelease tagged ``channel-<branch>`` whose
assets are ``Domestique-Setup.exe`` and ``build.json`` ({repo, branch, commit,
built_at, sha256}). The build carries the same identity in its bundle
(BUILD_INFO.json), so the app knows which branch it is and checks that
channel.

No version numbers are compared: branches fork, merge and cherry-pick freely,
so a channel's build is an update when it is another commit, built later on
the same branch -- or any build of a branch the rider switches to. An upstream
build carries no BUILD_INFO.json and nothing here applies to it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import httpx

from http_util import _icu_verify

GITHUB_API = "https://api.github.com"
TAG_PREFIX = "channel-"
SETUP_ASSET = "Domestique-Setup.exe"
INFO_ASSET = "build.json"
# Inno Setup: no wizard, no message boxes, close the running app, no reboot.
INSTALLER_ARGS = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS")
EXIT_AFTER_LAUNCH_S = 2.0   # lets the apply response reach the page first
_KEYS = ("repo", "branch", "commit", "built_at")


class UpdateError(Exception):
    """A channel update that cannot go ahead; the message is for the rider."""


def build_info(path: "Path | None" = None) -> "dict | None":
    """This build's identity, or None for a build no fork CI stamped.

    BUILD_INFO.json sits where VERSION does: next to the modules in a frozen
    bundle, at the repo root in a dev checkout."""
    candidates = [path] if path else [Path(__file__).parent / "BUILD_INFO.json",
                                      Path(__file__).parent.parent / "BUILD_INFO.json"]
    for p in candidates:
        try:
            info = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(info, dict) and all(isinstance(info.get(k), str) and info[k]
                                          for k in _KEYS):
            return {k: info[k] for k in _KEYS}
    return None


def can_self_update(info: "dict | None") -> bool:
    """Only the installed Windows build replaces itself."""
    return bool(info) and sys.platform == "win32" and bool(getattr(sys, "frozen", False))


def channel_tag(branch: str) -> str:
    """The release tag of a branch's channel (fork-channel.yml names it the same way)."""
    return TAG_PREFIX + re.sub(r"[^A-Za-z0-9._-]", "-", branch)


def is_update(current: dict, latest: dict) -> bool:
    """Another commit, built later on the same branch -- or any build of
    another branch, which the rider asked for by picking it."""
    if latest.get("commit") == current.get("commit"):
        return False
    if latest.get("branch") != current.get("branch"):
        return True
    return str(latest.get("built_at", "")) > str(current.get("built_at", ""))


def _get(url: str, **kw) -> httpx.Response:
    resp = httpx.get(url, timeout=kw.pop("timeout", 15), follow_redirects=True,
                     verify=_icu_verify(), **kw)
    if resp.status_code != 200:
        raise UpdateError(f"GitHub answered {resp.status_code} for {url}")
    return resp


def channels(repo: str) -> list:
    """The branches with a published build, newest release first."""
    rels = _get(f"{GITHUB_API}/repos/{repo}/releases?per_page=100").json()
    return [{"branch": r.get("name") or r["tag_name"][len(TAG_PREFIX):],
             "tag": r["tag_name"]}
            for r in rels if isinstance(r, dict)
            and str(r.get("tag_name", "")).startswith(TAG_PREFIX)]


def channel_build(repo: str, branch: str) -> dict:
    """The latest published build of ``branch``: its build.json plus the
    installer's download URL."""
    try:
        rel = _get(f"{GITHUB_API}/repos/{repo}/releases/tags/{channel_tag(branch)}").json()
    except UpdateError:
        raise UpdateError(f"no published build for branch {branch!r}")
    urls = {a.get("name"): a.get("browser_download_url") for a in rel.get("assets") or []}
    if not urls.get(INFO_ASSET) or not urls.get(SETUP_ASSET):
        raise UpdateError(f"the build for {branch!r} is incomplete (still publishing?)")
    info = _get(urls[INFO_ASSET]).json()
    if not all(isinstance(info.get(k), str) and info[k] for k in (*_KEYS, "sha256")):
        raise UpdateError(f"the build for {branch!r} has an unreadable build.json")
    return {**{k: info[k] for k in (*_KEYS, "sha256")}, "setup_url": urls[SETUP_ASSET]}


def download_verified(url: str, sha256: str, dest_dir: "Path | None" = None) -> Path:
    """Download the installer and check it against the build's sha256."""
    dest_dir = Path(dest_dir or tempfile.mkdtemp(prefix="domestique-update-"))
    dest = dest_dir / SETUP_ASSET
    digest = hashlib.sha256()
    with httpx.stream("GET", url, timeout=120, follow_redirects=True,
                      verify=_icu_verify()) as resp:
        if resp.status_code != 200:
            raise UpdateError(f"download failed ({resp.status_code})")
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes():
                digest.update(chunk)
                f.write(chunk)
    if digest.hexdigest().lower() != sha256.lower():
        dest.unlink(missing_ok=True)
        # The channel's assets are replaced one at a time: an installer newer
        # than its build.json is a publish still in progress.
        raise UpdateError("the download does not match its checksum -- "
                          "a new build may be publishing; try again in a few minutes")
    return dest


def launch_and_exit(installer: Path) -> None:
    """Start the installer detached, then end this process so it can replace
    the files. A hard exit is what the launcher already does on Windows
    (launcher._win_hard_exit: every DB write is committed as it happens)."""
    flags = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(installer), *INSTALLER_ARGS], creationflags=flags,
                     close_fds=True)
    threading.Timer(EXIT_AFTER_LAUNCH_S, os._exit, args=(0,)).start()
