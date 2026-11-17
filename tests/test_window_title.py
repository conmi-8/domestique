"""The window title says which build is running: plain on an upstream build,
branch@commit on a fork CI build (the taskbar is where the rider looks)."""
from __future__ import annotations

import channel_update
import launcher


def test_upstream_build_keeps_the_plain_title(monkeypatch):
    monkeypatch.setattr(channel_update, "build_info", lambda: None)
    assert launcher._window_title() == "Domestique"


def test_fork_build_names_its_branch_and_commit(monkeypatch):
    monkeypatch.setattr(channel_update, "build_info", lambda: {
        "repo": "conmi-8/domestique", "branch": "fork/dev",
        "commit": "2e982f62895caca68b8d3467bd86dbb9bb136d94",
        "built_at": "2026-11-17T10:48:07Z"})
    assert launcher._window_title() == "Domestique — fork/dev@2e982f6"


def test_unreadable_build_info_falls_back_to_the_plain_title(monkeypatch):
    def boom():
        raise OSError("disk")
    monkeypatch.setattr(channel_update, "build_info", boom)
    assert launcher._window_title() == "Domestique"
