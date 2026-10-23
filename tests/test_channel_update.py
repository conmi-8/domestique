"""Fork channel updates (src/channel_update.py, /api/update/channel|apply).

GitHub and the installer are stubbed: what is pinned is the decision (which
build is an update), the identity a build carries, the checksum gate, and
that an upstream build -- no BUILD_INFO.json -- is left exactly as it was.
"""
import hashlib
import json
from pathlib import Path

import httpx
import pytest

import channel_update as cu

CUR = {"repo": "me/domestique", "branch": "fork/dev",
       "commit": "a" * 40, "built_at": "2026-10-20T10:00:00Z"}


# ── identity and the decision ───────────────────────────────────────────────

def test_build_info_reads_a_stamped_build(tmp_path):
    p = tmp_path / "BUILD_INFO.json"
    p.write_text(json.dumps({**CUR, "extra": 1}), encoding="utf-8")
    assert cu.build_info(p) == CUR


@pytest.mark.parametrize("content", ["", "{}", "[]", json.dumps({**CUR, "commit": ""})])
def test_build_info_none_when_not_stamped(tmp_path, content):
    p = tmp_path / "BUILD_INFO.json"
    p.write_text(content, encoding="utf-8")
    assert cu.build_info(p) is None
    assert cu.build_info(tmp_path / "missing.json") is None


def test_channel_tag_matches_the_workflow():
    assert cu.channel_tag("fork/dev") == "channel-fork-dev"
    assert cu.channel_tag("feat_x.2") == "channel-feat_x.2"


@pytest.mark.parametrize("latest,expected", [
    ({**CUR}, False),                                                   # same commit
    ({**CUR, "commit": "b" * 40, "built_at": "2026-10-21T10:00:00Z"}, True),   # later, same branch
    ({**CUR, "commit": "b" * 40, "built_at": "2026-10-19T10:00:00Z"}, False),  # older, same branch
    ({**CUR, "branch": "other", "commit": "b" * 40,
      "built_at": "2026-10-01T10:00:00Z"}, True),                       # another branch, any age
])
def test_is_update(latest, expected):
    assert cu.is_update(CUR, latest) is expected


def test_only_the_frozen_windows_build_updates_itself(monkeypatch):
    monkeypatch.setattr(cu.sys, "platform", "win32")
    monkeypatch.setattr(cu.sys, "frozen", True, raising=False)
    assert cu.can_self_update(CUR)
    assert not cu.can_self_update(None)
    monkeypatch.setattr(cu.sys, "platform", "linux")
    assert not cu.can_self_update(CUR)


# ── GitHub, stubbed ─────────────────────────────────────────────────────────

SETUP = b"installer bytes"
LATEST = {**CUR, "commit": "b" * 40, "built_at": "2026-10-21T10:00:00Z",
          "sha256": hashlib.sha256(SETUP).hexdigest()}


def _github(monkeypatch, build=LATEST, setup=SETUP):
    rels = [{"tag_name": "channel-fork-dev", "name": "fork/dev"},
            {"tag_name": "channel-main", "name": "main"},
            {"tag_name": "v3.13.1", "name": "upstream tag"}]
    rel = {"assets": [{"name": "build.json", "browser_download_url": "https://x/build.json"},
                      {"name": "Domestique-Setup.exe", "browser_download_url": "https://x/setup"}]}

    def get(url, **kw):
        req = httpx.Request("GET", url)
        if url.endswith("/releases?per_page=100"):
            return httpx.Response(200, json=rels, request=req)
        if url.endswith("/releases/tags/channel-fork-dev"):
            return httpx.Response(200, json=rel, request=req)
        if url == "https://x/build.json":
            return httpx.Response(200, json=build, request=req)
        return httpx.Response(404, request=req)

    class _Stream:
        def __init__(self, *a, **kw):
            self.status_code = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def iter_bytes(self):
            yield setup

    monkeypatch.setattr(cu.httpx, "get", get)
    monkeypatch.setattr(cu.httpx, "stream", _Stream)


def test_channels_and_channel_build(monkeypatch):
    _github(monkeypatch)
    assert [c["branch"] for c in cu.channels("me/domestique")] == ["fork/dev", "main"]
    b = cu.channel_build("me/domestique", "fork/dev")
    assert b["commit"] == "b" * 40 and b["setup_url"] == "https://x/setup"
    with pytest.raises(cu.UpdateError, match="no published build"):
        cu.channel_build("me/domestique", "nope")


def test_download_checks_the_sha256(monkeypatch, tmp_path):
    _github(monkeypatch)
    path = cu.download_verified("https://x/setup", LATEST["sha256"], tmp_path)
    assert path.read_bytes() == SETUP
    _github(monkeypatch, setup=b"something else")
    with pytest.raises(cu.UpdateError, match="checksum"):
        cu.download_verified("https://x/setup", LATEST["sha256"], tmp_path)
    assert not (tmp_path / cu.SETUP_ASSET).exists()


# ── the endpoints ───────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import app as app_module
    return TestClient(app_module.app), app_module


def test_upstream_build_is_unchanged(client, monkeypatch):
    c, app_module = client
    monkeypatch.setattr(app_module, "_BUILD_INFO", None)
    assert c.get("/api/update/channel").json() == {"managed": False}
    assert c.get("/api/version").json()["build"] is None
    assert c.post("/api/update/apply", json={}).status_code == 400
    monkeypatch.setattr(app_module, "_upstream_update_check",
                        lambda force=0: {"update_available": True, "download_url": "u"})
    assert c.get("/api/update/check").json()["download_url"] == "u"


def test_fork_build_status_and_upstream_news(client, monkeypatch):
    c, app_module = client
    monkeypatch.setattr(app_module, "_BUILD_INFO", CUR)
    _github(monkeypatch)
    st = c.get("/api/update/channel").json()
    assert st["managed"] and st["update_available"]
    assert st["latest"]["commit"] == "b" * 40 and "setup_url" not in st["latest"]
    assert [ch["branch"] for ch in st["channels"]] == ["fork/dev", "main"]
    monkeypatch.setattr(app_module, "_upstream_update_check",
                        lambda force=0: {"update_available": True, "download_url": "u",
                                         "asset_name": "a"})
    up = c.get("/api/update/check").json()
    assert up["managed"] and up["download_url"] is None and up["update_available"]


def test_apply_downloads_verifies_and_launches(client, monkeypatch, tmp_path):
    c, app_module = client
    monkeypatch.setattr(app_module, "_BUILD_INFO", CUR)
    monkeypatch.setattr(cu, "can_self_update", lambda info: True)
    _github(monkeypatch)
    real = cu.download_verified
    monkeypatch.setattr(cu, "download_verified",
                        lambda url, sha: real(url, sha, tmp_path))
    launched = []
    monkeypatch.setattr(cu, "launch_and_exit", lambda p: launched.append(Path(p)))
    r = c.post("/api/update/apply", json={"branch": "fork/dev"})
    assert r.status_code == 200 and r.json()["commit"] == "b" * 40
    assert launched == [tmp_path / cu.SETUP_ASSET]

    _github(monkeypatch, setup=b"tampered")
    launched.clear()
    r = c.post("/api/update/apply", json={})
    assert r.status_code == 502 and "checksum" in r.json()["detail"]
    assert launched == []
