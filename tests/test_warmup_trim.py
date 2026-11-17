"""Warm-up trim + Time-crunched plan-style UI (MASTER_DECISIONS_time_crunched C1-C9, B8).

Pure trim tests run on real library .zwo files; endpoint tests go through
TestClient with the ICU transport mocked at icu_calendar_push._http (the
same stub-HOME harness as tests/test_calendar_push_workout.py). No GUI.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import types
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import db as db_module  # noqa: E402
import app as app_module  # noqa: E402
import training_planner as tp  # noqa: E402
import icu_calendar_push as icp  # noqa: E402
import warmup_trim as wt  # noqa: E402
from structure_fidelity import parse_zwo_text  # noqa: E402
from profile_manager import ProfileManager  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "src" / "workouts"
DASH = REPO / "src" / "templates" / "dashboard.html"
_ENV_KEYS = ("ICU_ATHLETE_ID", "ICU_API_KEY", "ICU_ACCESS_TOKEN")
D1 = (date.today() + timedelta(days=1)).isoformat()

# <Warmup> 1200 s 0.25->0.75, then SteadyState 300 s @0.5, then 300 s @1.25
WARMUP_EL = "anaerobic_1x5min_125pct_60min.zwo"
# <Warmup> 1200 s + SteadyState 900 s @0.65 before IntervalsT (on 1.14)
WARMUP_LONG = "anaerobic_4x5min-5min_114pct_120min.zwo"
# no <Warmup>: SteadyState 300 @0.5, 300 @0.65, then 180 @0.8
LEADING_SS = "anaerobic_3x30s_150pct_60min.zwo"
# no <Warmup>: SteadyState 300 @0.4, 300 @0.6, then 60 @0.8
LEADING_SS2 = "over_under_4x6x2min_85pct_91min_v2.zwo"
# content class ftp_test (filename not used for the decision)
RAMP_TEST = "ftp_test_ramp_ladder21_200pct_35min.zwo"
COGGAN_TEST = "ftp_test_coggan_3x1min-1min_95pct_59min.zwo"


def _text(name):
    return (LIB / name).read_text(encoding="utf-8")


def _steps(text):
    """Top-level <workout> children as raw strings, in order."""
    body = re.search(r"<workout>(.*)</workout>", text, re.S).group(1)
    return [m.group(0) for m in wt._TOKEN.finditer(body)
            if not m.group(0).startswith("<!--")]


def _tss(text):
    """Independent TSS estimate from structure_fidelity's parser."""
    t = 0.0
    for s in parse_zwo_text(text):
        lo = s["lo"] if s["lo"] is not None else 0.5
        hi = s["hi"] if s["hi"] is not None else lo
        t += s["dur_s"] * ((lo + hi) / 2) ** 2
    return t / 36.0


def _dur(text):
    return sum(s["dur_s"] for s in parse_zwo_text(text))


# ---------------------------------------------------------------------------
# C4 / C5 / C6 / C9 — the pure trim on real library files
# ---------------------------------------------------------------------------

class TestPureTrim:
    def test_C4_warmup_element_plus_easy_step_becomes_one_ramp(self):
        src = _text(WARMUP_EL)
        out, info = wt.trim_zwo_text(src, 5)
        assert info is not None and info["warmup_s"] == 1500
        before, after = _steps(src), _steps(out)
        # 2 warm-up steps -> ONE ramp from first power (0.25) to last (0.5)
        assert len(after) == len(before) - 1
        assert after[0] == '<Warmup Duration="300" PowerLow="0.25" PowerHigh="0.5" />'

    def test_C4_leading_steadystate_steps_below_76pct(self):
        out, info = wt.trim_zwo_text(_text(LEADING_SS), 4)
        assert info["warmup_s"] == 600
        assert _steps(out)[0] == '<Warmup Duration="240" PowerLow="0.5" PowerHigh="0.65" />'
        assert 'Power="0.8"' in _steps(out)[1]       # first >=76% step kept
        out2, info2 = wt.trim_zwo_text(_text(LEADING_SS2), 5)
        assert info2["warmup_s"] == 600
        assert _steps(out2)[0] == '<Warmup Duration="300" PowerLow="0.4" PowerHigh="0.6" />'

    def test_C4_warmup_ends_before_intervals(self):
        out, info = wt.trim_zwo_text(_text(WARMUP_LONG), 10)
        assert info["warmup_s"] == 2100
        assert _steps(out)[1].startswith("<IntervalsT")

    @pytest.mark.parametrize("name,n_warm", [
        (WARMUP_EL, 2), (WARMUP_LONG, 2), (LEADING_SS, 2), (LEADING_SS2, 2)])
    def test_C5_everything_after_warmup_identical(self, name, n_warm):
        src = _text(name)
        out, _ = wt.trim_zwo_text(src, 5)
        assert _steps(out)[1:] == _steps(src)[n_warm:]
        # the rest of the document, byte for byte
        tail = src.split(_steps(src)[n_warm], 1)[1]
        assert out.endswith(tail)

    def test_C6_short_enough_warmup_unchanged(self):
        src = _text(WARMUP_EL)                       # 25 min warm-up
        out, info = wt.trim_zwo_text(src, 20)
        assert info is not None                      # 25 > 20 -> trimmed
        src2 = _text(LEADING_SS)                     # 10 min warm-up
        out2, info2 = wt.trim_zwo_text(src2, 10)
        assert info2 is None and out2 == src2
        out3, info3 = wt.trim_zwo_text(src2, None)   # setting off
        assert info3 is None and out3 == src2

    def test_C6_zero_removes_warmup(self):
        src = _text(WARMUP_EL)
        out, info = wt.trim_zwo_text(src, 0)
        assert "<Warmup" not in out
        assert _steps(out) == _steps(src)[2:]
        assert info["new_s"] == info["orig_s"] - 1500

    def test_C9_duration_tss_and_name_follow_trim(self):
        src = _text(WARMUP_EL)
        out, info = wt.trim_zwo_text(src, 5)
        assert _dur(out) == _dur(src) - 1200         # 25 min -> 5 min
        assert _tss(out) < _tss(src)
        assert "<name>Anaerobic 1x5min (40min)</name>" in out \
            or re.search(r"<name>[^<]*\(40min\)</name>", out)
        assert wt.retitle("Anaerobic 60min — 1×5min", info["orig_s"],
                          info["new_s"]) == "Anaerobic 40min — 1×5min"

    def test_whole_library_parses_and_keeps_totals(self):
        # adversarial sweep: every library file stays valid XML and its
        # trimmed total equals the independent parser's total.
        import xml.etree.ElementTree as ET
        n = 0
        for f in sorted(LIB.glob("*.zwo"))[::7]:
            src = f.read_text(encoding="utf-8")
            out, info = wt.trim_zwo_text(src, 5)
            ET.fromstring(out.encode("utf-8"))
            if info:
                n += 1
                assert abs(_dur(out) - info["new_s"]) <= 2, f.name
        assert n > 100

    def test_endurance_ride_keeps_its_main_set(self):
        # no step >=76%: only a <Warmup> element is the warm-up
        src = ('<workout_file><name>Z2 (70min)</name><workout>'
               '<Warmup Duration="600" PowerLow="0.4" PowerHigh="0.65" />'
               '<SteadyState Duration="3600" Power="0.65" /></workout></workout_file>')
        out, info = wt.trim_zwo_text(src, 5)
        assert info["warmup_s"] == 600
        assert '<SteadyState Duration="3600" Power="0.65" />' in out


# ---------------------------------------------------------------------------
# C7 — protocol tests never trimmed (content class / session type)
# ---------------------------------------------------------------------------

class TestC7Tests:
    def test_C7_ftp_test_class_not_trimmed(self):
        pm = types.SimpleNamespace(_athlete={"warmup_max_min": 0})
        for name in (RAMP_TEST, COGGAN_TEST):
            raw = (LIB / name).read_bytes()
            assert wt.trim_zwo_text(raw.decode(), 0)[1] is not None  # would trim
            with mock.patch.object(app_module, "WORKOUT_DIR", LIB):
                out, info = wt.trim_for_export(raw, name, pm)
            assert info is None and out is raw

    def test_C7_session_type_ftp_test_not_trimmed(self):
        pm = types.SimpleNamespace(_athlete={"warmup_max_min": 0})
        raw = (LIB / WARMUP_EL).read_bytes()
        out, info = wt.trim_for_export(raw, WARMUP_EL, pm, session_type="ftp_test")
        assert info is None and out is raw


def test_C1_parse_and_setting_bounds():
    assert wt.parse_max(None) is None and wt.parse_max("") is None
    assert wt.parse_max(0) == 0 and wt.parse_max("20") == 20
    for bad in (21, -1, 2.5, "x", True):
        with pytest.raises(ValueError):
            wt.parse_max(bad)
    assert wt.setting(types.SimpleNamespace(_athlete={"warmup_max_min": 7})) == 7
    assert wt.setting(types.SimpleNamespace(_athlete={})) is None
    assert wt.setting(object()) is None


# ---------------------------------------------------------------------------
# Endpoint harness (pattern: tests/test_calendar_push_workout.py)
# ---------------------------------------------------------------------------

@pytest.fixture()
def stub(tmp_path):
    home = tmp_path
    patcher = mock.patch("pathlib.Path.home", return_value=home)
    patcher.start()
    old_instance = ProfileManager._instance
    old_db_path = db_module.DB_PATH
    old = (app_module.DATA_DIR, app_module.WORKOUT_DIR, app_module.GPX_DIR,
           tp.PLAN_DIR, tp.WORKOUT_DIR)
    old_environ = {k: os.environ.get(k) for k in _ENV_KEYS}
    for attr in _ENV_KEYS:
        try:
            delattr(config, attr)
        except AttributeError:
            pass
    ProfileManager._instance = None
    app_module.DATA_DIR = home / ".domestique"
    pm = ProfileManager.get()
    pid = pm.create_profile("Trimmer")
    pm.switch(pid)
    db_module._sync_stop.clear()

    workouts = home / "wk"
    workouts.mkdir()
    names = (WARMUP_EL, LEADING_SS, RAMP_TEST)
    for n in names:
        (workouts / n).write_bytes((LIB / n).read_bytes())
    full = json.loads((LIB / ".content_classification.json")
                      .read_text(encoding="utf-8"))["classifications"]
    (workouts / ".content_classification.json").write_text(
        json.dumps({"classifications": {n: full[n] for n in names}}),
        encoding="utf-8")
    app_module.WORKOUT_DIR = workouts
    ns = types.SimpleNamespace(home=home, pm=pm, pid=pid, workouts=workouts,
                               client=TestClient(app_module.app),
                               disk={n: (workouts / n).read_bytes() for n in names})
    try:
        yield ns
    finally:
        patcher.stop()
        t = app_module._icu_push_timer
        if t is not None:
            t.cancel()
        app_module._icu_push_timer = None
        tp.post_write_callback = None
        ProfileManager._instance = old_instance
        (app_module.DATA_DIR, app_module.WORKOUT_DIR, app_module.GPX_DIR,
         tp.PLAN_DIR, tp.WORKOUT_DIR) = old
        db_module.set_db_path(old_db_path)
        db_module._sync_stop.clear()
        app_module.clear_cache()
        for k, v in old_environ.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for attr in _ENV_KEYS:
            try:
                delattr(config, attr)
            except AttributeError:
                pass


class FakeICU:
    def __init__(self):
        self.events, self.calls = [], []

    def __call__(self, method, path, payload=None, timeout=30.0):
        self.calls.append((method, path, payload))
        if method == "GET":
            return 200, json.dumps(self.events).encode()
        if method == "POST":
            for ev in payload:
                self.events = [e for e in self.events
                               if e.get("external_id") != ev["external_id"]]
                self.events.append(dict(ev, id=len(self.events) + 1))
            return 200, b"[]"
        if method == "PUT":
            gone = {d.get("external_id") for d in payload}
            self.events = [e for e in self.events if e.get("external_id") not in gone]
            return 200, b"{}"
        raise AssertionError(method)


@pytest.fixture()
def fake(monkeypatch):
    f = FakeICU()
    monkeypatch.setattr(icp, "_http", f)
    return f


def _set(stub, v):
    r = stub.client.post("/api/settings", json={"warmup_max_min": v})
    return r


def _push(stub, **body):
    stub.pm.save_env("i123", "KEY123")
    return stub.client.post("/api/calendar/push-workout", json=body).json()


def _sent(fake):
    ev = [c for c in fake.calls if c[0] == "POST"][-1][2][0]
    return ev, base64.b64decode(ev["file_contents_base64"]).decode("utf-8")


def _write_plan(sessions):
    plan = {"goal": {"type": "ftp"}, "generated": "2026-07-01T00:00:00",
            "weeks": [{"week_num": 1, "start": D1, "end": D1, "phase": "base",
                       "tss_target": 300, "is_stepback": False,
                       "sessions": sessions}]}
    pdir = Path(tp.PLAN_DIR)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "current_plan.json").write_text(json.dumps(plan), encoding="utf-8")


def _sess(zwo, stype="anaerobic"):
    return {"day": D1, "day_name": "X", "session_type": stype,
            "duration_min": 60, "tss_estimate": 50.0, "description": "",
            "zwo_file": zwo, "zwo_name": Path(zwo).stem, "status": "pending"}


class TestC1Settings:
    def test_C1_setting_persists_per_profile(self, stub):
        assert stub.client.get("/api/settings").json()["warmup_max_min"] is None
        assert _set(stub, 5).status_code == 200
        assert stub.client.get("/api/settings").json()["warmup_max_min"] == 5
        athlete = json.loads((stub.pm.active_dir / "athlete.json")
                             .read_text(encoding="utf-8"))
        assert athlete["warmup_max_min"] == 5
        assert _set(stub, None).status_code == 200      # blank = off
        assert stub.client.get("/api/settings").json()["warmup_max_min"] is None
        # another profile does not see it
        _set(stub, 8)
        other = stub.pm.create_profile("Other")
        stub.pm.switch(other)
        assert stub.client.get("/api/settings").json()["warmup_max_min"] is None

    @pytest.mark.parametrize("bad", [21, -1, "abc", 2.5])
    def test_C1_out_of_range_rejected(self, stub, bad):
        assert _set(stub, bad).status_code == 400

    def test_C1_settings_input_in_dashboard(self):
        html = DASH.read_text(encoding="utf-8")
        assert 'id="set-warmup-max"' in html and "Max warm-up (min)" in html
        assert "body.warmup_max_min" in html


class TestC2Send:
    def test_C2_override_beats_setting(self, stub, fake):
        _set(stub, 10)
        r = _push(stub, source="library", zwo_file=WARMUP_EL, date=D1,
                  name="Anaerobic 60min — 1×5min", warmup_max_min=3)
        assert r["ok"] is True
        ev, body = _sent(fake)
        assert _steps(body)[0].startswith('<Warmup Duration="180"')
        # C9: the event title (classification display name) shows the new total
        assert "38min" in ev["name"] and "60min" not in ev["name"]

    def test_C2_blank_uses_setting(self, stub, fake):
        _set(stub, 10)
        for blank in (None, ""):
            _push(stub, source="library", zwo_file=WARMUP_EL, date=D1,
                  name="x", warmup_max_min=blank)
            assert _steps(_sent(fake)[1])[0].startswith('<Warmup Duration="600"')

    @pytest.mark.parametrize("bad", [21, -3, "ten", 1.5])
    def test_C2_bad_value_rejected_no_upsert(self, stub, fake, bad):
        r = _push(stub, source="library", zwo_file=WARMUP_EL, date=D1,
                  name="x", warmup_max_min=bad)
        assert r == {"error": "bad_warmup"}
        assert not any(c[0] == "POST" for c in fake.calls)

    def test_C2_send_input_next_to_button(self):
        html = DASH.read_text(encoding="utf-8")
        assert "_calWarmupInput('calpush-lib-warmup')" in html
        assert "_calWarmupInput('calpush-plan-warmup')" in html
        assert "warmup_max_min: (warmupMin == null) ? null : warmupMin" in html
        # the prefill value is escaped before going into the HTML
        assert 'value="${esc(v)}"' in html


class TestC3Consumers:
    def test_C3_off_by_default_bytes_identical(self, stub, fake):
        _push(stub, source="library", zwo_file=WARMUP_EL, date=D1, name="x")
        assert _sent(fake)[1].encode("utf-8") == stub.disk[WARMUP_EL]

    def test_C3_library_send(self, stub, fake):
        _set(stub, 5)
        _push(stub, source="library", zwo_file=WARMUP_EL, date=D1, name="x")
        assert _steps(_sent(fake)[1])[0] == \
            '<Warmup Duration="300" PowerLow="0.25" PowerHigh="0.5" />'

    def test_C3_planner_send_with_override(self, stub, fake):
        _set(stub, 5)
        _write_plan([_sess(LEADING_SS)])
        r = _push(stub, source="planner", zwo_file=LEADING_SS, date=D1,
                  warmup_max_min=2)
        assert r["ok"] is True
        assert _steps(_sent(fake)[1])[0] == \
            '<Warmup Duration="120" PowerLow="0.5" PowerHigh="0.65" />'

    def test_C3_plan_auto_sync(self, stub, fake):
        _set(stub, 5)
        stub.pm.save_env("i123", "KEY123")
        _write_plan([_sess(LEADING_SS)])
        res = icp.reconcile()
        assert res["pushed"] == 1
        assert _steps(_sent(fake)[1])[0] == \
            '<Warmup Duration="300" PowerLow="0.5" PowerHigh="0.65" />'

    @pytest.mark.parametrize("url", [
        f"/api/download/zwo/{WARMUP_EL}",
        f"/api/download/zwo/anaerobic/{WARMUP_EL}",
        f"/api/workout/download/{WARMUP_EL}"])
    def test_C3_zwo_download(self, stub, url):
        plain = stub.client.get(url)
        assert plain.status_code == 200 and plain.content == stub.disk[WARMUP_EL]
        _set(stub, 5)
        r = stub.client.get(url)
        assert r.status_code == 200
        txt = r.content.decode("utf-8")
        assert _steps(txt)[0] == \
            '<Warmup Duration="300" PowerLow="0.25" PowerHigh="0.5" />'
        assert re.search(r"<name>[^<]*\(40min\)</name>", txt)   # C9

    def test_C3_fit_export(self, stub):
        from fit_tool.fit_file import FitFile
        from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage
        from fit_tool.profile.messages.workout_message import WorkoutMessage

        def fit(url):
            r = stub.client.get(url)
            assert r.status_code == 200, r.text
            recs = [x.message for x in FitFile.from_bytes(r.content).records]
            steps = [m for m in recs if isinstance(m, WorkoutStepMessage)]
            name = next(m.workout_name for m in recs if isinstance(m, WorkoutMessage))
            return sum(float(s.duration_time or 0) for s in steps), name

        url = (f"/api/export/fit-workout?zwo_file={WARMUP_EL}"
               f"&name=Anaerobic%2060min&view=power")
        full, n0 = fit(url)
        _set(stub, 5)
        short, n1 = fit(url)
        assert abs((full - short) - 1200) <= 2           # 25 min -> 5 min
        assert n0 == "Anaerobic 60min" and n1 == "Anaerobic 40min"   # C9

    def test_C3_launcher_fallback_read(self, stub):
        import launcher
        _set(stub, 5)
        api = launcher.JsApi.__new__(launcher.JsApi)
        data = api._read_zwo_serverside(WARMUP_EL).decode("utf-8")
        assert _steps(data)[0].startswith('<Warmup Duration="300"')


class TestC7C8Endpoints:
    def test_C7_ramp_test_send_not_trimmed(self, stub, fake):
        _set(stub, 0)
        _push(stub, source="library", zwo_file=RAMP_TEST, date=D1, name="x")
        assert _sent(fake)[1].encode("utf-8") == stub.disk[RAMP_TEST]
        r = stub.client.get(f"/api/download/zwo/{RAMP_TEST}")
        assert r.content == stub.disk[RAMP_TEST]

    def test_C7_planner_ftp_test_session_not_trimmed(self, stub, fake):
        _set(stub, 0)
        _write_plan([_sess(WARMUP_EL, stype="ftp_test")])
        _push(stub, source="planner", zwo_file=WARMUP_EL, date=D1)
        assert _sent(fake)[1].encode("utf-8") == stub.disk[WARMUP_EL]

    def test_C8_library_files_never_modified(self, stub, fake):
        _set(stub, 0)
        _write_plan([_sess(LEADING_SS)])
        _push(stub, source="library", zwo_file=WARMUP_EL, date=D1, name="x")
        _push(stub, source="planner", zwo_file=LEADING_SS, date=D1)
        stub.client.get(f"/api/download/zwo/{WARMUP_EL}")
        stub.client.get(f"/api/export/fit-workout?zwo_file={WARMUP_EL}&view=power")
        for n, b in stub.disk.items():
            assert (stub.workouts / n).read_bytes() == b, n


# ---------------------------------------------------------------------------
# B8 — Time-crunched plan style in the UI
# ---------------------------------------------------------------------------

class TestB8:
    def test_B8_dropdown_option(self):
        html = DASH.read_text(encoding="utf-8")
        sel = re.search(r'<select id="plan-mode".*?</select>', html, re.S).group(0)
        assert '<option value="time_crunched">Time-crunched (short &amp; hard)</option>' in sel

    def test_B8_popover_and_summary_label(self):
        html = DASH.read_text(encoding="utf-8")
        pop = re.search(r'<div class="info-popover" id="planpop-mode">.*?</div>',
                        html, re.S).group(0)
        assert "Time-crunched" in pop
        assert "time_crunched: 'Time-crunched'" in html

    def test_B8_app_passes_plan_mode_through(self, stub):
        seen = {}

        def spy(*a, **kw):
            seen["plan_mode"] = kw.get("plan_mode")
            raise RuntimeError("stop after capture")
        with mock.patch.object(tp, "Goal", spy):
            try:
                stub.client.post("/api/plan/generate", json={
                    "goal": "ftp", "weeks": 8, "hours_per_week": 5,
                    "plan_mode": "time_crunched"})
            except RuntimeError:
                pass
        assert seen.get("plan_mode") == "time_crunched"
