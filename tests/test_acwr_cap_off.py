"""Manual switch (Settings tab, athlete.json "acwr_cap_off_until"): lifts the
ACWR ceiling (and the gap regen's recovery ramp) until the date given.

A rider at CTL ~70 who stops for two weeks comes back to a ceiling of 1.3 x
a 28-day load that counted every day off as zero. With the switch set the
week is bounded by the other guards only; unset, the plan is as before.
"""
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import clock
import profile_manager
import ride_storage
import training_planner as tp

TODAY = date(2026, 3, 2)            # a Monday, the return day
_PATTERN = {0: 0, 1: 60, 2: 90, 3: 60, 4: 40, 5: 140, 6: 100}   # ~490 TSS a week


def _history(off_days: int) -> list:
    rides, d = [], TODAY - timedelta(days=200)
    while d < TODAY:
        if d < TODAY - timedelta(days=off_days) and _PATTERN[d.weekday()]:
            rides.append({"started_at": f"{d.isoformat()}T08:00:00",
                          "date": d.isoformat(), "tss": _PATTERN[d.weekday()]})
        d += timedelta(days=1)
    return rides


def _ctl(rides) -> float:
    per = {r["date"]: r["tss"] for r in rides}
    ctl, d = 0.0, TODAY - timedelta(days=200)
    while d < TODAY:
        ctl += (per.get(d.isoformat(), 0) - ctl) / 42
        d += timedelta(days=1)
    return ctl


def _goal() -> tp.Goal:
    return tp.Goal(goal_type="ftp", plan_weeks=12, hours_per_week=10.0,
                   max_weekday_hours=1.5, max_weekend_hours=3.5,
                   available_days=[1, 2, 3, 4, 5, 6], rest_days=[0])


@pytest.fixture
def prefs(monkeypatch):
    """The stored setting, as a dict the test writes "acwr_cap_off_until" into."""
    clock.freeze(TODAY)
    p: dict = {}
    monkeypatch.setattr(
        profile_manager.ProfileManager, "get",
        classmethod(lambda cls: SimpleNamespace(
            acwr_cap_off_until=p.get("acwr_cap_off_until"))))
    yield p
    clock.unfreeze()


def _generate(rides):
    _, weeks = tp.generate_plan(
        _goal(), current_ctl=_ctl(rides),
        recent_weekly_tss=ride_storage.chronic_weekly_tss(rides, today=TODAY),
        activities=rides)
    return [w for w in weeks if w.end >= TODAY]


@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("2026-03-15", date(2026, 3, 15)),
    ("2026-03-01", None),            # already past
    ("next week", None),             # not a date
])
def test_reading_the_setting(prefs, value, expected):
    if value is not None:
        prefs["acwr_cap_off_until"] = value
    assert tp.acwr_cap_off_until() == expected


def test_ceiling_lifted_only_until_the_date():
    until = TODAY + timedelta(days=6)
    ramp = tp.LoadRamp(50, 180, cap_off_until=until)
    assert ramp.budget("base", on=TODAY) == 7 * (50 + tp.LOAD_K_DEFAULT)
    assert ramp.budget("base", on=TODAY + timedelta(days=7)) == round(1.3 * 180)
    assert tp.LoadRamp(50, 180).budget("base", on=TODAY) == round(1.3 * 180)


def test_return_after_two_weeks_off(prefs):
    rides = _history(14)
    capped = _generate(rides)[0]
    prefs["acwr_cap_off_until"] = (TODAY + timedelta(days=13)).isoformat()
    lifted = _generate(rides)[0]
    assert capped.tss_target < 0.5 * 490          # the decayed ceiling
    assert lifted.tss_target > capped.tss_target


def test_gap_regen_lays_no_recovery_ramp(prefs):
    rides = _history(14)
    clock.freeze(TODAY - timedelta(days=21))
    before = [r for r in rides if r["date"] < (TODAY - timedelta(days=21)).isoformat()]
    _, old = tp.generate_plan(
        _goal(), current_ctl=70.0,
        recent_weekly_tss=ride_storage.chronic_weekly_tss(
            before, today=TODAY - timedelta(days=21)),
        activities=before)
    clock.freeze(TODAY)
    ridden = {r["date"] for r in rides}
    for w in old:
        for s in w.sessions:
            if s.day < TODAY and s.session_type != "rest":
                s.status = "done" if s.day.isoformat() in ridden else "missed"
    chronic = ride_storage.chronic_weekly_tss(rides, today=TODAY)

    _, _, info = tp.regenerate_from_today(_goal(), old, _ctl(rides),
                                          activities=rides, recent_weekly_tss=chronic)
    assert info["recovery_ramp_weeks"] > 0        # unset: as before

    prefs["acwr_cap_off_until"] = (TODAY + timedelta(days=13)).isoformat()
    _, weeks, info = tp.regenerate_from_today(_goal(), old, _ctl(rides),
                                              activities=rides, recent_weekly_tss=chronic)
    assert info["recovery_ramp_weeks"] == 0
    first = next(w for w in weeks if w.start <= TODAY <= w.end)
    assert first.phase not in ("recon", "recovery_ramp")


def test_unset_or_expired_changes_nothing(prefs):
    rides = _history(14)
    unset = repr(_generate(rides))
    prefs["acwr_cap_off_until"] = (TODAY - timedelta(days=1)).isoformat()
    assert repr(_generate(rides)) == unset


# ── the Settings tab ────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    old = profile_manager.ProfileManager._instance
    profile_manager.ProfileManager._instance = None
    pm = profile_manager.ProfileManager.get()
    pm.switch(pm.create_profile("Rider"))
    clock.freeze(TODAY)
    try:
        yield TestClient(app_module.app), pm
    finally:
        clock.unfreeze()
        profile_manager.ProfileManager._instance = old


def test_settings_set_read_and_clear(client):
    c, pm = client
    assert c.post("/api/settings", json={"acwr_cap_off_until": "2026-03-15"}).status_code == 200
    assert pm.acwr_cap_off_until == "2026-03-15"
    assert tp.acwr_cap_off_until() == date(2026, 3, 15)
    assert c.post("/api/settings", json={"acwr_cap_off_until": ""}).status_code == 200
    assert pm.acwr_cap_off_until is None
    assert tp.acwr_cap_off_until() is None


@pytest.mark.parametrize("bad", ["2026-03-01", "next week"])
def test_settings_refuse_a_past_or_bad_date(client, bad):
    c, pm = client
    assert c.post("/api/settings", json={"acwr_cap_off_until": bad}).status_code == 400
    assert pm.acwr_cap_off_until is None
