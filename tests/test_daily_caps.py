"""Per-day hours are caps, not targets, in every plan mode (MASTER_DECISIONS A).

The bug (scratchpad probe caps.py): an FTP goal, every day available, 0.75 h
weekdays and 1.5 h weekends, 4.5 h a week, came out as weeks of 7 rides and
~370 minutes, mostly z2/recovery -- generate_plan put a session on every day
the rider gave hours to and filled each one. ride_day_caps now decides which
days ride from the week's budget, for every week builder.

Reverting the fix (sample_week_workouts/plan_week/expand_blueprint_week back
to "every available day rides", hold_week_minutes a no-op) fails A2, A3 and
A4 below; checked on test_a4_probe_scenario_auto.
"""
import functools
from datetime import date, timedelta

import pytest

from conftest import PLANNER_PIN_ARGS
import training_planner as tp

MONDAY = date(2026, 11, 23)
THURSDAY = MONDAY + timedelta(days=3)
DMH = {0: 0.75, 1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75, 5: 1.5, 6: 1.5}
MODES = ("auto", "fixed_core", "template", "time_crunched")


def _probe_goal(mode="auto", hours=4.5, rest_days=(), dmh=None, start=MONDAY):
    dmh = dict(DMH if dmh is None else dmh)
    return tp.Goal(goal_type="ftp", target_date=start + timedelta(weeks=10),
                   hours_per_week=hours, max_weekday_hours=0.75, max_weekend_hours=1.5,
                   available_days=[d for d in range(7) if d not in rest_days],
                   rest_days=list(rest_days), daily_max_hours=dmh, plan_mode=mode,
                   template_id="polarized_base" if mode == "template" else "")


@functools.lru_cache(maxsize=None)
def _plan(mode="auto", hours=4.5, rest_days=(), today=MONDAY, overrides=None):
    """Callers freeze the clock on ``today`` first (freeze_clock)."""
    assert tp.clock.today() == today
    return tp.generate_plan(
        _probe_goal(mode, hours, rest_days, start=today),
        availability_overrides=dict(overrides) if overrides else None,
        athlete={"ftp": 264, "weight_kg": 98}, **PLANNER_PIN_ARGS)[1]


def _rides(w):
    return [s for s in w.sessions if s.session_type != "rest" and (s.duration_min or 0) > 0]


def _minutes(w):
    return sum(s.duration_min for s in _rides(w))


@pytest.mark.parametrize("mode", MODES)
def test_a1_no_session_exceeds_its_day_cap(mode, freeze_clock):
    """A1: every session fits its day's cap (daily_max_hours here)."""
    freeze_clock(MONDAY)
    bad = [(s.day, s.session_type, s.duration_min)
           for w in _plan(mode) for s in _rides(w)
           if s.duration_min > DMH[s.day.weekday()] * 60]
    assert not bad, bad


def test_a1_a_dated_availability_override_is_the_cap(freeze_clock):
    """A1: an availability override for a date caps that date's session."""
    freeze_clock(MONDAY)
    sat = MONDAY + timedelta(days=12)
    weeks = _plan(overrides=((sat.isoformat(), 0.5),))
    s = next(s for w in weeks for s in w.sessions if s.day == sat)
    assert s.session_type == "rest" or s.duration_min <= 30, (s.session_type, s.duration_min)


@pytest.mark.parametrize("mode", MODES)
def test_a2_week_minutes_never_exceed_hours_per_week(mode, freeze_clock):
    """A2: no week plans more than hours_per_week x 60 minutes."""
    freeze_clock(MONDAY)
    over = [(w.week_num, _minutes(w)) for w in _plan(mode) if _minutes(w) > 4.5 * 60]
    assert not over, over


@pytest.mark.parametrize("mode", MODES)
def test_a2_a_partial_first_week_gets_its_budget_pro_rata(mode, freeze_clock):
    """A2: a plan opened on a Thursday holds its Thu..Sun row to 4/7 of the week."""
    freeze_clock(THURSDAY)
    first = _plan(mode, today=THURSDAY)[0]
    days = (first.end - first.start).days + 1
    assert days == 4
    assert _minutes(first) <= 4.5 * 60 * days / 7, _minutes(first)


@pytest.mark.parametrize("mode", ("auto", "fixed_core", "template"))
def test_a3_the_phase_rest_days_hold_when_the_rider_lists_none(mode, freeze_clock):
    """A3: hours on every day do not mean riding every day -- each full week
    keeps its phase's rest_days_per_week."""
    freeze_clock(MONDAY)
    short = []
    for w in _plan(mode, hours=12.0):     # a budget big enough to ride daily
        if (w.end - w.start).days + 1 < 7:
            continue
        need = (tp._PHASE_SHAPE.get(w.phase) or tp._PHASE_SHAPE["base"]).rest_days_per_week
        if 7 - len(_rides(w)) < need:
            short.append((w.week_num, w.phase, len(_rides(w))))
    assert not short, short


@pytest.mark.parametrize("mode", MODES)
def test_a3_days_the_rider_marked_as_rest_are_rest(mode, freeze_clock):
    """A3: Monday and Friday marked rest never carry a session."""
    freeze_clock(MONDAY)
    ridden = [s.day for w in _plan(mode, hours=12.0, rest_days=(0, 4))
              for s in _rides(w) if s.day.weekday() in (0, 4)]
    assert not ridden, ridden


def test_a3_the_budget_decides_how_many_days_ride():
    """A3: the planner picks the ride days from the week's budget -- 3 h is
    three rides of an hour, the longest caps first."""
    goal = _probe_goal(hours=3.0, dmh={d: 1.0 for d in range(7)})
    caps = tp.ride_day_caps(goal, MONDAY, MONDAY + timedelta(days=6), "build1")
    assert len(caps) == 3
    assert sorted((MONDAY + timedelta(days=o)).weekday() for o in caps) == [1, 3, 5]
    # 4.5 h with 0.75 h weekdays and 1.5 h weekends: both weekend days and
    # Tue/Thu, exactly the week's 270 minutes.
    caps = tp.ride_day_caps(_probe_goal(), MONDAY, MONDAY + timedelta(days=6), "base")
    assert sum(caps.values()) == 270 and len(caps) == 4


def test_a4_probe_scenario_auto(freeze_clock):
    """A4: the probe -- 10 weeks, every day available, 0.75/1.5 h, 4.5 h a
    week: no week has 7 ride days and none exceeds 270 minutes (auto)."""
    freeze_clock(MONDAY)
    weeks = _plan("auto")
    assert len(weeks) >= 10
    assert max(len(_rides(w)) for w in weeks) < 7
    assert max(_minutes(w) for w in weeks) <= 270, [(w.week_num, _minutes(w)) for w in weeks]


def test_a5_reforecast_free_hours_stay_a_ceiling(freeze_clock):
    """A5: reforecast's availability ceiling (9138110d, 8e564678, f770a97a) is
    unchanged: raising free hours never pushes a week past its tss_target.
    Mirrors tests/test_availability_reforecast.py::TestAvailabilityIsACeilingNotATarget,
    which with the rest of that suite is untouched."""
    freeze_clock(MONDAY - timedelta(days=1))
    sessions = [tp.PlannedSession(day=MONDAY, day_name="Mon", session_type="z2",
                                  duration_min=60, tss_estimate=45.0, description="")]
    sessions += [tp.PlannedSession(day=MONDAY + timedelta(days=i),
                                   day_name=(MONDAY + timedelta(days=i)).strftime("%a"),
                                   session_type="rest", duration_min=0, tss_estimate=0.0,
                                   description="")
                 for i in range(1, 7)]
    week = tp.PlannedWeek(week_num=1, start=MONDAY, end=MONDAY + timedelta(days=6),
                          phase="base", tss_target=200.0, is_stepback=False,
                          sessions=sessions)
    overrides = {(MONDAY + timedelta(days=i)).isoformat(): 3.0 for i in range(5)}
    tp.reforecast(tp.Goal(goal_type="general", hours_per_week=15.0), [week],
                  tsb_series={MONDAY + timedelta(days=i): 0.0 for i in range(7)},
                  availability_overrides=overrides)
    assert sum(float(s.tss_estimate or 0) for s in week.sessions) <= 200.0
