"""The time-crunched plan style, plan_mode "time_crunched" (MASTER_DECISIONS B).

Short and hard: every ride Z3+, two Z5 days in a week of four or more rides,
never back to back, no stepback weeks, at least one rest day, the load rails
still on, and the mode carried through every rebuild like fixed_core.

Reverting time_crunched_week (propose_week back to the sampler for this mode)
fails B1 and B7 below; checked on test_b1_every_ride_is_z3_or_harder.
"""
import copy
import functools
from datetime import date, timedelta

import pytest

from conftest import PLANNER_PIN_ARGS
import plan_invariants as pi
import training_planner as tp

MONDAY = date(2026, 11, 23)
DMH = {0: 0.75, 1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75, 5: 1.5, 6: 1.5}
ATHLETE = {"ftp": 264, "weight_kg": 98}
EASY = {"z2", "long_z2", "recovery", "endurance"}
Z3_PLUS = {"tempo", "sweetspot", "threshold", "overunder", "vo2max", "ftp_test"}


def _goal(hours=4.5, weeks=10, goal_type="ftp", dmh=None, **kw):
    dmh = dict(DMH if dmh is None else dmh)
    return tp.Goal(goal_type=goal_type, target_date=MONDAY + timedelta(weeks=weeks),
                   hours_per_week=hours, max_weekday_hours=max(dmh[d] for d in range(5)),
                   max_weekend_hours=max(dmh[5], dmh[6]),
                   available_days=list(range(7)), rest_days=[], daily_max_hours=dmh,
                   plan_mode="time_crunched", **kw)


GOALS = {
    "probe": lambda: _goal(),
    "6h": lambda: _goal(hours=6.0, weeks=14, dmh={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0,
                                                    4: 1.0, 5: 2.0, 6: 2.0}),
    "event": lambda: _goal(hours=6.0, weeks=12, goal_type="event", event_km=120.0,
                           event_climb_m=1200.0, event_name="tc event",
                           dmh={d: 1.25 for d in range(7)}),
}


@functools.lru_cache(maxsize=None)
def _plan(name):
    """Callers freeze the clock on MONDAY first (freeze_clock)."""
    assert tp.clock.today() == MONDAY
    return tp.generate_plan(GOALS[name](), athlete=ATHLETE, **PLANNER_PIN_ARGS)


def _weeks(name):
    return _plan(name)[1]


def _rides(w):
    return [s for s in w.sessions if s.session_type != "rest" and (s.duration_min or 0) > 0]


def _training(s):
    return not (getattr(s, "is_race", False) or getattr(s, "is_opener", False))


@pytest.fixture(autouse=True)
def _monday(freeze_clock):
    freeze_clock(MONDAY)


@pytest.mark.parametrize("name", GOALS)
def test_b1_every_ride_is_z3_or_harder(name):
    """B1: no z2, long_z2 or recovery ride -- every training ride is tempo,
    sweetspot, threshold, over-under or vo2max (an FTP test is a test)."""
    bad = [(s.day, s.session_type) for w in _weeks(name) for s in _rides(w)
           if _training(s) and s.session_type not in Z3_PLUS]
    assert not bad, bad


@pytest.mark.parametrize("name", GOALS)
def test_b2_two_z5_days_a_week_never_back_to_back(name):
    """B2: exactly 2 Z5 sessions in a week of 4+ rides, 1 in a week of 2-3,
    and no two Z5 days in a row anywhere in the plan."""
    wrong = []
    for w in _weeks(name):
        rides = [s for s in _rides(w) if _training(s)]
        want = 2 if len(rides) >= 4 else 1 if len(rides) >= 2 else None
        got = sum(1 for s in rides if tp._session_is_z5(s))
        if want is not None and got != want:
            wrong.append((w.week_num, len(rides), got))
    assert not wrong, wrong
    z5 = sorted(s.day for w in _weeks(name) for s in _rides(w) if tp._session_is_z5(s))
    assert all((b - a).days >= 2 for a, b in zip(z5, z5[1:])), z5


def test_b2_the_week_builder_spaces_its_z5_days():
    """B2 at the source: the week builder itself puts the two Z5 days apart."""
    goal = _goal()
    phase = tp.Phase(name="build1", start=MONDAY, end=MONDAY + timedelta(days=6),
                     weeks=1, focus="", weekly_tss_target=300, z2_pct=80,
                     hit_per_week=2, session_types=[])
    ctx = tp.week_context(goal, phase, 3, MONDAY, is_stepback=False, week_in_phase=0)
    out = tp.time_crunched_week(ctx, tp.week_plan.PlanState(library=tp.load_workout_library()))
    z5 = [s.day for s in out if tp._session_is_z5(s)]
    assert len(z5) == 2 and abs((z5[1] - z5[0]).days) >= 2, z5
    assert not [s for s in out if s.session_type in EASY]


@pytest.mark.parametrize("name", GOALS)
def test_b3_only_z5_counts_toward_the_hard_cap(name):
    """B3: sweetspot/threshold/tempo/over-unders are not demoted by the weekly
    HIT cap -- weeks carry more "HIT"-typed days than the cap of 2 while
    never more than 2 Z5 days."""
    weeks = _weeks(name)
    assert max(sum(1 for s in _rides(w) if tp._session_is_z5(s)) for w in weeks) <= 2
    assert max(tp._week_hit_count(w) for w in weeks) >= 3


@pytest.mark.parametrize("name", GOALS)
def test_b3_the_70_percent_z1_floor_does_not_apply(name):
    """B3: a time-crunched week is not held to 70% easy -- its training
    minutes are all above endurance."""
    for w in _weeks(name):
        rides = [s for s in _rides(w) if _training(s)]
        mins = sum(s.duration_min for s in rides)
        easy = sum(s.duration_min for s in rides if s.session_type in EASY)
        assert mins == 0 or easy / mins < 0.30, (w.week_num, easy, mins)


@pytest.mark.parametrize("name", ("probe", "6h"))
def test_b4_no_stepback_weeks(name):
    """B4: no week is a stepback."""
    assert not [w.week_num for w in _weeks(name) if w.is_stepback]


def test_b4_an_event_keeps_its_taper():
    """B4: the event goal keeps its race-specific taper, still no stepback."""
    weeks = _weeks("event")
    assert any(w.phase == "taper" for w in weeks)
    assert not [w.week_num for w in weeks if w.is_stepback]


@pytest.mark.parametrize("name", GOALS)
def test_b4_an_ftp_test_follows_a_rest_day(name):
    """B4: the day before a scheduled FTP test is rest, not an unload week."""
    by_day = {s.day: s for w in _weeks(name) for s in w.sessions}
    tests = [d for d, s in by_day.items() if s.session_type == "ftp_test"]
    for d in tests:
        prev = by_day.get(d - timedelta(days=1))
        assert prev is None or prev.session_type == "rest", (d, prev.session_type)


def test_b4_stepback_due_is_never_true_for_the_style():
    """B4 at the source: the 3:1 rhythm never asks a time-crunched plan to unload."""
    load = [tp.PlannedWeek(week_num=i, start=MONDAY + timedelta(weeks=i),
                           end=MONDAY + timedelta(weeks=i, days=6), phase="build1",
                           tss_target=300, is_stepback=False, sessions=[])
            for i in range(3)]
    end = MONDAY + timedelta(weeks=3, days=6)
    assert tp.stepback_due(load, "build1", end) is True
    assert tp.stepback_due(load, "build1", end, goal=_goal()) is False


@pytest.mark.parametrize("name", GOALS)
def test_b5_at_least_one_rest_day_a_week(name):
    """B5: every full week keeps a rest day; the caps of A still hold."""
    for w in _weeks(name):
        if (w.end - w.start).days + 1 < 7:
            continue
        assert len(_rides(w)) <= 6, w.week_num
        assert sum(s.duration_min for s in _rides(w)
                   if not getattr(s, "is_race", False)) <= GOALS[name]().hours_per_week * 60


@pytest.mark.parametrize("name", ("probe", "6h"))
def test_b6_load_rails_still_apply(name):
    """B6: the volume ceiling and the ACWR danger line hold for the style."""
    weeks = _weeks(name)
    over = [(w.week_num, sum(s.tss_estimate or 0 for s in _rides(w)), w.tss_target)
            for w in weeks if w.phase != "taper"
            and sum(s.tss_estimate or 0 for s in _rides(w)) > 1.05 * w.tss_target]
    assert not over, over
    assert not pi.check_acwr(weeks, PLANNER_PIN_ARGS["recent_weekly_tss"],
                             limit=pi.ACWR_DANGER)


def _assert_crunched(weeks, since):
    bad = [(s.day, s.session_type) for w in weeks for s in _rides(w)
           if s.day >= since and _training(s) and s.session_type not in Z3_PLUS]
    assert not bad, bad
    assert not [w.week_num for w in weeks if w.is_stepback and w.end >= since]


def test_b7_the_mode_survives_the_goal_round_trip():
    """B7: plan_mode is carried in the stored goal, like fixed_core."""
    assert tp.goal_from_dict(tp.goal_to_dict(_goal())).plan_mode == "time_crunched"


def test_b7_survives_regenerate(freeze_clock):
    """B7: regenerate_from_today rebuilds a time-crunched plan as one."""
    weeks = copy.deepcopy(_weeks("6h"))
    today = MONDAY + timedelta(weeks=3, days=2)
    freeze_clock(today)
    _ph, new, _info = tp.regenerate_from_today(
        _goal(hours=6.0, weeks=14, dmh=GOALS["6h"]().daily_max_hours), weeks,
        PLANNER_PIN_ARGS["current_ctl"], athlete=ATHLETE,
        recent_weekly_tss=PLANNER_PIN_ARGS["recent_weekly_tss"])
    _assert_crunched(new, today)


def test_b7_survives_the_weekly_recalc(freeze_clock):
    """B7: recalculate_plan (the weekly recalc) keeps the style."""
    weeks = copy.deepcopy(_weeks("6h"))
    today = MONDAY + timedelta(weeks=4)
    freeze_clock(today)
    _ph, new, _info = tp.recalculate_plan(
        GOALS["6h"](), weeks, PLANNER_PIN_ARGS["current_ctl"], athlete=ATHLETE,
        recent_weekly_tss=PLANNER_PIN_ARGS["recent_weekly_tss"])
    _assert_crunched(new, today)


def test_b7_survives_refit(freeze_clock):
    """B7: a missed Z5 day refits the rest of the week as time-crunched."""
    weeks = copy.deepcopy(_weeks("6h"))
    wk = next(w for w in weeks if w.start >= MONDAY + timedelta(weeks=2)
              and any(tp._session_is_z5(s) and s.day.weekday() <= 2 for s in w.sessions))
    missed = next(s for s in wk.sessions if tp._session_is_z5(s) and s.day.weekday() <= 2)
    missed.status = "missed"
    today = missed.day + timedelta(days=1)
    freeze_clock(today)
    tp.refit_remaining_week(GOALS["6h"](), weeks, today)
    _assert_crunched([wk], today)
    rest = sorted(s.day for s in wk.sessions if s.day >= today and tp._session_is_z5(s))
    assert all((b - a).days >= 2 for a, b in zip(rest, rest[1:])), rest


def test_b7_survives_reforecast(freeze_clock):
    """B7: reforecast_dict (every sync) leaves the stored plan_mode and the
    style alone."""
    weeks = copy.deepcopy(_weeks("6h"))
    plan = {"goal": tp.goal_to_dict(GOALS["6h"]()),
            "weeks": [tp.week_to_dict(w) for w in weeks]}
    today = MONDAY + timedelta(weeks=2)
    freeze_clock(today)
    plan, _n, _info = tp.reforecast_dict(plan, today_iso=today.isoformat())
    assert plan["goal"]["plan_mode"] == "time_crunched"
    # Reforecast is unchanged (A5): its event-eve guard still eases the last
    # EVENT_EVE_EASY_DAYS before the target date to an easy opener.
    eve = GOALS["6h"]().target_date - timedelta(days=tp.EVENT_EVE_EASY_DAYS)
    weeks = [tp.week_from_dict(w) for w in plan["weeks"]]
    for w in weeks:
        w.sessions = [s for s in w.sessions if s.day < eve]
    _assert_crunched(weeks, today)
