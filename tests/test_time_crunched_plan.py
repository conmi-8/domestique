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



# ── Budget first (B6', B2', B9, B8') ─────────────────────────────────────────
# The rider QA found it with: CTL 6.5, 264 W / 98 kg, 4.5 h, weekdays 0.5-0.75 h,
# weekends 1.5 h. Built rides-first, week 1 was 112 TSS against a 58 budget.
USER_DMH = {0: 0.5, 1: 0.75, 2: 0.5, 3: 0.75, 4: 0.5, 5: 1.5, 6: 1.5}


@functools.lru_cache(maxsize=None)
def _low_plan(mode, hours=4.5, ctl=6.5, chronic=45.0, dmh=tuple(sorted(USER_DMH.items()))):
    assert tp.clock.today() == MONDAY
    goal = _goal(hours=hours, weeks=12, dmh=dict(dmh))
    goal.plan_mode = mode
    return tp.generate_plan(goal, athlete=ATHLETE, current_ctl=ctl,
                            recent_weekly_tss=chronic)[1]


@pytest.mark.parametrize("mode", ("auto", "fixed_core", "time_crunched"))
def test_b6_budget_first_every_week_within_its_budget(mode):
    """B6': every week of the low-fitness rider at most 1.05 x its budget and
    under the ACWR danger line, in every mode the rider can pick."""
    weeks = _low_plan(mode)
    over = [(w.week_num, sum(s.tss_estimate or 0 for s in _rides(w)), w.tss_target)
            for w in weeks if w.phase != "taper"
            and sum(s.tss_estimate or 0 for s in _rides(w)) > 1.05 * w.tss_target + 1]
    assert not over, over
    assert not pi.check_acwr(weeks, 45.0, tolerance=1.0, limit=pi.ACWR_DANGER)


def test_b6_budget_first_rides_shrink_before_z5_goes():
    """B6': a tight budget keeps its Z5 day, shorter, and rides the rest at
    tempo/sweetspot -- not two long sessions over the budget."""
    for w in _low_plan("time_crunched")[:6]:
        rides = _rides(w)
        assert sum(1 for s in rides if tp._session_is_z5(s)) >= 1, w.week_num
        assert {s.session_type for s in rides if not tp._session_is_z5(s)} <= {
            "tempo", "sweetspot", "ftp_test"}, [(s.day, s.session_type) for s in rides]


def test_b2_two_hours_rides_twice_with_a_z5_day():
    """B2': 2 h a week over 5+ free days is at least two rides and one Z5
    day every full week."""
    for w in _low_plan("time_crunched", hours=2.0, ctl=40.0, chronic=280.0):
        if (w.end - w.start).days < 6:
            continue
        rides = _rides(w)
        assert len(rides) >= 2, (w.week_num, len(rides))
        assert any(tp._session_is_z5(s) for s in rides), w.week_num


def test_b2_more_shorter_rides_over_fewer_longer_ones():
    """B2': a budget that holds six rides of 20+ minutes rides six days."""
    layout = tp._tc_layout({o: 90 for o in range(7) if o != 0}, 300.0, 270.0,
                           MONDAY, week_num=3)
    assert len(layout) == 6 and all(m >= tp._RIDE_FLOOR_MIN for _t, m in layout.values())
    assert sum(m for _t, m in layout.values()) <= 270


def test_b9_the_budget_picks_the_intensity():
    """B9: a tight budget rides tempo/sweetspot; one its minutes cannot hold
    at that intensity climbs to threshold/over-unders."""
    caps = {o: 45 for o in range(1, 7)}
    low = tp._tc_layout(caps, 120.0, 270.0, MONDAY, week_num=1)
    high = tp._tc_layout(caps, 380.0, 270.0, MONDAY, week_num=1)
    assert {t for t, _m in low.values()} <= {"vo2max", "tempo", "sweetspot"}, low
    assert {t for t, _m in high.values()} & {"threshold", "overunder"}, high


def test_b9_the_non_z5_days_rotate():
    """B9: the non-Z5 days of a week are not all one type, and no more than
    three weeks in a row are the same week."""
    caps = {o: 60 for o in range(1, 7)}
    weeks = [tp._tc_layout(caps, 260.0, 270.0, MONDAY, week_num=n) for n in range(1, 9)]
    for lay in weeks:
        nz = [t for t, _m in lay.values() if t != "vo2max"]
        assert len(set(nz)) >= 2, lay
    shapes = [tuple(sorted(lay.items())) for lay in weeks]
    assert not any(shapes[i] == shapes[i + 1] == shapes[i + 2] == shapes[i + 3]
                   for i in range(len(shapes) - 3))


def test_b2_the_z5_days_keep_off_last_weeks():
    """B2: a Z5 day the Sunday before blocks Monday from being one."""
    caps = {0: 60, 2: 60, 4: 60, 5: 60, 6: 60}
    free = tp._tc_layout(caps, 300.0, 300.0, MONDAY, week_num=1)
    # and never two Z5 days in a row within the week: Tue and Wed come first
    # in the Z5 order, Tue and Fri is the answer.
    two = tp._tc_layout({1: 60, 2: 60, 4: 60, 5: 60}, 300.0, 240.0, MONDAY, week_num=1)
    assert sorted(o for o, (st, _m) in two.items() if st == "vo2max") == [1, 4], two
    blocked = tp._tc_layout(caps, 300.0, 300.0, MONDAY, week_num=1,
                            blocked={MONDAY - timedelta(days=1)})
    assert free[0][0] == "vo2max"          # Monday would be a Z5 day ...
    assert blocked[0][0] != "vo2max", blocked   # ... but not the day after one


def test_b8_a_z5_slot_serves_z5_content():
    """B8': every vo2max session is served a file whose content is Z5, no
    other session one whose content is, and none an endurance file."""
    plans = [_weeks("probe"), _weeks("6h"), _low_plan("time_crunched"),
             _low_plan("time_crunched", hours=2.0, ctl=40.0, chronic=280.0)]
    for weeks in plans:
        for w in weeks:
            for s in _rides(w):
                if s.session_type == "ftp_test" or not _training(s):
                    continue
                cc = tp._content_class_for_zwo(s.zwo_file or "")
                assert (s.session_type == "vo2max") == (cc in tp._Z5_CONTENT_CLASSES), (
                    s.day, s.session_type, s.zwo_file, cc)
                # ... and a Z3/Z4 day is not served an endurance file (B1).
                assert cc not in tp._TC_EASY_CONTENT, (s.day, s.session_type, s.zwo_file, cc)


def test_a7_the_eve_of_a_general_goal_stays_z3_plus(freeze_clock):
    """A7: reforecast's eve guard eases the days before the target date with
    a short tempo ride in a time-crunched plan, never endurance."""
    goal = _goal(hours=3.0, weeks=12, goal_type="general")
    weeks = tp.generate_plan(goal, athlete=ATHLETE, **PLANNER_PIN_ARGS)[1]
    today = MONDAY + timedelta(days=16)
    freeze_clock(today)
    plan = {"goal": tp.goal_to_dict(goal), "weeks": [tp.week_to_dict(w) for w in weeks]}
    plan, _n, _info = tp.reforecast_dict(plan, today_iso=today.isoformat())
    _assert_crunched([tp.week_from_dict(w) for w in plan["weeks"]], today)


def test_b7_a_refit_keeps_back_to_back_hard_days(freeze_clock):
    """B7: refit of a six-ride week (hard days back to back by design) keeps
    every remaining ride Z3+ and its two Z5 days: the 48 h easing, the HIT cap
    and the re-owing passes are not the style's rules. The stored week's
    remaining rides are a few minutes off what the builder lays out, so the
    refit re-splices them."""
    weeks = copy.deepcopy(_weeks("6h"))
    wk = next(w for w in weeks if w.start >= MONDAY + timedelta(weeks=2)
              and len(_rides(w)) >= 6)
    first = min(_rides(wk), key=lambda s: s.day)
    first.status = "missed"
    today = first.day + timedelta(days=1)
    for s in _rides(wk):
        if s.day >= today:
            s.duration_min += 3
    freeze_clock(today)
    _weeks_out, info = tp.refit_remaining_week(GOALS["6h"](), weeks, today)
    assert info.get("refit_days"), info
    _assert_crunched([wk], today)
    left = sorted((s for s in _rides(wk) if s.day >= today), key=lambda s: s.day)
    assert len(left) >= 4, [(s.day, s.session_type) for s in left]
    assert any(b.day - a.day == timedelta(days=1) for a, b in zip(left, left[1:]))


def _settle_week(types):
    """A week Mon.. with ``types`` (None = rest), matched from the library."""
    lib = tp.load_workout_library()
    sessions = []
    for i, st in enumerate(types):
        d = MONDAY + timedelta(days=7 + i)
        s = tp.PlannedSession(day=d, day_name=d.strftime("%a"), session_type="rest",
                              duration_min=0, tss_estimate=0, description="")
        if st:
            s.duration_min, s.tss_estimate = 45, 60
            tp._tc_retype(s, st, lib, "test")
        sessions.append(s)
    return lib, tp.PlannedWeek(week_num=2, start=MONDAY + timedelta(days=7),
                               end=MONDAY + timedelta(days=13), phase="build1",
                               tss_target=400.0, is_stepback=False, sessions=sessions)


def test_b9_an_extra_z5_day_is_demoted_below_threshold():
    """B2/B9: a third Z5 day in a four-ride week is demoted, to sweetspot or
    tempo -- never to threshold by default."""
    lib, week = _settle_week(["vo2max", None, "vo2max", None, "vo2max", "tempo", None])
    tp._settle_time_crunched([week], lib)
    rides = _rides(week)
    assert sum(1 for s in rides if tp._session_is_z5(s)) == 2
    assert {s.session_type for s in rides if not tp._session_is_z5(s)} <= {"tempo", "sweetspot"}


def test_b5_a_week_ridden_every_day_gets_a_rest_day():
    """B5: a rebuild that fills all seven days (a comeback ramp) leaves the
    time-crunched week with a rest day."""
    lib, week = _settle_week(["tempo", "vo2max", "tempo", "sweetspot", "vo2max",
                              "tempo", "sweetspot"])
    tp._settle_time_crunched([week], lib)
    assert len(_rides(week)) == 6


def test_b2_back_to_back_z5_days_are_split():
    """B2: Z5 on Tuesday and Wednesday (a pass left them so) -- one of them
    moves, the week keeps two Z5 days, never on consecutive days."""
    lib, week = _settle_week(["tempo", "vo2max", "vo2max", "tempo", None, "sweetspot", None])
    tp._settle_time_crunched([week], lib)
    z5 = sorted(s.day for s in _rides(week) if tp._session_is_z5(s))
    assert len(z5) == 2 and (z5[1] - z5[0]).days >= 2, z5


# ── After QA2: the edit settle (A7'), the budget from below (B10), more rides
# (B2''), VO2 content (B8''), the FTP rest day and the last row (T2) ────────
def _rows(weeks, since=None):
    return [w for w in weeks if w.tss_target and w.phase != "taper"
            and (since is None or w.end >= since)]


def _tss(w):
    return sum(float(s.tss_estimate or 0) for s in _rides(w))


EDIT_CASES = {
    # rider, goal, CTL, chronic: QA2's rf2.py (user65 broke B6' x4.3, ev6 B5)
    "user65": (dict(hours=4.5, dmh=USER_DMH), 6.5, 45.0),
    "ev6": (dict(hours=6.0, goal_type="event", event_km=100.0, event_climb_m=1000.0,
                 event_name="ev", dmh={d: 1.5 for d in range(7)}), 40.0, 280.0),
}


@pytest.mark.parametrize("case", EDIT_CASES)
def test_a7p_an_availability_edit_is_settled_like_generation(case, freeze_clock):
    """A7': the rider sets 1.5 h on each of the next seven days. Those hours
    are caps, and every future week comes out of the same budget-first settle
    as generation: at most 1.05 x its budget, a rest day (B5), two Z5 days
    never back to back in a week of four or more rides (B2), the ACWR danger
    line not crossed."""
    kw, ctl, chronic = EDIT_CASES[case]
    goal = _goal(weeks=12, **kw)
    weeks = tp.generate_plan(goal, athlete=ATHLETE, current_ctl=ctl,
                             recent_weekly_tss=chronic)[1]
    today = MONDAY + timedelta(days=16)
    freeze_clock(today)
    days = [(today + timedelta(days=i)).isoformat() for i in range(1, 8)]
    plan = {"goal": tp.goal_to_dict(goal), "weeks": [tp.week_to_dict(w) for w in weeks]}
    plan, _n, _info = tp.reforecast_dict(plan, today_iso=today.isoformat(),
                                         availability_overrides={d: 1.5 for d in days},
                                         explicit_days=set(days))
    assert plan["goal"]["plan_mode"] == "time_crunched"
    weeks = [tp.week_from_dict(w) for w in plan["weeks"]]
    for w in _rows(weeks, since=today):
        rides = _rides(w)
        shape = [(s.day, s.session_type, s.duration_min, s.tss_estimate) for s in rides]
        assert _tss(w) <= 1.05 * w.tss_target + 1, (w.week_num, w.tss_target, shape)
        if (w.end - w.start).days + 1 == 7:
            assert len(rides) <= 6, (w.week_num, shape)
        z5 = sorted(s.day for s in rides if tp._session_is_z5(s))
        if len(rides) >= 4:
            assert len(z5) == 2, (w.week_num, shape)
        assert all((b - a).days >= 2 for a, b in zip(z5, z5[1:])), (w.week_num, shape)
        assert not [s for s in rides if s.session_type in EASY and _training(s)], shape
    assert not pi.check_acwr(weeks, chronic, tolerance=1.0, limit=pi.ACWR_DANGER)


def test_b10_the_users_weeks_reach_their_budget(freeze_clock):
    """B10: the rider's real plan (QA2 real.py: from Wednesday 2026-11-18,
    0.75 h weekdays, 1.5 h weekends, 4.5 h, CTL 6.5) and the low-fitness
    plan above deliver at least 0.85 x every full week's budget -- the FTP
    test week included, whose test outgrew its day and shed a ride -- and no
    row (the last two-day one included, T2) more than 1.05 x."""
    start = date(2026, 11, 18)
    freeze_clock(start)
    goal = tp.Goal(goal_type="ftp", target_date=start + timedelta(weeks=12),
                   hours_per_week=4.5, max_weekday_hours=0.75, max_weekend_hours=1.5,
                   available_days=list(range(7)), rest_days=[], plan_mode="time_crunched")
    real = tp.generate_plan(goal, athlete=ATHLETE, current_ctl=6.5, recent_weekly_tss=45.0)[1]
    freeze_clock(MONDAY)
    for weeks in (real, _low_plan("time_crunched")):
        rows = [(w.week_num, (w.end - w.start).days + 1, round(_tss(w)), round(w.tss_target))
                for w in _rows(weeks)]
        assert not [r for r in rows if r[1] == 7 and r[2] < 0.85 * r[3]], rows
        assert not [r for r in rows if r[2] > 1.05 * r[3] + 0.5], rows
        assert any(s.session_type == "ftp_test" for w in weeks for s in w.sessions)


def test_b6p_the_layout_never_lays_out_more_than_its_load():
    """B6': whatever the budget, the builder's own week carries no more than
    1.05 x it (not trimmed back by a later rail)."""
    caps = {o: 90 for o in range(1, 7)}
    for tss in range(25, 420, 15):
        layout = tp._tc_layout(caps, float(tss), 270.0, MONDAY, week_num=2)
        got = sum(m * tp.TSS_PER_HOUR[t] / 60 for t, m in layout.values())
        assert layout and got <= 1.05 * tss + 0.5, (tss, got, layout)


def test_b2pp_a_third_and_fourth_ride_beat_a_longer_z5():
    """B2'': the 3rd/4th ride of 20+ minutes is added when it fits, the Z5
    ride shortened toward 20 minutes: QA2's weeks 3 (70 TSS) and 6 (96)."""
    caps = {1: 45, 2: 45, 3: 45, 4: 45, 5: 90, 6: 90}
    three = tp._tc_layout(caps, 70.0, 270.0, MONDAY, week_num=3)
    four = tp._tc_layout(caps, 96.0, 270.0, MONDAY, week_num=6)
    assert len(three) == 3 and [t for t, _m in three.values()].count("vo2max") == 1, three
    assert len(four) == 4 and [t for t, _m in four.values()].count("vo2max") == 2, four
    assert all(m >= tp._RIDE_FLOOR_MIN for _t, m in {**three, **four}.values())


def test_b2pp_a_z5_ride_starts_at_30_minutes_when_no_ride_is_lost():
    """B2'': where the budget holds the same rides with 30-minute Z5 days,
    they are 30 minutes; the library's Z5 files are mostly 25-35."""
    layout = tp._tc_layout({1: 45, 3: 45, 5: 45, 6: 45}, 125.0, 180.0, MONDAY, week_num=1)
    z5 = [m for t, m in layout.values() if t == "vo2max"]
    assert len(layout) == 4 and len(z5) == 2 and min(z5) >= tp._TC_Z5_MIN, layout


def test_b8pp_a_vo2max_slot_is_served_vo2_content():
    """B8'': every vo2max session is VO2 work (vo2max, ladder, short
    intervals), not the anaerobic/neuromuscular files."""
    for weeks in (_weeks("probe"), _weeks("6h"), _low_plan("time_crunched")):
        bad = [(s.day, s.zwo_file) for w in weeks for s in _rides(w)
               if s.session_type == "vo2max"
               and tp._content_class_for_zwo(s.zwo_file or "") not in tp._VO2_CONTENT_CLASSES]
        assert not bad, bad


def test_b8pp_anaerobic_files_serve_only_when_no_vo2_file_does():
    """B8'/B8'': with no VO2 file in the library a vo2max slot falls back to
    the other Z5 files -- never to a Z3/Z4 one."""
    lib = tp.load_workout_library()
    by_cc = {}
    for r in lib:
        by_cc.setdefault(tp._content_class_for_zwo(r.get("File") or ""), r)
    small = [by_cc["anaerobic"], by_cc["sweet_spot"], by_cc["tempo"]]
    assert tp._tc_library(small, True) == [by_cc["anaerobic"]]
    assert tp._tc_library(small + [by_cc["vo2max"]], True) == [by_cc["vo2max"]]


def test_b4_the_settle_rests_the_day_before_an_ftp_test():
    """B4: a ride the day before an FTP test (an edit put it there) rests;
    the test is not ridden on tired legs."""
    lib, week = _settle_week(["tempo", "sweetspot", None, "tempo", None, "tempo", None])
    week.sessions[2].session_type = "ftp_test"
    week.sessions[2].duration_min, week.sessions[2].tss_estimate = 45, 52
    tp._settle_time_crunched([week], lib)
    assert week.sessions[1].session_type == "rest"
    assert week.sessions[2].session_type == "ftp_test"


def test_b6p_the_builder_alone_is_budget_first(monkeypatch):
    """B6'/B10 at the source: with the final re-layout and the minute hold
    switched off, the week builder still lays every full week before the
    FTP test at 0.85-1.05 x its budget and within hours_per_week -- it reads
    the week's budget and minutes, not the phase label."""
    monkeypatch.setattr(tp, "_tc_hold_budget", lambda *a, **k: set())
    monkeypatch.setattr(tp, "hold_week_minutes", lambda *a, **k: None)
    goal = _goal(hours=4.5, weeks=12, dmh=USER_DMH)
    weeks = tp.generate_plan(goal, athlete=ATHLETE, current_ctl=6.5,
                             recent_weekly_tss=45.0)[1]
    rows = []
    for w in weeks:
        if any(s.session_type == "ftp_test" for s in w.sessions):
            break
        if (w.end - w.start).days + 1 == 7:
            rows.append((w.week_num, round(_tss(w)), round(w.tss_target),
                         sum(s.duration_min for s in _rides(w))))
    assert len(rows) >= 4, rows
    assert all(0.85 * t <= got <= 1.05 * t + 0.5 and m <= 270 for _n, got, t, m in rows), rows


def test_b6_the_acwr_rail_holds_a_time_crunched_week():
    """B6: the final-plan ACWR rail trims a time-crunched week over 1.3 x the
    load carried, its Z5 days shortened, not dropped."""
    lib, week = _settle_week(["tempo", "vo2max", "sweetspot", None, "vo2max", "tempo", None])
    for s in _rides(week):
        s.duration_min, s.tss_estimate = 60, 80
    tp._hold_acwr_on_plan([week], 100.0, crunched=True)
    assert _tss(week) <= 1.3 * 100 * 1.05 + 1, [(s.session_type, s.duration_min) for s in _rides(week)]
    assert sum(1 for s in _rides(week) if s.session_type == "vo2max") == 2


# ── After QA3: the floor ride (A6'), rest days after an edit (B11), B2 after
# the rails (B12), the week's budget at the builder (T3) ────────────────────
def _real_goal(start, weeks=12, **kw):
    base = dict(goal_type="ftp", target_date=start + timedelta(weeks=weeks),
                hours_per_week=4.5, max_weekday_hours=0.75, max_weekend_hours=1.5,
                available_days=list(range(7)), rest_days=[], plan_mode="time_crunched")
    base.update(kw)
    return tp.Goal(**base)


@pytest.mark.parametrize("ctl", (1.0, 2.0, 3.0))
def test_a6p_a_low_fitness_plan_rides_every_week_and_builds(ctl, freeze_clock):
    """A6': at CTL 1-3 (a returning rider) even a 20-min tempo is over 1.05 x
    the first budgets. The week rides it anyway -- never a zero-ride week --
    and the budgets after it build instead of decaying to nothing (QA3 X1:
    0 rides in all 13 rows, targets 19 -> 0)."""
    start = date(2026, 11, 18)
    freeze_clock(start)
    weeks = tp.generate_plan(_real_goal(start), athlete=ATHLETE, current_ctl=ctl,
                             recent_weekly_tss=ctl * 7)[1]
    rows = [(w.week_num, (w.end - w.start).days + 1, len(_rides(w)), round(w.tss_target))
            for w in weeks]
    assert not [r for r in rows[:-1] if r[2] == 0], rows
    full = [r for r in rows if r[1] == 7]
    assert max(r[3] for r in full) >= 2 * full[0][3], rows


def test_a6p_the_layout_rides_its_floor_unless_the_week_rides_already():
    """A6' at the source: a budget a 20-min tempo overshoots still rides it;
    a week already holding a ride (an FTP test), or the plan's short last
    row, rests instead."""
    caps = {1: 45, 5: 90}
    assert tp._tc_layout(caps, 8.0, 135.0, MONDAY, week_num=1) == {5: ("tempo", 20)}
    assert tp._tc_layout(caps, 8.0, 135.0, MONDAY, week_num=1, fixed_rides=1) == {}
    assert tp._tc_layout(caps, 8.0, 135.0, MONDAY, week_num=1, may_rest=True) == {}


def test_b11_rider_rest_days_stay_rest_after_an_edit(freeze_clock):
    """B11: the rider's own rest weekdays (Mon, Fri) are never ridden after
    a reforecast re-lays the edited weeks -- only a rest day the rider set
    hours on rides."""
    goal = _real_goal(MONDAY, weeks=10, available_days=[1, 2, 3, 5, 6], rest_days=[0, 4])
    weeks = tp.generate_plan(goal, athlete=ATHLETE, current_ctl=30.0,
                             recent_weekly_tss=210.0)[1]
    today = MONDAY + timedelta(days=16)
    freeze_clock(today)
    days = [(today + timedelta(days=i)).isoformat() for i in range(1, 8)
            if (today + timedelta(days=i)).weekday() not in (0, 4)]
    plan = {"goal": tp.goal_to_dict(goal), "weeks": [tp.week_to_dict(w) for w in weeks]}
    plan, _n, _info = tp.reforecast_dict(plan, today_iso=today.isoformat(),
                                         availability_overrides={d: 1.5 for d in days},
                                         explicit_days=set(days))
    weeks = [tp.week_from_dict(w) for w in plan["weeks"]]
    ridden = [(s.day, s.session_type) for w in weeks for s in _rides(w)
              if s.day > today and s.day.weekday() in (0, 4) and _training(s)]
    assert not ridden, ridden
    assert any(_rides(w) for w in weeks if w.start > today)


def test_b12_b2_holds_after_the_acwr_rail(freeze_clock):
    """B12: a week the ACWR rail cuts (here after a one-ride FTP-test week:
    2 h, 1.5/2.5 h caps, CTL 6.5) is laid out again from its allowance, so
    it still rides 1 Z5 in 2-3 rides and 2 in 4+ (QA3 S3: vo2, vo2, tempo)."""
    start = date(2026, 11, 18)
    freeze_clock(start)
    goal = _real_goal(start, weeks=8, hours_per_week=2.0, max_weekday_hours=1.5,
                      max_weekend_hours=2.5)
    weeks = tp.generate_plan(goal, athlete=ATHLETE, current_ctl=6.5,
                             recent_weekly_tss=45.5)[1]
    wrong = []
    for w in _rows(weeks):
        rides = [s for s in _rides(w) if _training(s)]
        want = 2 if len(rides) >= 4 else 1 if len(rides) >= 2 else None
        got = sum(1 for s in rides if tp._session_is_z5(s))
        if want is not None and got != want:
            wrong.append((w.week_num, [(s.session_type, s.duration_min) for s in rides]))
    assert not wrong, wrong
    assert not pi.check_acwr(weeks, 45.5)


def test_t3_the_builder_lays_out_the_budget_it_is_given():
    """T3/B10: the builder (through propose_week) sizes the week from the
    budget it is handed -- net of rides done, so not ctx.tss_target -- and
    not from the context's gross target."""
    from types import SimpleNamespace
    goal = _goal()
    phase = tp.Phase(name="build1", start=MONDAY, end=MONDAY + timedelta(days=6),
                     weeks=1, focus="", weekly_tss_target=300, z2_pct=80,
                     hit_per_week=2, session_types=[])
    ctx = tp.week_context(goal, phase, 3, MONDAY, is_stepback=False, week_in_phase=0,
                          tss_target=300.0)
    state = tp.week_plan.PlanState(library=tp.load_workout_library())
    for out in (tp.propose_week(ctx, state, SimpleNamespace(tss_per_week=60)),
                tp.time_crunched_week(ctx, state, SimpleNamespace(tss_per_week=60))):
        load = sum(float(s.tss_estimate or 0) for s in out if s.session_type != "rest")
        assert 0.85 * 60 <= load <= 1.05 * 60 + 0.5, [(s.session_type, s.duration_min) for s in out]
