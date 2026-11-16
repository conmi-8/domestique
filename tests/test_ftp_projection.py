"""Projected FTP card: pure model (src/ftp_projection.py), personal
narrowing, the /api/plan/ftp-projection endpoint and the plan-screen card.

Each test cites the MASTER_DECISIONS_projected_ftp contract item (#1-#22)
it covers.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app as app_module
import clock
import ftp_projection as fp

TODAY = date(2026, 9, 14)  # a Monday
DASHBOARD = Path(__file__).resolve().parent.parent / "src" / "templates" / "dashboard.html"

# wave0_literature.md §6: the only PMIDs the card may cite.
VERIFIED_PMIDS = {
    "1435180", "3988668", "11719894", "20056848", "21812820", "24382021",
    "24550842", "23264537", "27300278", "8933495", "9134360", "10331896",
    "31977120", "23134196", "3529283", "26243014", "11772161", "39788807",
    "19346973", "25357188", "33348618", "11016988", "28095061", "28133739",
    "29863593", "39888556", "30733142", "27182424", "6511559", "3902770",
    "27476326", "25663672", "29357481", "25823596", "27967274", "11286357",
    "23428482", "26182438", "34965513", "30676826", "36791017",
}


def _week(start: date, num: int, sessions=None, stepback=False, tss_target=470):
    if sessions is None:
        types = [("threshold", 90), ("vo2max", 80), ("z2", 100), ("z2", 100), ("z2", 100)]
        offsets = [1, 3, 2, 5, 6]
        sessions = [{"day": (start + timedelta(days=o)).isoformat(),
                     "session_type": t, "tss_estimate": tss, "zwo_file": ""}
                    for (t, tss), o in zip(types, offsets)]
    return {"week_num": num, "start": start.isoformat(),
            "end": (start + timedelta(days=6)).isoformat(), "phase": "build1",
            "tss_target": tss_target, "is_stepback": stepback, "sessions": sessions}


def _plan(n_weeks=12, start=TODAY, distribution="auto", **kw):
    return {"goal": {"type": "ftp", "distribution": distribution},
            "weeks": [_week(start + timedelta(weeks=i), i + 1, **kw) for i in range(n_weeks)]}


def _gain(**kw):
    args = dict(ftp=250, weight_kg=75, ctl=60, weeks=10, hit_per_week=2, load_ratio=1.0)
    args.update(kw)
    return fp.project_ftp(**args)["gain_pct"]


# ── Model ──────────────────────────────────────────────────────────────────

class ModelTests(unittest.TestCase):

    def test_tier_is_more_trained_signal_contract_1(self):
        """#1: tier = max(W/kg tier, CTL tier); cut-offs 2.5/3.2/4.0/4.8 and 25/50/80/110."""
        self.assertEqual(fp.tier_for(180, 75, 20), 0)     # 2.4 W/kg, CTL 20
        self.assertEqual(fp.tier_for(180, 75, 60), 2)     # CTL wins
        self.assertEqual(fp.tier_for(330, 75, 20), 3)     # 4.4 W/kg wins
        self.assertEqual(fp.tier_for(240, 75, 50), 2)     # 3.2 / 50 boundaries go up
        self.assertEqual(fp.tier_for(400, 75, 120), 4)
        self.assertEqual(fp.tier_for(250, 75, None), 2)   # unknown CTL: W/kg only

    def test_female_wkg_shift_contract_1(self):
        """#1: sex F shifts W/kg cut-offs by -0.5."""
        self.assertEqual(fp.tier_for(210, 70, None), 1)          # 3.0 W/kg
        self.assertEqual(fp.tier_for(210, 70, None, "F"), 2)     # 3.0 >= 2.7
        self.assertEqual(fp.tier_for(210, 70, None, "M"), 1)

    def test_base_gains_at_reference_contract_2(self):
        """#2: 10 wk, 2 HIT, maintained load -> the base table per tier."""
        cases = [((150, 75, 10), (5, 12, 20)), ((210, 75, 30), (3, 7, 11)),
                 ((250, 75, 60), (2, 5, 8)), ((330, 75, 90), (1, 3, 6)),
                 ((400, 75, 120), (0, 1.5, 3))]
        for (ftp, kg, ctl), want in cases:
            g = _gain(ftp=ftp, weight_kg=kg, ctl=ctl)
            got = (g["conservative"], g["central"], g["optimistic"])
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=6)

    def test_duration_factor_contract_3(self):
        """#3: D(10)=1.00, D(4)~0.55, D(16)~1.21; capped at 16 (optimistic 20)."""
        self.assertAlmostEqual(fp.duration_factor(10), 1.0, places=9)
        self.assertAlmostEqual(fp.duration_factor(4), 0.55, delta=0.01)
        self.assertAlmostEqual(fp.duration_factor(16), 1.21, delta=0.01)
        self.assertEqual(fp.duration_factor(30), fp.duration_factor(16))
        self.assertEqual(fp.duration_factor(30, 20), fp.duration_factor(20, 20))
        g = _gain(weeks=30)
        g16 = _gain(weeks=16)
        self.assertEqual(g["central"], g16["central"])
        self.assertGreater(g["optimistic"], g16["optimistic"])

    def test_remaining_weeks_from_today_contract_4(self):
        """#4: a plan in progress projects only what is left; a future plan from its start."""
        plan = _plan(12, start=TODAY - timedelta(weeks=4))
        self.assertAlmostEqual(fp.plan_inputs(plan, TODAY)["weeks"], 8.0)
        mid = fp.plan_inputs(plan, TODAY + timedelta(days=3))["weeks"]
        self.assertAlmostEqual(mid, 8 - 3 / 7)
        future = _plan(12, start=TODAY + timedelta(weeks=2))
        self.assertAlmostEqual(fp.plan_inputs(future, TODAY)["weeks"], 12.0)
        self.assertIsNone(fp.plan_inputs(plan, TODAY + timedelta(weeks=9)))

    def test_hit_counting_contract_5(self):
        """#5: HIT/wk over remaining non-stepback weeks; ftp_test, past and missed sessions excluded."""
        plan = _plan(4)
        plan["weeks"][1]["is_stepback"] = True
        plan["weeks"][1]["sessions"] = []          # stepback weeks never count
        plan["weeks"][2]["sessions"].append(
            {"day": (TODAY + timedelta(days=18)).isoformat(), "session_type": "ftp_test",
             "tss_estimate": 60})
        plan["weeks"][3]["sessions"][0]["status"] = "missed"
        inp = fp.plan_inputs(plan, TODAY)
        self.assertAlmostEqual(inp["hit_per_week"], (2 + 2 + 1) / 3)
        # Past sessions excluded: from Wednesday, week 1 only has its Thursday VO2.
        inp = fp.plan_inputs(_plan(1), TODAY + timedelta(days=2))
        self.assertAlmostEqual(inp["hit_per_week"], 1 / (5 / 7))

    def test_sweetspot_only_counts_as_one_contract_5(self):
        """#5: if every HIT session is sweet spot -> counted as 1/wk."""
        plan = _plan(4)
        for wk in plan["weeks"]:
            for s in wk["sessions"]:
                if s["session_type"] in ("threshold", "vo2max"):
                    s["session_type"] = "sweetspot"
        inp = fp.plan_inputs(plan, TODAY)
        self.assertTrue(inp["sweetspot_only"])
        self.assertEqual(inp["hit_per_week"], 1.0)
        self.assertFalse(fp.plan_inputs(_plan(4), TODAY)["sweetspot_only"])

    def test_hit_multiplier_contract_6(self):
        """#6: 0 -> 0.8/0.5, 1 -> 0.9/0.8, >=2 -> 1.0; more than 2 never rewarded."""
        self.assertEqual(fp.hit_multiplier(0, 0), 0.8)
        self.assertEqual(fp.hit_multiplier(0, 2), 0.5)
        self.assertEqual(fp.hit_multiplier(1, 0), 0.9)
        self.assertEqual(fp.hit_multiplier(1, 4), 0.8)
        self.assertEqual(fp.hit_multiplier(2, 3), 1.0)
        self.assertEqual(fp.hit_multiplier(4, 1), 1.0)
        self.assertEqual(_gain(hit_per_week=4), _gain(hit_per_week=2))

    def test_load_multiplier_contract_7(self):
        """#7: r bands <0.7 / 0.7-0.9 / 0.9-1.15 / >1.15 with tier-dependent values."""
        self.assertEqual(fp.load_multiplier(0.5, 1), 0.3)
        self.assertEqual(fp.load_multiplier(0.8, 1), 0.7)
        self.assertEqual(fp.load_multiplier(0.8, 2), 0.8)
        self.assertEqual(fp.load_multiplier(1.0, 3), 1.0)
        self.assertEqual(fp.load_multiplier(1.15, 3), 1.0)
        self.assertEqual(fp.load_multiplier(1.2, 0), 1.25)
        self.assertEqual(fp.load_multiplier(1.2, 4), 1.1)
        self.assertEqual(fp.load_multiplier(None, 4), 1.0)

    def test_load_ratio_from_plan_contract_7(self):
        """#7: r = mean planned weekly TSS of the remaining weeks / (CTL x 7)."""
        res = fp.build_projection(_plan(12), 250, 75, 60, None, TODAY)
        self.assertAlmostEqual(res["load_ratio"], round(470 / 420, 2))

    def test_reduction_floor_contract_8(self):
        """#8: r < 0.7 -> conservative = -2% x min(w,5) x (0.7-r)/0.7 (may be negative)."""
        g = _gain(weeks=8, load_ratio=0.35)
        self.assertAlmostEqual(g["conservative"], -2 * 5 * 0.5)
        g = _gain(weeks=3, load_ratio=0.0)
        self.assertAlmostEqual(g["conservative"], -6.0)
        self.assertGreaterEqual(_gain(load_ratio=0.7)["conservative"], 0)

    def test_distribution_label_ignored_contract_9(self):
        """#9: polarized / pyramidal / threshold label never changes the result."""
        out = [fp.build_projection(_plan(12, distribution=d), 250, 75, 60, None, TODAY)
               for d in ("polarized", "pyramidal", "threshold", "auto")]
        for o in out[1:]:
            self.assertEqual(o, out[0])

    def test_worked_example_contract_10(self):
        """#10: 75 kg, FTP 250, CTL 60, 12 wk, 2 HIT, r=1.12 -> 2.2/5.5/8.7 %, 255/265/270 W."""
        g = fp.project_ftp(250, 75, 60, 12, 2, 1.12)["gain_pct"]
        # The contract rounds D(12) to 1.09; exact D(12) = 1.089.
        self.assertAlmostEqual(g["conservative"], 2.2, delta=0.1)
        self.assertAlmostEqual(g["central"], 5.5, delta=0.1)
        self.assertAlmostEqual(g["optimistic"], 8.7, delta=0.1)
        res = fp.build_projection(_plan(12), 250, 75, 60, None, TODAY)
        self.assertEqual(res["tier"], 2)
        self.assertEqual(res["weeks"], 12.0)
        self.assertEqual(res["hit_per_week"], 2.0)
        self.assertEqual(res["watts"], {"conservative": 255, "central": 265, "optimistic": 270})

    def test_band_ordered_property_contract_11(self):
        """#11: cons <= central <= opt over a grid of inputs."""
        for ftp in (150, 220, 280, 340, 420):
            for ctl in (None, 15, 40, 70, 100, 130):
                for weeks in (1, 4, 10, 16, 24):
                    for hit in (0, 0.5, 1, 2, 3):
                        for r in (None, 0.0, 0.5, 0.8, 1.0, 1.3):
                            for sex in (None, "F"):
                                g = fp.project_ftp(ftp, 72, ctl, weeks, hit, r, sex)["gain_pct"]
                                self.assertLessEqual(g["conservative"], g["central"])
                                self.assertLessEqual(g["central"], g["optimistic"])

    def test_within_noise_flag_contract_12(self):
        """#12: central gain < 3% -> within_noise."""
        hi = fp.build_projection(_plan(12), 250, 75, 60, None, TODAY)
        self.assertFalse(hi["within_noise"])
        lo = fp.build_projection(_plan(12), 400, 75, 120, None, TODAY)  # T4
        self.assertTrue(lo["within_noise"])

    def test_held_flat_past_16_weeks_contract_13(self):
        """#13: plan longer than 16 weeks -> held_flat."""
        self.assertTrue(fp.build_projection(_plan(20), 250, 75, 60, None, TODAY)["held_flat"])
        self.assertFalse(fp.build_projection(_plan(16), 250, 75, 60, None, TODAY)["held_flat"])


# ── Personal narrowing ─────────────────────────────────────────────────────

def _riding(start: date, end: date, tss=60.0, skip=()):
    out, d = {}, start
    while d <= end:
        out[d.isoformat()] = 0.0 if d in skip else tss
        d += timedelta(days=1)
    return out


def _hist(points, kind="tested_ramp"):
    return [{"date": d.isoformat() + "T08:00:00Z", "ftp": f, "type": kind} for d, f in points]


class PersonalTests(unittest.TestCase):

    def setUp(self):
        self.d0 = TODAY - timedelta(weeks=40)
        self.daily = _riding(self.d0 - timedelta(days=60), TODAY)
        self.ctl = {k: 60.0 for k in self.daily}

    def _blocks(self, hist, daily=None):
        return fp.past_blocks(hist, self.daily if daily is None else daily,
                              self.ctl, 75, None, TODAY)

    def test_block_rules_contract_14(self):
        """#14: pairs 6-16 wk apart, tested_*/eftp_auto only, <= 3 years, no gap > 14 d."""
        d0 = self.d0
        pts = [(d0, 250), (d0 + timedelta(weeks=8), 262), (d0 + timedelta(weeks=16), 270)]
        self.assertEqual(len(self._blocks(_hist(pts))), 2)
        self.assertEqual(len(self._blocks(_hist(pts, "eftp_auto"))), 2)
        self.assertEqual(self._blocks(_hist(pts, "manual")), [])
        self.assertEqual(self._blocks(_hist(pts, "eftp_icu")), [])
        # 4 weeks and 20 weeks apart: not blocks.
        self.assertEqual(self._blocks(_hist([(d0, 250), (d0 + timedelta(weeks=4), 260)])), [])
        self.assertEqual(self._blocks(_hist([(d0, 250), (d0 + timedelta(weeks=20), 260)])), [])
        # Older than 3 years.
        old = TODAY - timedelta(days=3 * 365 + 30)
        self.assertEqual(self._blocks(_hist([(old, 250), (old + timedelta(weeks=8), 262)])), [])
        # A 15-day riding gap inside the block.
        gap = {d0 + timedelta(days=20 + i) for i in range(15)}
        daily = _riding(self.d0 - timedelta(days=60), TODAY, skip=gap)
        self.assertEqual(len(self._blocks(_hist(pts), daily)), 1)

    def test_shrinkage_formula_contract_15(self):
        """#15: rho_hat = (n tau^2 mean rho + sigma^2) / (n tau^2 + sigma^2), tau 0.5."""
        blocks = [{"rho": 1.6, "sigma": 0.4}, {"rho": 1.2, "sigma": 0.4}]
        model = fp.project_ftp(250, 75, 60, 12, 2, 1.0)
        res = fp.personalise(model, blocks)
        n, t2, s2 = 2, 0.25, 0.16
        want = (n * t2 * 1.4 + s2) / (n * t2 + s2)
        self.assertAlmostEqual(res["personal_ratio"], round(want, 3))
        self.assertAlmostEqual(res["gain_pct"]["central"], model["gain_pct"]["central"] * want)

    def test_block_rho_and_sigma_contract_15(self):
        """#15: rho = observed / retrospective central; sigma = 2.8% (tested) or 4% (eFTP) / central."""
        d0 = self.d0
        pts = [(d0, 250), (d0 + timedelta(weeks=8), 262)]
        b = self._blocks(_hist(pts))[0]
        self.assertAlmostEqual(b["rho"], (262 / 250 - 1) * 100 / b["model_central_pct"])
        self.assertAlmostEqual(b["sigma"], 2.8 / b["model_central_pct"])
        b = self._blocks(_hist(pts, "eftp_auto"))[0]
        self.assertAlmostEqual(b["sigma"], 4.0 / b["model_central_pct"])

    def test_applied_with_two_blocks_clamped_and_floored_contract_16(self):
        """#16: >=2 blocks; rho clamped to [0.3, 1.7]; band never narrower than test noise."""
        model = fp.project_ftp(150, 75, 10, 12, 2, 1.0)   # T0, wide band
        res = fp.personalise(model, [{"rho": 9.0, "sigma": 0.01}] * 2)
        self.assertTrue(res["personalised"])
        self.assertEqual(res["personal_blocks"], 2)
        self.assertEqual(res["personal_ratio"], 1.7)
        g = res["gain_pct"]
        self.assertGreaterEqual(g["central"] - g["conservative"], 2.8 - 1e-9)
        self.assertGreaterEqual(g["optimistic"] - g["central"], 2.8 - 1e-9)
        self.assertEqual(fp.personalise(model, [{"rho": 0.0, "sigma": 0.01}] * 3)["personal_ratio"], 0.3)
        # Narrowed: with precise history the band is tighter than the model's.
        m = model["gain_pct"]
        self.assertLess(g["optimistic"] - g["conservative"], m["optimistic"] - m["conservative"])

    def test_end_to_end_personalised_flag_contract_16(self):
        """#16: build_projection reports personalised + n blocks from real history."""
        d0 = self.d0
        pts = [(d0, 240), (d0 + timedelta(weeks=8), 255), (d0 + timedelta(weeks=16), 265)]
        res = fp.build_projection(_plan(12), 265, 75, 60, None, TODAY, history=_hist(pts),
                                  daily_tss=self.daily, ctl_by_day=self.ctl)
        self.assertTrue(res["personalised"])
        self.assertEqual(res["personal_blocks"], 2)
        self.assertIn("Hecksteden et al. 2017 (PMID 27967274)", res["model_citations"])

    def test_fewer_than_two_blocks_is_pure_model_contract_17(self):
        """#17: < 2 blocks -> identical to the model output."""
        d0 = self.d0
        one = _hist([(d0, 250), (d0 + timedelta(weeks=8), 280)])
        base = fp.build_projection(_plan(12), 250, 75, 60, None, TODAY)
        res = fp.build_projection(_plan(12), 250, 75, 60, None, TODAY, history=one,
                                  daily_tss=self.daily, ctl_by_day=self.ctl)
        self.assertFalse(res["personalised"])
        self.assertEqual(res["personal_blocks"], 1)
        res["personal_blocks"] = 0
        self.assertEqual(res, base)
        model = fp.project_ftp(250, 75, 60, 12, 2, 1.12)
        self.assertEqual(fp.personalise(model, [{"rho": 2, "sigma": 0.1}])["gain_pct"],
                         model["gain_pct"])


# ── Endpoint ───────────────────────────────────────────────────────────────

def _pm(athlete):
    return SimpleNamespace(_athlete=athlete, ftp=athlete.get("ftp", 200),
                           sex=athlete.get("sex"), ftp_test_history=[])


class EndpointTests(unittest.TestCase):

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.plan_path = self._tmp / "current_plan.json"
        clock.freeze(TODAY)

    def tearDown(self):
        clock.unfreeze()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _get(self, athlete=None, ctl=60.0):
        athlete = {"ftp": 250, "weight_kg": 75} if athlete is None else athlete
        with patch.object(app_module, "_plan_dir", return_value=self._tmp), \
             patch("profile_manager.ProfileManager.get", return_value=_pm(athlete)), \
             patch.object(app_module, "_fitness_state", return_value={"ctl": ctl}):
            res = TestClient(app_module.app).get("/api/plan/ftp-projection")
        self.assertEqual(res.status_code, 200)
        return res.json()

    def _write(self, plan):
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def test_no_plan_unavailable_contract_18(self):
        """#18: no plan -> available false."""
        self.assertEqual(self._get(), {"available": False, "reason": "no_plan"})

    def test_no_ftp_or_weight_unavailable_contract_18(self):
        """#18: no FTP / no weight -> available false with a reason."""
        self._write(_plan(12))
        self.assertEqual(self._get({"weight_kg": 75})["reason"], "no_ftp")
        self.assertEqual(self._get({"ftp": 250})["reason"], "no_weight")

    def test_unknown_ctl_flagged_contract_18(self):
        """#18: unknown CTL -> tier from W/kg only, flagged."""
        self._write(_plan(12))
        d = self._get(ctl=None)
        self.assertTrue(d["available"])
        self.assertTrue(d["ctl_unknown"])
        self.assertIsNone(d["ctl"])
        self.assertIsNone(d["load_ratio"])
        self.assertEqual(d["tier"], 2)  # 3.33 W/kg

    def test_response_shape_and_citations_contract_19(self):
        """#19: inputs, three % and W values, flags, model_citations with verified PMIDs."""
        self._write(_plan(12))
        d = self._get()
        self.assertTrue(d["available"])
        for k in ("tier", "weeks", "hit_per_week", "load_ratio", "ctl", "ftp",
                  "gain_pct", "watts", "within_noise", "held_flat", "personalised",
                  "personal_blocks", "model_citations"):
            self.assertIn(k, d)
        for band in (d["gain_pct"], d["watts"]):
            self.assertEqual(set(band), {"conservative", "central", "optimistic"})
        self.assertEqual(d["watts"], {"conservative": 255, "central": 265, "optimistic": 270})
        pmids = set()
        for c in d["model_citations"]:
            m = re.search(r"PMID (\d+)", c)
            self.assertIsNotNone(m, c)
            pmids.add(m.group(1))
        self.assertTrue(pmids <= VERIFIED_PMIDS, pmids - VERIFIED_PMIDS)
        for c in fp.PERSONAL_CITATIONS:
            self.assertIn(re.search(r"PMID (\d+)", c).group(1), VERIFIED_PMIDS)

    def test_read_only_contract_20(self):
        """#20: the endpoint never writes the plan or the profile."""
        self._write(_plan(12))
        before = self.plan_path.read_bytes()
        athlete = {"ftp": 250, "weight_kg": 75}
        snapshot = dict(athlete)
        self._get(athlete)
        self.assertEqual(self.plan_path.read_bytes(), before)
        self.assertEqual(athlete, snapshot)
        self.assertEqual([p.name for p in self._tmp.iterdir()], ["current_plan.json"])

    def test_malformed_plan_never_500_contract_20(self):
        """#20: a malformed stored plan -> 200 available false."""
        for bad in ("{not json", json.dumps([1, 2]), json.dumps({"weeks": [{"start": "x"}]}),
                    json.dumps({"weeks": [{"start": "2026-09-14", "end": "2026-09-20",
                                           "sessions": [{"day": "2026-09-15", "tss_estimate": "abc"}]}]})):
            self.plan_path.write_text(bad, encoding="utf-8")
            d = self._get()
            self.assertFalse(d["available"], bad)


# ── Card ───────────────────────────────────────────────────────────────────

def _html():
    return DASHBOARD.read_text(encoding="utf-8")


def test_card_outside_legacy_grid_and_loaded_from_plan_tab_contract_21():
    """#21: the card sits outside #legacy-plan-grid (shown in Calendar view)
    and the Plan tab loader calls loadFtpProjection."""
    src = _html()
    card = src.index('id="ftp-projection-card"')
    legacy = src.index('id="legacy-plan-grid"')
    self_overview = src.index("<h3>Plan Overview</h3>")
    assert self_overview < card < legacy
    loader = src[src.index("await runPlanOpenSequence();"):]
    loader = loader[:loader.index("}, settings:")]
    assert "loadFtpProjection();" in loader
    assert "fetch('/api/plan/ftp-projection')" in src


def _render_fn(src):
    start = src.index("function renderFtpProjection(p) {")
    end = src.index("\n}\n", start) + 3
    return src[start:end]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("within_noise", [False, True])
def test_card_text_contract_21(within_noise):
    """#21/#12/#13: '~X W in N weeks (X1-X3 W)', the inputs line, 'Why this number'
    with citations; within-noise and held-flat wording; injected text escaped."""
    payload = fp.build_projection(_plan(20), 250, 75, 60, None, TODAY)
    payload["within_noise"] = within_noise
    payload["tier_label"] = "<b>x</b>"
    harness = """
const el = {innerHTML: ''};
const $ = () => el;
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
""" + _render_fn(_html()) + f"""
renderFtpProjection({json.dumps(payload)});
console.log(el.innerHTML);
"""
    out = subprocess.run(["node", "-e", harness], capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    html = out.stdout
    w = payload["watts"]
    if within_noise:
        assert "likely to hold (within test noise)" in html
        assert f"~{w['central']} W" not in html
    else:
        assert f"~{w['central']} W in 20 weeks" in html
    assert f"({w['conservative']}–{w['optimistic']} W)" in html
    assert "Why this number" in html and "PMID 21812820" in html
    assert "evidence ends at ~16 weeks" in html
    assert "CTL 60" in html and "FTP 250 W" in html
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
