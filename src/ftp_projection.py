"""Projected FTP at the end of the stored plan: a literature band, optionally
narrowed by the rider's own past FTP blocks.

Pure: no database, no files, no clock. ``today`` is always a parameter. The
one outside call is ``training_planner._session_is_hit`` (the planner's single
hardness predicate), which may read the cached content classification of a
session's workout file.

The model and every number below are documented in docs/SCIENCE.md
("3b. Projected FTP"). Values marked "judgement" are not fitted to any data.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import training_planner as tp

# ── Tier (judgement; named after De Pauw 2013, PMID 23428482, and McKay 2022,
# PMID 34965513, not derived from their tables). The MORE trained of the two
# signals wins, which is the conservative choice.
WKG_CUTOFFS = (2.5, 3.2, 4.0, 4.8)
CTL_CUTOFFS = (25, 50, 80, 110)
# Women: W/kg bands shift down ~0.5 (judgement; Decroix 2016, PMID 26182438,
# shows lower values per level, exact offset unverified).
WKG_SHIFT_FEMALE = 0.5
TIER_LABELS = ("novice", "recreational", "trained", "well-trained", "elite")

# ── Base gain, % FTP over a 10-week block with ~2 HIT/wk and load maintained:
# (conservative, central, optimistic). Central sits below the trial means
# (small n, lab outcomes, regression to the mean -- Atkinson & Batterham 2015).
BASE_GAIN_PCT = (
    (5.0, 12.0, 20.0),  # T0  Jenkins & Quigley 1992 (+28-31%, n=12), discounted
    (3.0, 7.0, 11.0),   # T1  Seiler 2013 (5.5-11.4% composite, 7 wk)
    (2.0, 5.0, 8.0),    # T2  Sylta 2016 (5-10%, n=63, 12 wk); Ronnestad 2015
    (1.0, 3.0, 6.0),    # T3  Lindsay 1996 (+8.3%, no prior HIT); Westgarth-Taylor 1997
    (0.0, 1.5, 3.0),    # T4  Ronnestad 2020 (20-min +4.7 +/- 4.4%, 3 wk)
)

# ── Duration: saturating, relative to 10 weeks. Shape from
# Scharhag-Rosenberger 2009 (gains front-load) and Cove 2025 (mild rise with
# duration). tau = 8 wk and the 16-wk cap are judgement: no trial in trained
# cyclists runs past ~16 weeks, so the projection is held flat beyond it.
DURATION_TAU_WEEKS = 8.0
DURATION_REF_WEEKS = 10.0
DURATION_CAP_WEEKS = 16.0
DURATION_CAP_OPTIMISTIC_WEEKS = 20.0

# ── HIT sessions/wk multiplier: (T0, T1-T4) at 0 and 1 session; 2+ is the
# reference (Seiler 2013, Ronnestad 2015, Sylta 2016 all use ~2/wk). More than
# 2 is never rewarded (no chronic evidence that 3 beats 2). 0 HIT: Seiler
# low-only, Stoggl 2014 HVT/THR, Laursen & Jenkins 2002. 1 HIT: interpolation.
M_HIT = {0: (0.8, 0.5), 1: (0.9, 0.8), 2: (1.0, 1.0)}

# ── Load ratio r = planned weekly TSS / (CTL x 7). Thresholds and values are
# judgement; direction from Montero & Lundby 2017 (untrained dose-response),
# Sanders 2017 (TSS vs threshold change, r ~ .75) and Cove 2025 (no volume
# effect between studies in trained cyclists) -> only a small bump.
LOAD_REDUCTION_R = 0.7
LOAD_MAINTAIN_LO = 0.9
LOAD_MAINTAIN_HI = 1.15
M_LOAD_REDUCTION = 0.3
M_LOAD_SLIGHT = (0.7, 0.8)    # (T0-T1, T2-T4)
M_LOAD_INCREASE = (1.25, 1.1)  # (T0-T1, T2-T4)
# Detraining: about -2%/wk of threshold power under full cessation, first
# ~5 weeks (Maldonado-Martin 2017: LT2 power -11.5% in 5 wk). Linear scaling
# toward r = 0.7 is judgement.
DETRAIN_PCT_PER_WEEK = 2.0
DETRAIN_MAX_WEEKS = 5.0

# ── Test noise (Hopkins 2001, PMID 11286357: CV ~1.5-3% per test). A change
# score between two tests has SD ~ sqrt(2) x CV.
WITHIN_NOISE_PCT = 3.0
CHANGE_SD_TESTED_PCT = 2.8
CHANGE_SD_EFTP_PCT = 4.0

# ── Personal narrowing (normal-normal shrinkage toward rho = 1; precedent:
# Hecksteden 2017 Bayesian individualisation). tau, the clamp, the 3-year
# horizon and the 14-day gap rule are judgement; >= 2 blocks because single
# response estimates are unstable (Hecksteden 2015/2018).
PERSONAL_TAU = 0.5
PERSONAL_MIN_BLOCKS = 2
PERSONAL_RHO_CLAMP = (0.3, 1.7)
BLOCK_MIN_DAYS = 6 * 7
BLOCK_MAX_DAYS = 16 * 7
HISTORY_MAX_DAYS = 3 * 365
MAX_RIDING_GAP_DAYS = 14
# Past blocks: how many hard sessions the rider did is not known, so the
# retrospective model assumes the 2/wk reference (judgement).
RETRO_HIT_PER_WEEK = 2.0

MODEL_CITATIONS = [
    "Jenkins & Quigley 1992 (PMID 1435180)",
    "Seiler et al. 2013 (PMID 21812820)",
    "Rønnestad et al. 2015 (PMID 24382021)",
    "Sylta et al. 2016 (PMID 27300278)",
    "Lindsay et al. 1996 (PMID 8933495)",
    "Rønnestad et al. 2020 (PMID 31977120)",
    "Scharhag-Rosenberger et al. 2009 (PMID 19346973)",
    "Cove et al. 2025 (PMID 39788807)",
    "Rosenblat et al. 2025 (PMID 39888556)",
    "Montero & Lundby 2017 (PMID 28133739)",
    "Sanders et al. 2017 (PMID 28095061)",
    "Maldonado-Martín et al. 2017 (PMID 27476326)",
    "Hopkins et al. 2001 (PMID 11286357)",
]
PERSONAL_CITATIONS = [
    "Hecksteden et al. 2017 (PMID 27967274)",
    "Hecksteden et al. 2018 (PMID 29357481)",
    "Atkinson & Batterham 2015 (PMID 25823596)",
]

_TESTED = ("tested_coggan_20min", "tested_ramp", "tested_sixty_min", "tested_cp_test")
_NOT_RIDDEN = ("missed", "dismissed")


def _band(value: float, cutoffs) -> int:
    return sum(1 for c in cutoffs if value >= c)


def tier_for(ftp: float, weight_kg: float, ctl: float | None,
             sex: str | None = None) -> int:
    """0..4. The more trained of the W/kg tier and the CTL tier; W/kg only
    when CTL is unknown."""
    shift = WKG_SHIFT_FEMALE if (sex or "").upper() == "F" else 0.0
    t = _band(ftp / weight_kg, [c - shift for c in WKG_CUTOFFS])
    if ctl is not None:
        t = max(t, _band(ctl, CTL_CUTOFFS))
    return t


def duration_factor(weeks: float, cap: float = DURATION_CAP_WEEKS) -> float:
    """D(w) = (1 - e^(-w/8)) / (1 - e^(-10/8)), w capped."""
    w = max(0.0, min(weeks, cap))
    return ((1 - math.exp(-w / DURATION_TAU_WEEKS))
            / (1 - math.exp(-DURATION_REF_WEEKS / DURATION_TAU_WEEKS)))


def hit_multiplier(hit_per_week: float, tier: int) -> float:
    """Table at 0 / 1 / >=2 sessions; a fractional plan average is
    interpolated linearly between the table points (judgement)."""
    col = 0 if tier == 0 else 1
    h = max(0.0, min(hit_per_week, 2.0))
    lo = int(math.floor(h))
    hi = min(lo + 1, 2)
    a, b = M_HIT[lo][col], M_HIT[hi][col]
    return a + (b - a) * (h - lo)


def load_multiplier(r: float | None, tier: int) -> float:
    if r is None:
        return 1.0
    col = 0 if tier <= 1 else 1
    if r < LOAD_REDUCTION_R:
        return M_LOAD_REDUCTION
    if r < LOAD_MAINTAIN_LO:
        return M_LOAD_SLIGHT[col]
    if r <= LOAD_MAINTAIN_HI:
        return 1.0
    return M_LOAD_INCREASE[col]


def _round5(watts: float) -> int:
    return int(math.floor(watts / 5.0 + 0.5) * 5)


def project_ftp(ftp: float, weight_kg: float, ctl: float | None,
                weeks: float, hit_per_week: float, load_ratio: float | None,
                sex: str | None = None) -> dict:
    """The literature band for one block. ``ctl``/``load_ratio`` may be None
    (unknown): the tier then comes from W/kg only and load is neutral."""
    tier = tier_for(ftp, weight_kg, ctl, sex)
    mult = hit_multiplier(hit_per_week, tier) * load_multiplier(load_ratio, tier)
    d = duration_factor(weeks)
    d_opt = duration_factor(weeks, DURATION_CAP_OPTIMISTIC_WEEKS)
    cons, central, opt = BASE_GAIN_PCT[tier]
    gains = [cons * d * mult, central * d * mult, opt * d_opt * mult]
    load_reduced = load_ratio is not None and load_ratio < LOAD_REDUCTION_R
    if load_reduced:
        # Amendment 8b: a plan this far below current load holds at best;
        # the optimistic end keeps the 0.3-scaled gain.
        gains[0] = (-DETRAIN_PCT_PER_WEEK * min(weeks, DETRAIN_MAX_WEEKS)
                    * (LOAD_REDUCTION_R - load_ratio) / LOAD_REDUCTION_R)
        gains[1] = 0.0
    return {
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "gain_pct": {"conservative": gains[0], "central": gains[1],
                     "optimistic": gains[2]},
        "held_flat": weeks > DURATION_CAP_WEEKS,
        "load_reduced": load_reduced,
    }


# ── Plan inputs ────────────────────────────────────────────────────────────

def _is_sweetspot(sess: dict) -> bool:
    cls = tp._content_class_for_zwo(sess.get("zwo_file") or "") if sess.get("zwo_file") else ""
    if cls:
        return cls.startswith("sweet_spot")
    return sess.get("session_type") == "sweetspot"


def plan_inputs(plan: dict, today: date) -> dict | None:
    """Remaining weeks, HIT/wk and mean weekly TSS of a stored plan, counted
    from today (or the plan start, if later). None when nothing is left.

    A week already under way counts as the fraction of it that is left, so a
    half-ridden week does not halve the HIT average.
    """
    weeks = plan.get("weeks") or []
    if not weeks:
        return None
    first = date.fromisoformat(str(weeks[0]["start"])[:10])
    last = date.fromisoformat(str(weeks[-1]["end"])[:10])
    start = max(today, first)
    w = ((last - start).days + 1) / 7.0
    if w <= 0:
        return None
    total_tss = hit = hit_wk = 0.0
    hit_sessions: list[dict] = []
    for wk in weeks:
        ws = date.fromisoformat(str(wk["start"])[:10])
        we = date.fromisoformat(str(wk["end"])[:10])
        frac = ((we - max(ws, start)).days + 1) / 7.0
        if frac <= 0:
            continue
        sessions = [s for s in (wk.get("sessions") or [])
                    if str(s.get("day") or "")[:10] >= start.isoformat()
                    and s.get("status") not in _NOT_RIDDEN]
        if wk.get("sessions"):
            total_tss += sum(float(s.get("tss_estimate") or 0) for s in sessions)
        else:
            total_tss += float(wk.get("tss_target") or 0) * frac
        if wk.get("is_stepback"):
            continue
        hit_wk += frac
        for s in sessions:
            if s.get("session_type") != "ftp_test" and tp._session_is_hit(s):
                hit += 1
                hit_sessions.append(s)
    hit_per_week = hit / hit_wk if hit_wk else 0.0
    sweetspot_only = bool(hit_sessions) and all(_is_sweetspot(s) for s in hit_sessions)
    if sweetspot_only:
        # Sub-threshold-only HIT is treated as one session a week (judgement;
        # Neal 2013 THR +2%, Stoggl 2014 THR ns).
        hit_per_week = 1.0
    return {"weeks": w, "hit_per_week": hit_per_week,
            "weekly_tss": total_tss / w, "sweetspot_only": sweetspot_only}


# ── Personal narrowing ─────────────────────────────────────────────────────

def _ctl_on(ctl_by_day: dict, day: date) -> float | None:
    for back in range(8):
        v = ctl_by_day.get((day - timedelta(days=back)).isoformat())
        if v:
            return float(v)
    return None


def past_blocks(history: list[dict], daily_tss: dict, ctl_by_day: dict,
                weight_kg: float, sex: str | None, today: date) -> list[dict]:
    """Qualifying past blocks: consecutive, non-overlapping pairs of FTP
    entries 6-16 weeks apart, tested_* or eftp_auto, within 3 years, with no
    riding gap > 14 rest days. Each carries its response ratio rho and its noise
    sigma on the rho scale."""
    oldest = today - timedelta(days=HISTORY_MAX_DAYS)
    pts = []
    for e in history:
        kind = e.get("type") or ""
        if kind not in _TESTED and kind != "eftp_auto":
            continue
        try:
            d = date.fromisoformat(str(e.get("date"))[:10])
            f = float(e.get("ftp"))
        except (TypeError, ValueError):
            continue
        if oldest <= d <= today and f > 0:
            pts.append((d, f, kind in _TESTED))
    pts.sort(key=lambda p: p[0])
    blocks = []
    i = 0
    while i < len(pts):
        j = next((k for k in range(i + 1, len(pts))
                  if BLOCK_MIN_DAYS <= (pts[k][0] - pts[i][0]).days <= BLOCK_MAX_DAYS), None)
        if j is None:
            i += 1
            continue
        (d0, f0, t0), (d1, f1, t1) = pts[i], pts[j]
        i = j
        days = (d1 - d0).days
        ridden = [d0] + sorted(
            date.fromisoformat(k) for k, v in daily_tss.items()
            if v and d0.isoformat() <= k <= d1.isoformat()) + [d1]
        # Rest days between two rides: one ride on day 0 and the next on day 15
        # is a 14-day gap.
        if max((b - a).days - 1 for a, b in zip(ridden, ridden[1:])) > MAX_RIDING_GAP_DAYS:
            continue
        ctl0 = _ctl_on(ctl_by_day, d0)
        tss = sum(float(v or 0) for k, v in daily_tss.items()
                  if d0.isoformat() <= k < d1.isoformat())
        r = (tss / (days / 7.0)) / (ctl0 * 7) if ctl0 else None
        model = project_ftp(f0, weight_kg, ctl0, days / 7.0, RETRO_HIT_PER_WEEK, r, sex)
        central = model["gain_pct"]["central"]
        if central <= 0:
            continue  # a reduced-load block (8b) projects 0%: rho is undefined
        observed = (f1 / f0 - 1) * 100
        sd = CHANGE_SD_TESTED_PCT if (t0 and t1) else CHANGE_SD_EFTP_PCT
        blocks.append({"start": d0.isoformat(), "end": d1.isoformat(),
                       "observed_pct": observed, "model_central_pct": central,
                       "rho": observed / central, "sigma": sd / central})
    return blocks


def personalise(model: dict, blocks: list[dict]) -> dict:
    """Shrink the rider's mean response ratio toward 1 and narrow the band.
    With fewer than two blocks the model is returned unchanged."""
    n = len(blocks)
    out = dict(model, personalised=False, personal_blocks=n, personal_ratio=None)
    if n < PERSONAL_MIN_BLOCKS:
        return out
    tau2 = PERSONAL_TAU ** 2
    sigma2 = sum(b["sigma"] ** 2 for b in blocks) / n
    mean_rho = sum(b["rho"] for b in blocks) / n
    rho = (n * tau2 * mean_rho + sigma2 * 1.0) / (n * tau2 + sigma2)
    rho = max(PERSONAL_RHO_CLAMP[0], min(PERSONAL_RHO_CLAMP[1], rho))
    shrink = math.sqrt(sigma2 / (n * tau2 + sigma2))  # sqrt(posterior / prior var)
    g = model["gain_pct"]
    central = g["central"] * rho
    lo = g["central"] - g["conservative"]
    hi = g["optimistic"] - g["central"]
    lo = max(lo * shrink, min(lo, CHANGE_SD_TESTED_PCT))
    hi = max(hi * shrink, min(hi, CHANGE_SD_TESTED_PCT))
    out["gain_pct"] = {"conservative": central - lo, "central": central,
                       "optimistic": central + hi}
    out.update(personalised=True, personal_ratio=round(rho, 3))
    return out


# ── Whole card ─────────────────────────────────────────────────────────────

def build_projection(plan: dict, ftp: float, weight_kg: float,
                     ctl: float | None, sex: str | None, today: date,
                     history: list[dict] | None = None,
                     daily_tss: dict | None = None,
                     ctl_by_day: dict | None = None) -> dict:
    """The endpoint payload. ``ctl`` None means unknown (planning fallback)."""
    ctl = ctl or None  # CTL 0 says nothing about load: treat it as unknown
    inp = plan_inputs(plan, today)
    if inp is None:
        return {"available": False, "reason": "no_remaining_weeks"}
    r = inp["weekly_tss"] / (ctl * 7) if ctl else None
    model = project_ftp(ftp, weight_kg, ctl, inp["weeks"], inp["hit_per_week"], r, sex)
    blocks = past_blocks(history or [], daily_tss or {}, ctl_by_day or {},
                         weight_kg, sex, today)
    res = personalise(model, blocks)
    g = res["gain_pct"]
    citations = MODEL_CITATIONS + (PERSONAL_CITATIONS if res["personalised"] else [])
    return {
        "available": True,
        "ftp": ftp,
        "weight_kg": weight_kg,
        "ctl": round(ctl, 1) if ctl is not None else None,
        "ctl_unknown": ctl is None,
        "tier": res["tier"],
        "tier_label": res["tier_label"],
        "weeks": round(inp["weeks"], 1),
        "hit_per_week": round(inp["hit_per_week"], 2),
        "sweetspot_only": inp["sweetspot_only"],
        "load_ratio": round(r, 2) if r is not None else None,
        "gain_pct": {k: round(v, 2) for k, v in g.items()},
        "watts": {k: _round5(ftp * (1 + v / 100)) for k, v in g.items()},
        "within_noise": g["central"] < WITHIN_NOISE_PCT,
        "held_flat": res["held_flat"],
        "load_reduced": res["load_reduced"],
        "personalised": res["personalised"],
        "personal_blocks": res["personal_blocks"],
        "personal_ratio": res["personal_ratio"],
        "model_citations": citations,
    }
