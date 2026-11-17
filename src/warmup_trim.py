"""Warm-up trim: shorten a workout's warm-up to at most N minutes.

The rider sets "Max warm-up (min)" per profile (athlete.json
``warmup_max_min``; blank = off, 0-20) and may override it per Send. The trim
runs on the bytes that are sent or exported -- the intervals.icu push (library,
planner, plan auto-sync), the .zwo download and the FIT export -- never on the
library file on disk.

Warm-up = the leading ``<Warmup>`` elements and any leading steps below 76 %
FTP before the first step at or above 76 %. When it is longer than N minutes
it becomes ONE ramp of N minutes from its first power to its last power (N = 0
removes it). Everything after it is left byte-for-byte as it was. FTP / CP /
ramp tests (session type, the file's ftp_test tag or its content class; the
filename rule only as fallback) are never trimmed: their warm-up is part of
the protocol.
"""
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

WARMUP_MAX_LIMIT = 20          # minutes; the input and the API accept 0..20
_HARD = 0.76                   # first step at/above this ends the warm-up

# One top-level child of <workout>: a comment, a self-closing element, or an
# element with children (e.g. <SteadyState ...><textevent .../></SteadyState>).
_TOKEN = re.compile(
    r"<!--.*?-->|<(\w+)\b[^>]*?/>|<(\w+)\b[^>]*>.*?</\2\s*>", re.S)


def parse_max(value):
    """A warm-up limit from user input: int 0..20, None for blank/absent.
    Raises ValueError for anything else."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("warm-up minutes must be a number")
    f = float(value)
    if not math.isfinite(f) or f != int(f) or not (0 <= f <= WARMUP_MAX_LIMIT):
        raise ValueError(f"warm-up minutes must be a whole number 0-{WARMUP_MAX_LIMIT}")
    return int(f)


def setting(pm):
    """The profile's Max warm-up (min), or None when off/unset/garbage."""
    try:
        return parse_max(pm._athlete.get("warmup_max_min"))
    except (ValueError, TypeError, AttributeError):
        return None


def _steps(body: str):
    """[(start, end, el)] for the top-level elements of a <workout> body."""
    out, pos = [], 0
    for m in _TOKEN.finditer(body):
        if m.group(0).startswith("<!--"):
            continue
        try:
            el = ET.fromstring(m.group(0))
        except ET.ParseError:
            return None
        out.append((m.start(), m.end(), el))
    return out


def _f(el, name):
    try:
        v = el.get(name)
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _dur(el) -> float:
    if el.tag == "IntervalsT":
        return (_f(el, "Repeat") or 1) * ((_f(el, "OnDuration") or 0)
                                          + (_f(el, "OffDuration") or 0))
    return _f(el, "Duration") or 0.0


def _peak(el):
    vals = [_f(el, a) for a in ("Power", "PowerLow", "PowerHigh",
                                "OnPower", "OffPower")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _ends(el):
    """(first power, last power) as the file spells them."""
    if el.tag == "IntervalsT":
        return el.get("OnPower"), el.get("OffPower")
    first = el.get("PowerLow") if el.get("PowerLow") is not None else el.get("Power")
    last = el.get("PowerHigh") if el.get("PowerHigh") is not None else el.get("Power")
    return first, last


def total_seconds(text: str) -> float:
    """Total duration of a ZWO's <workout>, 0 when unreadable."""
    m = re.search(r"<workout>(.*)</workout>", text, re.S)
    steps = _steps(m.group(1)) if m else None
    return sum(_dur(el) for _, _, el in steps or [])


def _warmup_count(els) -> int:
    hard_at = next((i for i, el in enumerate(els)
                    if el.tag != "Warmup" and (_peak(el) or 0) >= _HARD), None)
    n = 0
    for i, el in enumerate(els):
        if el.tag == "Warmup":
            n = i + 1
            continue
        p = _peak(el)
        # Easy leading steps count only when a hard step follows them; with
        # none (an endurance ride) they ARE the workout, not its warm-up.
        if hard_at is not None and i < hard_at and p is not None and p < _HARD:
            n = i + 1
            continue
        break
    return n


def trim_zwo_text(text: str, max_min):
    """(text, info). ``info`` is None when nothing changed (limit off, warm-up
    already short enough, unreadable file); else {orig_s, new_s, warmup_s}."""
    if max_min is None:
        return text, None
    m = re.search(r"(<workout>)(.*)(</workout>)", text, re.S)
    if not m:
        return text, None
    body = m.group(2)
    steps = _steps(body)
    if not steps:
        return text, None
    els = [el for _, _, el in steps]
    n = _warmup_count(els)
    if n == 0:
        return text, None
    warm_s = sum(_dur(el) for el in els[:n])
    limit_s = int(max_min) * 60
    if warm_s <= limit_s:
        return text, None
    if limit_s == 0:
        repl = ""
    else:
        first = _ends(els[0])[0]
        last = _ends(els[n - 1])[1]
        if first is None or last is None:
            return text, None
        # A <Warmup> climbs; a warm-up that ends lower than it starts is
        # written as a plain <Ramp> so no player reads it as low->high.
        tag = "Ramp" if float(first) > float(last) else "Warmup"
        cads = {el.get("Cadence") for el in els[:n]}
        cad = (f' Cadence="{cads.pop()}"'
               if len(cads) == 1 and None not in cads else "")
        repl = (f'<{tag} Duration="{limit_s}" PowerLow="{first}" '
                f'PowerHigh="{last}"{cad} />')
    a, b = steps[0][0], steps[n - 1][1]
    if limit_s == 0:
        # the next step moves up into the warm-up's place (and indentation)
        while b < len(body) and body[b] in " \t\r\n":
            b += 1
    new_body = body[:a] + repl + body[b:]
    orig_s = sum(_dur(el) for el in els)
    new_s = orig_s - warm_s + limit_s
    out = text[:m.start(2)] + new_body + text[m.end(2):]
    out = re.sub(r"(<name>)(.*?)(</name>)",
                 lambda nm: nm.group(1) + retitle(nm.group(2), orig_s, new_s)
                 + nm.group(3), out, count=1, flags=re.S)
    return out, {"orig_s": orig_s, "new_s": new_s, "warmup_s": warm_s}


def _minutes(seconds: float) -> int:
    """Whole minutes, halves rounded UP like the library titles (54.5 -> 55;
    Python's round() would give 54)."""
    return int(math.floor(seconds / 60 + 0.5))


def retitle_filename(filename: str, orig_s: float, new_s: float) -> str:
    """The library filename's trailing total ("..._60min.zwo", "..._60min_v2.zwo")
    with the trimmed total. Only that last token changes."""
    old, new = _minutes(orig_s), _minutes(new_s)
    if old == new or not filename:
        return filename
    return re.sub(rf"(?<![\d]){old}min(?=(_v\d+)?\.zwo$)", f"{new}min",
                  filename, flags=re.I)


def retitle(name: str, orig_s: float, new_s: float) -> str:
    """Swap the workout's total minutes in a title ("... (64min)",
    "Threshold 64min — ...") for the trimmed total. Other numbers untouched."""
    if not name:
        return name
    # Library titles round a half minute both ways (<name> "54min", display
    # name "55min" for 54.5): match whichever the title uses, keep its rule.
    for rnd in (_minutes, lambda sec: int(sec // 60)):
        old, new = rnd(orig_s), rnd(new_s)
        pat = rf"(?<!\d){old}(\s?min)\b"
        if re.search(pat, name):
            if old == new:
                return name
            return re.sub(pat, lambda m, new=new: f"{new}{m.group(1)}", name)
    # Some <name>s were authored off their own total ("(60min)" on a 62-min
    # file). A single "(Nmin)" is still the total: shift it by what was removed.
    totals = re.findall(r"\((\d+)\s?min\)", name)
    delta = _minutes(new_s) - _minutes(orig_s)
    if len(totals) == 1 and delta:
        return re.sub(r"\((\d+)(\s?min)\)",
                      lambda m: f"({max(0, int(m.group(1)) + delta)}{m.group(2)})", name)
    return name


def is_protocol_test(filename: str, session_type: str = "",
                     text: str = "") -> bool:
    """FTP / CP / ramp test, never trimmed (C7). Content first: the session
    type, the workout's own ``<tag name="ftp_test"/>``, the planner's content
    class / tags for the file. The planner's filename rule is the fallback for
    a file the classification does not know (extra folder, new file)."""
    if (session_type or "").lower() == "ftp_test":
        return True
    if text and re.search(r'<tag\s+name="ftp_test', text):
        return True
    name = Path(filename or "").name
    if not name:
        return False
    try:
        import training_planner as tp
        if tp._content_class_for_zwo(name) == "ftp_test":
            return True
        ent = (tp._load_content_classifications() or {}).get(name) or {}
        return "ftp_test" in (ent.get("tags") or [])
    except Exception:
        # ponytail: no planner -> trust only the name rather than trim a test
        return name.lower().startswith("ftp_test_")


def limit_for(pm, override=None, session_type: str = ""):
    """The warm-up minutes for a workout with NO library file (a generated
    block): the per-Send override, else the setting; None = leave it."""
    if (session_type or "").lower() == "ftp_test":
        return None
    return override if override is not None else setting(pm)


def trim_for_export(raw: bytes, filename: str, pm, override=None,
                    session_type: str = ""):
    """THE hook every send/export path calls. (bytes, info): the bytes to
    send, and info as trim_zwo_text (None = ``raw`` returned untouched).
    ``override`` (a per-Send value, already validated) beats the setting."""
    limit = override if override is not None else setting(pm)
    if limit is None:
        return raw, None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw, None
    if is_protocol_test(filename, session_type, text):
        return raw, None
    out, info = trim_zwo_text(text, limit)
    if info is None:
        return raw, None
    return out.encode("utf-8"), info
