#!/usr/bin/env python3
"""
pii_duel.py — a two-pane "redaction race" between two privacy-filter engines,
rendered with rich and captured by recorder-for-agents into an MP4.

Both panes stream the SAME (large) document. Each engine consumes it at its REAL
measured throughput (tok/s from the README bench): the view scrolls a page at a
time as tokens are processed, PII spans snap to ████ as they're passed, and a
live counter tracks tokens + entities found. The faster engine clears the whole
document; a slower/heavier one can hit a memory wall — its bar turns red and
prints OOM at the token count where it dies.

Data (per scene dir, built by gen_corpus.py):
  <scene>/content.json -> {"document", "n_tokens", "note", "entities":[{type,start,end}]}
  <scene>/engines.json -> [{"key","label","device","tps","hero",["oom_at_tokens"]}]

  python3 pii_duel.py --scene traces/gpu --dilate 4
  python3 pii_duel.py --scene traces/cpu --dilate 1
"""
import argparse, bisect, json, time
from pathlib import Path

from rich import box
from rich.console import Group, Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

# ---- palette ---------------------------------------------------------------
INK   = "#d7dde5"
DIM   = "#6e7681"
FAINT = "#3b424c"
RULE  = "#222b34"
HERO_FILL,  HERO_EMPTY,  HERO_IDLE  = "#3ec8e0", "#1d6c7b", "#3ec8e0"   # teal
RIVAL_FILL, RIVAL_EMPTY, RIVAL_IDLE = "#94a3b2", "#39424d", "#46515e"   # slate
RED, RED_EMPTY = "#f0565b", "#5c1f22"
GREEN = "#46c266"
GOLD  = "#e3b341"

TYPES = {            # type -> (counter label, redaction-block colour)
    "person":  ("PERSON",  "#e0709a"),
    "email":   ("EMAIL",   "#7aa2f7"),
    "phone":   ("PHONE",   "#e0a458"),
    "date":    ("DATE",    "#9d7cd8"),
    "account": ("ACCOUNT", "#56b6c2"),
    "address": ("ADDRESS", "#98c379"),
}


def accent(is_hero):
    return (HERO_FILL, HERO_EMPTY, HERO_IDLE) if is_hero else (RIVAL_FILL, RIVAL_EMPTY, RIVAL_IDLE)


def wrap_ranges(s, width):
    """Greedy word-wrap over the source string -> (start, end) char ranges."""
    n = len(s); width = max(8, width)
    out, i = [], 0
    while i < n:
        nl = s.find("\n", i, i + width)        # honour hard newlines in the doc
        if nl != -1:
            out.append((i, nl)); i = nl + 1; continue
        if i + width >= n:
            out.append((i, n)); break
        sp = s.rfind(" ", i, i + width)
        if sp > i:
            out.append((i, sp)); i = sp + 1
        else:
            out.append((i, i + width)); i += width
    return out or [(0, 0)]


def prep(content, inner_w):
    """One-time layout: wrap the whole document, index entities for fast lookup."""
    doc = content["document"]
    ents = sorted(content["entities"], key=lambda e: e["start"])
    ranges = wrap_ranges(doc, inner_w)
    line_starts = [s for s, _ in ranges]
    starts = [e["start"] for e in ents]
    by_type = {}
    for e in ents:
        by_type.setdefault(e["type"], []).append(e["start"])
    types_seen = list(dict.fromkeys(e["type"] for e in ents))   # document order
    return {"doc": doc, "ents": ents, "starts": starts, "ranges": ranges,
            "line_starts": line_starts, "by_type": by_type, "types": types_seen,
            "n_tokens": content["n_tokens"]}


def _emit_norm(t, doc, a, b, playhead, cursor):
    """Append a non-entity run [a,b): processed (INK) up to the playhead, the
    block cursor, then pending (FAINT). Batched per run so it stays fast even
    with a tiny font and hundreds of visible lines."""
    if a >= b:
        return
    if playhead <= a:
        t.append(doc[a:b], style=FAINT)
    elif playhead >= b:
        t.append(doc[a:b], style=INK)
    else:
        t.append(doc[a:playhead], style=INK)
        t.append(doc[playhead], style=cursor)
        t.append(doc[playhead + 1:b], style=FAINT)


def render_page(P, page_lines, playhead, fill):
    """Render one page of wrapped lines for the given playhead char index."""
    doc, ents, starts, n = P["doc"], P["ents"], P["starts"], len(P["ents"])
    cursor = f"{fill} reverse"
    lines = []
    for s, e in page_lines:
        t = Text()
        k = bisect.bisect_left(starts, s)
        if k > 0 and ents[k - 1]["end"] > s:           # span straddling the line start
            k -= 1
        cur = s
        while k < n and ents[k]["start"] < e:
            en = ents[k]; k += 1
            a, b = max(s, en["start"]), min(e, en["end"])
            if b <= cur:
                continue
            _emit_norm(t, doc, cur, a, playhead, cursor)
            if playhead >= en["start"]:                # discovered -> redact
                t.append("█" * (b - a), style=TYPES.get(en["type"], ("", DIM))[1])
            else:
                t.append(doc[a:b], style=FAINT)        # pending PII: hidden in prose
            cur = b
        _emit_norm(t, doc, cur, e, playhead, cursor)
        lines.append(t)
    return lines


def bar(frac, width, fill, empty, label):
    f = max(0, min(width, round(width * frac)))
    t = Text()
    t.append("━" * f, style=fill)
    t.append("━" * (width - f), style=empty)
    t.append(f"  {label}", style=fill)
    return t


def counter(P, playhead, tokens_done, dead):
    """Bottom line: tokens processed + per-type entity tally (found so far)."""
    t = Text()
    t.append(f"{tokens_done:,}", style=(RED if dead else INK))
    t.append(f" / {P['n_tokens']:,} tok    ", style=DIM)
    for k, typ in enumerate(P["types"]):
        found = bisect.bisect_left(P["by_type"][typ], playhead)
        label, colour = TYPES.get(typ, (typ.upper(), DIM))
        on = found > 0
        if k:
            t.append("  ")
        t.append(f"{label} ", style=(f"bold {colour}" if on else FAINT))
        t.append(f"{found:,}", style=(INK if on else FAINT))
    return t


def pane(P, eng, real_elapsed, height, inner_w, winner):
    n_tok = P["n_tokens"]
    is_hero = eng["hero"]
    fill, empty, idle = accent(is_hero)
    body_lines = max(4, height - 8)

    oom_at = eng.get("oom_at_tokens")
    proc = eng["proc_s"]
    raw = 0.0 if proc <= 0 else min(1.0, real_elapsed / proc)
    if oom_at is not None:                       # progress capped at the memory wall
        frac = raw * (oom_at / n_tok)
        dead = real_elapsed >= proc
        done = False
    else:
        frac = raw
        dead = False
        done = real_elapsed >= proc
    playhead = round(frac * len(P["doc"]))
    tokens_done = min(round(frac * n_tok), oom_at if oom_at else n_tok)

    # scroll a page at a time: the page holding the processing frontier
    cur_line = max(0, bisect.bisect_right(P["line_starts"], playhead) - 1)
    page0 = (cur_line // body_lines) * body_lines
    page_lines = P["ranges"][page0:page0 + body_lines]
    lines = render_page(P, page_lines, playhead, fill)
    body_txt = Text("\n").join(lines)
    pad = body_lines - len(lines)
    if pad > 0:
        body_txt.append("\n" * pad)

    title = Text()
    title.append(f" {eng['label']} ", style=f"bold {fill}")
    title.append(f"{eng['device']} ", style=DIM)

    status = Text()
    if dead:
        status.append("✗ OOM ", style=f"bold {RED}")
        status.append(f"killed at {oom_at:,} tok", style=f"bold {INK}")
        status.append(f"   {eng['tps']:,} tok/s", style=DIM)
    elif done:
        status.append("✓ ", style=f"bold {GREEN}")
        status.append(f"{n_tok:,} tok", style=f"bold {INK}")
        status.append(f"   {proc * 1000:,.0f} ms   {eng['tps']:,} tok/s", style=DIM)
        if winner:
            status.append("   ★", style=f"bold {GOLD}")
    else:
        status.append("▸ ", style=fill)
        status.append(f"{real_elapsed * 1000:,.0f} ms", style=f"bold {INK}")
        status.append(f"   {eng['tps']:,} tok/s", style=DIM)

    if dead:
        bfill, bempty, blabel, border = RED, RED_EMPTY, "OOM", RED
    elif done:
        bfill, bempty, blabel, border = fill, empty, "100%", fill
        if winner:
            blabel = "100%  ★"
    else:
        bfill, bempty, blabel, border = fill, empty, f"{int(round(frac * 100)):>3d}%", idle

    body = Group(body_txt, Text(""), counter(P, playhead, tokens_done, dead),
                 bar(frac, max(10, inner_w - 8), bfill, bempty, blabel), status)
    return Panel(body, title=title, title_align="left", border_style=border,
                 box=box.ROUNDED, padding=(1, 2), height=height)


def header(engines, note, dilate, cols):
    a, b = engines
    left = Text()
    left.append(a["label"], style=f"bold {HERO_FILL}")
    left.append("  vs  ", style=DIM)
    left.append(b["label"], style=f"bold {RIVAL_FILL}")
    right = Text(justify="right")
    if note:
        right.append(note, style=INK)
    if dilate > 1.01:
        right.append(f"   ·   slowed {dilate:g}×", style=FAINT)
    g = Table.grid(expand=True)
    g.add_column(justify="left"); g.add_column(justify="right")
    g.add_row(left, right)
    return Group(g, Text("─" * cols, style=RULE), Text(""))


def view(P, engines, note, real_elapsed, dilate, cols, rows):
    avail = rows - 3
    finishers = [e for e in engines if e.get("oom_at_tokens") is None]
    fastest = min(finishers, key=lambda e: e["proc_s"]) if finishers else None
    pane_h = max(12, avail)                    # fill the screen (more text with a small font)
    top_pad = max(0, (avail - pane_h) // 2)
    inner_w = P["inner_w"]
    g = Table.grid(expand=True, padding=(0, 1))
    g.add_column(ratio=1); g.add_column(ratio=1)
    g.add_row(*[pane(P, e, real_elapsed, pane_h, inner_w, e is fastest) for e in engines])
    return Group(header(engines, note, dilate, cols), Text("\n" * top_pad), g)


def end_card(P, engines, note, link):
    finishers = [e for e in engines if e.get("oom_at_tokens") is None]
    fastest = min(finishers, key=lambda e: e["proc_s"])
    other = [e for e in engines if e is not fastest][0]
    n_tok = P["n_tokens"]
    g = Text()
    if note:
        g.append(note.upper() + "\n\n", style=f"bold {DIM}")
    wbar, top = 34, max(e["tps"] for e in engines)
    for e in engines:
        oom = e.get("oom_at_tokens")
        fill = RED if oom else accent(e["hero"])[0]
        g.append(f"{e['label']:<20}", style=f"bold {fill}")
        f = max(1, round(wbar * e["tps"] / top))
        g.append("━" * f, style=fill); g.append(" " * (wbar - f))
        if oom:
            g.append(f"   ✗ OOM at {oom:,} tok\n", style=f"bold {RED}")
        else:
            g.append(f"   {n_tok / e['tps'] * 1000:,.0f} ms   {e['tps']:,} tok/s\n", style=DIM)
    g.append("\n")
    if other.get("oom_at_tokens"):
        g.append(f"{n_tok:,} tokens, flat memory", style=f"bold {HERO_FILL}")
        g.append("  —  the other ran out of memory", style=f"bold {INK}")
    else:
        g.append(f"{fastest['tps'] / other['tps']:.1f}× faster", style=f"bold {HERO_FILL}")
        g.append(f" on the same {fastest['device'].split(' ')[0]}", style=f"bold {INK}")
    if link:
        g.append(f"\n\n{link}", style=DIM)
    return Panel(Align.center(g, vertical="middle"), border_style=HERO_EMPTY,
                 box=box.ROUNDED, padding=(2, 6))


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--scene", default=str(here / "traces/cpu"), help="dir with content.json + engines.json")
    ap.add_argument("--link", default="github.com/richiejp/privacy-filter.cpp")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dilate", type=float, default=1.0, help="time scale (1 = real time)")
    ap.add_argument("--hold", type=float, default=1.4)
    ap.add_argument("--card", type=float, default=3.5)
    a = ap.parse_args()

    scene = Path(a.scene)
    content = json.load(open(scene / "content.json"))
    engines = json.load(open(scene / "engines.json"))
    if len(engines) != 2:
        raise SystemExit(f"need exactly 2 engines, got {len(engines)}")
    for e in engines:
        oom = e.get("oom_at_tokens")
        e["proc_s"] = (oom if oom else content["n_tokens"]) / e["tps"]

    console = Console()
    cols, rows = console.size
    P = prep(content, max(20, cols // 2 - 10))
    P["inner_w"] = max(20, cols // 2 - 10)

    real_end = max(e["proc_s"] for e in engines)
    wall_end = real_end * a.dilate + a.hold
    dt = 1.0 / a.fps
    note = content.get("note", "")
    with Live(console=console, refresh_per_second=a.fps, screen=True) as live:
        t0 = time.perf_counter()
        while (w := time.perf_counter() - t0) < wall_end:
            live.update(view(P, engines, note, w / a.dilate, a.dilate, cols, rows))
            time.sleep(dt)
        live.update(view(P, engines, note, real_end, a.dilate, cols, rows))
        time.sleep(0.7)
        live.update(Panel(Align.center(end_card(P, engines, note, a.link), vertical="middle"),
                          border_style="black", box=box.SIMPLE, height=rows))
        time.sleep(a.card)


if __name__ == "__main__":
    main()
