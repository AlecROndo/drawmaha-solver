"""The site moves the way web/DESIGN.md says it does, on every stylesheet.

The standard has a mechanical half — which curve, how long, what a press
does, what reduced motion keeps — and this is where it is enforced, because
the next edit to a minified `<style>` block or a copied theme is exactly the
kind of thing a reviewer reads past. What is checked, per stylesheet:

- the motion tokens exist in every `:root` and agree across the four copies;
- no `transition: all`, no `ease-in`, and no hard-coded curve outside the
  tokens except in a figure that is allowlisted with its reason;
- nothing in the chrome runs over 300 ms, and only `transform`/`opacity`
  move, with the same allowlist for the figures that are the exception;
- every pressable has a press (`scale(0.97)` on `:active`, with `transform`
  in its transition), and a press is always between 0.95 and 0.98;
- a hover that moves something is gated behind `(hover: hover)`;
- the reduced-motion block exists, is not the nuclear `* { none }`, turns
  off every pressable's press by name, and names every rule that pulses
  (`animation`) or travels (a `transform` transition that is not the press).

Durations are read as every `<n>s` / `<n>ms` in a declaration, delays
included: a 300 ms delay is a wait the user feels just as a 300 ms slide is.

Adding motion that breaks a rule on purpose means adding its selector to
`EXPLANATORY` with a one-line reason, the way the existing ones are.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

APPS = ("rung0-viz", "rung1-viz", "rung2-viz")
PAGES = ("index.html", "rules.html", "cover.html")

TOKENS = ("--ease-out", "--ease-in-out", "--t-press")
TOKEN_VALUES = {
    "--ease-out": "cubic-bezier(0.23,1,0.32,1)",
    "--ease-in-out": "cubic-bezier(0.77,0,0.175,1)",
    "--t-press": "160ms",
}

MAX_UI_MS = 300
PRESS_RANGE = (0.95, 0.98)

# Selectors whose motion is explanatory rather than chrome: a figure drawing
# itself, a section arriving on scroll, a status pulse. Each is allowed to
# exceed 300 ms, to own a curve, or to animate a layout property — and each
# says why. Matched inside the rule's selector at a selector boundary, so
# `.card` covers `.card.lit` and `.card .n` but not `.cardinal`.
EXPLANATORY = {
    ".reveal": "a section arriving on scroll, seen once per visit; 550 ms travel",
    ".beat": "the Rules hand's current beat lights as it reaches mid-screen",
    ".fig ": "the ladder's figures draw themselves once as they scroll in",
    ".meter .track": "the Leduc meter fills once; its fill is a gradient scaleX would squash",
    ".sizes .bar": "the Rules tab's game-size bars fill once on scroll",
    ".act .bar": "the homepage's action-mix bars, a reading not a control",
    ".bars .b i": "the homepage's side-panel strategy bars, a reading not a control",
    ".console .prog .bar": "a progress bar: linear, 120 ms, tracks a live counter",
    ".live i": "the live dot's pulse, 2 s, constant",
    ".st i": "the in-progress status dot's pulse",
    ".st.prog i": "the ladder's in-progress badge pulse",
    ".st::before": "the Cover ladder's in-progress dot pulse",
    ".sweep": "rung 2's chips crossing 58 px of felt to the pot, one-shot, 400 ms",
    ".rungs-menu": "the Rungs menu animates width on purpose: it pushes Rules aside",
    ".card": "the Rules tab's worked-hand cards, highlighted once by the reading",
}

# Every pressable thing, per stylesheet. A pressable has `.x:active` with a
# scale in range and `transform` in its base transition list.
PRESSABLES = {
    "theme": (".btn", ".navbtn", ".view", ".rmain", ".racts a"),
    "rung0-viz": (".mode-tabs button",),
    "rung1-viz": (".locks button.unlock",),
    "rung2-viz": (".tl button.stn", ".next button", ".lockbtn"),
    "index.html": (
        ".btn",
        ".navbtn",
        ".rmain",
        ".racts a",
        ".console button",
        ".titlebar .tabs a",
        ".seg button",
        ".act",
        ".playbtns button",
    ),
    "rules.html": (".btn", ".navbtn", ".rmain", ".racts a"),
    "cover.html": (".btn", ".navbtn", ".rmain", ".racts a"),
}


# ---------------------------------------------------------------- the CSS


def _sheets() -> dict[str, str]:
    sheets = {"theme": (WEB / APPS[0] / "src" / "theme.css").read_text()}
    for app in APPS:
        sheets[app] = (WEB / app / "src" / "index.css").read_text()
    for page in PAGES:
        html = (WEB / "cover" / page).read_text()
        styles = re.findall(r"<style>(.*?)</style>", html, re.DOTALL)
        assert styles, f"{page} has no <style> block — the regex stopped matching"
        sheets[page] = "\n".join(styles)
    return sheets


SHEETS = _sheets()


class Rule:
    def __init__(self, media: str | None, selector: str, body: str) -> None:
        self.media = media
        self.selector = selector
        self.body = body

    def __repr__(self) -> str:
        return f"{self.media + ' ' if self.media else ''}{self.selector}"

    def declarations(self, prop: str) -> list[str]:
        """Values of `prop` (exact name, not a `prop-*` longhand)."""
        return [
            m.group(1).strip()
            for m in re.finditer(rf"(?:^|[;{{\s]){re.escape(prop)}\s*:\s*([^;}}]+)", self.body)
        ]

    @property
    def explanatory(self) -> bool:
        return any(
            re.search(rf"{re.escape(key)}(?![\w-])", self.selector) for key in EXPLANATORY
        )


def _rules(css: str, keyframes: dict[str, str] | None = None) -> list[Rule]:
    """Every `selector { body }` with the @media it sits in, comments stripped.

    `@keyframes` blocks are not rules: their `from` / `to` / `50%` steps are
    kept aside in `keyframes` by name, so a reduced-motion check can read
    what a keyframe moves, and the duration that matters stays on the
    `animation` that uses it.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    out: list[Rule] = []
    frames = keyframes if keyframes is not None else {}

    def walk(s: str, media: str | None) -> None:
        i = 0
        while True:
            j = s.find("{", i)
            if j < 0:
                return
            head = s[i:j].strip()
            depth, k = 1, j + 1
            while depth:
                depth += (s[k] == "{") - (s[k] == "}")
                k += 1
            body = s[j + 1 : k - 1]
            if head.startswith(("@media", "@supports")):
                walk(body, head)
            elif head.startswith("@keyframes"):
                frames[head.split()[-1]] = body
            else:
                out.append(Rule(media, head, body))
            i = k

    walk(css, None)
    assert out, "no rules parsed — the tokenizer stopped matching"
    return out


KEYFRAMES: dict[str, dict[str, str]] = {name: {} for name in SHEETS}
RULES = {name: _rules(css, KEYFRAMES[name]) for name, css in SHEETS.items()}


def _durations_ms(value: str) -> list[float]:
    """Every `<n>s` / `<n>ms` in a transition or animation value, in ms."""
    out = []
    for num, unit in re.findall(r"(\d*\.?\d+)(ms|s)\b", value):
        out.append(float(num) * (1 if unit == "ms" else 1000))
    return out


def _motion_values(rule: Rule) -> list[str]:
    values = []
    for prop in ("transition", "animation", "transition-duration", "animation-duration"):
        values.extend(rule.declarations(prop))
    return values


# -------------------------------------------------------------- the tokens


@pytest.mark.parametrize("name", ["theme", *PAGES])
def test_the_motion_tokens_are_declared_and_agree(name: str) -> None:
    roots = [r for r in RULES[name] if r.selector == ":root"]
    assert roots, f"{name}: no :root block"
    body = " ".join(r.body for r in roots)
    for token in TOKENS:
        m = re.search(rf"{re.escape(token)}\s*:\s*([^;]+);", body)
        assert m, f"{name}: {token} is not declared in :root (see web/DESIGN.md)"
        got = re.sub(r"\s+", "", m.group(1))
        got = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", got)  # .23 → 0.23
        assert got == TOKEN_VALUES[token], (
            f"{name}: {token} is {got}, the standard says {TOKEN_VALUES[token]}"
        )


# ------------------------------------------------------------ the curves


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_no_transition_all_and_no_ease_in(name: str) -> None:
    for rule in RULES[name]:
        for value in _motion_values(rule):
            assert not re.match(r"all\b", value), (
                f"{name}: {rule!r} transitions `all`; name the property"
            )
            assert not re.search(r"\bease-in\b(?!-out)", value), (
                f"{name}: {rule!r} uses ease-in; entrances use var(--ease-out)"
            )


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_chrome_reads_the_easing_tokens_rather_than_writing_a_curve(name: str) -> None:
    for rule in RULES[name]:
        if rule.selector == ":root" or rule.explanatory:
            continue
        for value in _motion_values(rule):
            assert "cubic-bezier(" not in value, (
                f"{name}: {rule!r} hard-codes a curve; use var(--ease-out) / "
                "var(--ease-in-out), or allowlist the figure in EXPLANATORY"
            )


# ---------------------------------------------------------- the durations


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_chrome_motion_stays_under_300ms(name: str) -> None:
    for rule in RULES[name]:
        if rule.explanatory:
            continue
        for value in _motion_values(rule):
            for ms in _durations_ms(value):
                assert ms <= MAX_UI_MS, (
                    f"{name}: {rule!r} runs {ms:g} ms (`{value}`); the chrome stays "
                    f"under {MAX_UI_MS} ms, or the figure goes in EXPLANATORY with a reason"
                )


LAYOUT_PROPS = ("width", "height", "padding", "margin", "top", "left", "right", "bottom")


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_only_transform_and_opacity_move_in_the_chrome(name: str) -> None:
    for rule in RULES[name]:
        if rule.explanatory:
            continue
        parts = [p for v in rule.declarations("transition") for p in v.split(",")]
        parts += [p for v in rule.declarations("transition-property") for p in v.split(",")]
        for part in parts:
            prop = part.strip().split()[0] if part.strip() else ""
            assert prop not in LAYOUT_PROPS, (
                f"{name}: {rule!r} animates `{prop}`, which lays out and paints; "
                "move it with transform, or allowlist it in EXPLANATORY"
            )


# --------------------------------------------------------------- the press


def _press_scale(body: str) -> float | None:
    m = re.search(r"transform\s*:\s*scale\(\s*(\d*\.?\d+)\s*\)", body)
    return float(m.group(1)) if m else None


@pytest.mark.parametrize("name", sorted(PRESSABLES))
def test_every_pressable_answers_a_press(name: str) -> None:
    rules = RULES[name]
    for sel in PRESSABLES[name]:
        actives = [
            r
            for r in rules
            if r.media is None
            and any(
                re.fullmatch(rf"{re.escape(sel)}:active(:not\([^)]*\))?", s.strip())
                for s in r.selector.split(",")
            )
        ]
        assert actives, f"{name}: {sel} has no :active rule; a press has to be heard"
        scales = [_press_scale(r.body) for r in actives]
        assert any(s is not None for s in scales), (
            f"{name}: {sel}:active does not scale; the standard is scale(0.97)"
        )

        bases = [r for r in rules if r.media is None and sel in [s.strip() for s in r.selector.split(",")]]
        assert bases, f"{name}: no base rule for {sel}"
        transitions = " ".join(v for r in bases for v in r.declarations("transition"))
        assert re.search(r"\btransform\s+var\(--t-press\)\s+var\(--ease-out\)", transitions), (
            f"{name}: {sel}'s transition list does not carry "
            "`transform var(--t-press) var(--ease-out)`, so the press would snap"
        )


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_a_press_is_subtle(name: str) -> None:
    lo, hi = PRESS_RANGE
    for rule in RULES[name]:
        if ":active" not in rule.selector:
            continue
        scale = _press_scale(rule.body)
        if scale is None:
            continue
        assert lo <= scale <= hi, (
            f"{name}: {rule!r} presses to scale({scale:g}); the range is {lo}–{hi}"
        )


# --------------------------------------------------------------- the hover


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_a_hover_that_moves_something_is_gated_on_a_real_pointer(name: str) -> None:
    for rule in RULES[name]:
        if ":hover" not in rule.selector:
            continue
        moves = [v for v in rule.declarations("transform") if v.strip() != "none"]
        if not moves:
            continue
        assert rule.media and "hover: hover" in re.sub(r"\s*:\s*", ": ", rule.media), (
            f"{name}: {rule!r} moves on hover outside `@media (hover: hover) and "
            "(pointer: fine)`; a tap would leave it stuck"
        )


# ------------------------------------------------------- reduced motion


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_reduced_motion_is_fewer_and_gentler_not_none(name: str) -> None:
    reduced = [
        r for r in RULES[name] if r.media and "prefers-reduced-motion" in r.media
    ]
    assert reduced, f"{name}: no @media (prefers-reduced-motion: reduce) block"
    for rule in reduced:
        if rule.selector.strip() == "*":
            assert "!important" not in rule.body, (
                f"{name}: reduced motion is `* {{ none !important }}`; remove the "
                "travel and the pulse by name and keep colour and opacity"
            )
    # the press is one of the things that goes quiet, for every pressable
    quiet = [
        s.strip()
        for r in reduced
        if re.search(r"transform\s*:\s*none", r.body)
        for s in r.selector.split(",")
    ]
    for sel in PRESSABLES[name]:
        assert any(
            re.fullmatch(rf"{re.escape(sel)}:active(:not\([^)]*\))?", q) for q in quiet
        ), f"{name}: {sel}'s press is not turned off under reduced motion"


def _parts(selector: str) -> set[str]:
    return {re.sub(r"\s+", " ", s.strip()) for s in selector.split(",")}


TRAVEL_PROPS = ("transform", "translate", "scale", "rotate")


def _travels(rule: Rule) -> list[str]:
    """The transition parts of a rule that move it, the press excepted."""
    parts = [p for v in rule.declarations("transition") for p in v.split(",")]
    parts += [p for v in rule.declarations("transition-property") for p in v.split(",")]
    out = []
    for part in parts:
        words = part.strip().split()
        if words and words[0] in TRAVEL_PROPS and "--t-press" not in part:
            out.append(part.strip())
    return out


def _quiets_a_pulse(rule: Rule, frames: dict[str, str]) -> bool:
    """`animation: none`, or a one-shot keyframe defined in this sheet that only fades.

    A looping animation is a pulse however gentle its keyframe, so `infinite`
    (or an iteration count above one) never counts as quiet.
    """
    loops = any(
        "infinite" in v or re.search(r"(?<![\d.])([2-9]|\d{2,}|1\.\d*[1-9])(?![\d.])", v)
        for v in rule.declarations("animation-iteration-count")
    )
    for value in rule.declarations("animation") + rule.declarations("animation-name"):
        name = value.strip().split()[0]
        if name == "none":
            return True
        if loops or "infinite" in value:
            continue
        body = frames.get(name)
        if body is not None and not re.search(r"\b(transform|translate|scale|rotate)\s*:", body):
            return True
    return False


def _quiets_a_travel(rule: Rule) -> bool:
    """`transform: none`, or a transition that no longer lists a travel."""
    if any(v.strip() == "none" for v in rule.declarations("transform")):
        return True
    transitions = rule.declarations("transition")
    return bool(transitions) and not _travels(rule)


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_every_pulse_and_travel_is_quieted_under_reduced_motion(name: str) -> None:
    """Reduced motion is by name, so every name has to be there.

    A rule that animates, or transitions a transform for anything other than
    the press, needs a reduced-motion rule for the same selector that
    actually quiets it: `animation: none`, or a keyframe this sheet defines
    with no transform in it (for a token the code waits on via
    `animationend`); `transform: none`, or a transition with the travel
    taken out.
    """
    rules = RULES[name]
    frames = KEYFRAMES[name]
    reduced = [r for r in rules if r.media and "prefers-reduced-motion" in r.media]
    quiet_anim = {
        sel for r in reduced if _quiets_a_pulse(r, frames) for sel in _parts(r.selector)
    }
    quiet_move = {
        sel for r in reduced if _quiets_a_travel(r) for sel in _parts(r.selector)
    }
    for rule in rules:
        if rule.media and "prefers-reduced-motion" in rule.media:
            continue
        pulses = [
            v
            for v in rule.declarations("animation") + rule.declarations("animation-name")
            if v.strip() != "none"
        ]
        travels = _travels(rule)
        for sel in _parts(rule.selector):
            if pulses:
                assert sel in quiet_anim, (
                    f"{name}: {sel} animates (`{pulses[0]}`) and no reduced-motion rule "
                    "quiets it; set `animation: none` by name, or swap in a keyframe "
                    "that only fades"
                )
            if travels:
                assert sel in quiet_move, (
                    f"{name}: {sel} travels (`{travels[0]}`) and no reduced-motion rule "
                    "quiets it; set its transform to none, or its transition to none "
                    "or to the non-moving properties, by name"
                )
