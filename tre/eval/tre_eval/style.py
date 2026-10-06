"""Consistent colours and line styles.

Models take categorical slots 1-3 (blue, orange, aqua; validated all-pairs for CVD) in the
registry order. Arms take slots 4-8 by family (TRE violet, APA red, Chiron green, TokenScale
yellow, PreServe magenta) plus a per-arm line style / marker as the secondary encoding, so an
arm never shares a hue with a model. Unknown families fall back to grey with distinct styles.
"""

from __future__ import annotations

MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#008300"]
ALL_COLOR = "#0b0b0b"
FAMILY_COLORS = {"tre": "#4a3aa7", "apa": "#e34948", "chiron": "#008300", "tokenscale": "#c98500",
                 "preserve": "#d55181"}
OTHER_COLOR = "#52514e"
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
HATCHES = [None, "..", "//", "xx", "\\\\", "++"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
STAGE_COLORS = {"decision": "#2a78d6", "awake": "#eb6834", "routable": "#1baf7a", "donor": "#52514e",
                "slo_ok": "#eda100"}


def model_color(models: list[str], m: str) -> str:
    if m == "ALL":
        return ALL_COLOR
    try:
        return MODEL_COLORS[models.index(m) % len(MODEL_COLORS)]
    except ValueError:
        return OTHER_COLOR


def family(name: str, label: str = "") -> str:
    s = (name + " " + label).lower()
    for f in ("tokenscale", "preserve", "chiron", "apa", "tre"):
        if f in s:
            return f
    return "other"


class ArmStyles:
    """Stable style per arm for one report: family colour + n-th line style within the family."""

    def __init__(self, arms: list[tuple[str, str]]):
        self._s: dict[str, dict] = {}
        count: dict[str, int] = {}
        for i, (name, label) in enumerate(arms):
            f = family(name, label)
            k = count.get(f, 0)
            count[f] = k + 1
            self._s[name] = {"color": FAMILY_COLORS.get(f, OTHER_COLOR), "ls": LINESTYLES[k % len(LINESTYLES)],
                             "marker": MARKERS[(k + (0 if f != "other" else i)) % len(MARKERS)],
                             "hatch": HATCHES[k % len(HATCHES)]}

    def __getitem__(self, name: str) -> dict:
        return self._s[name]


def apply_rc():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.4,
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
    })
