"""Consulting-style light theme: white surface, navy/charcoal ink, red/green for sign only.

Categorical slots follow the validated reference order (blue, orange, aqua, yellow, magenta,
violet; validator PASS on the light surface). Three slots are below 3:1 contrast, so every chart
ships hover details and a table view (relief rule).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = "#1f2a44"
CHARCOAL = "#3a3f47"
MUTED = "#6b7280"
GRID = "#e6e8ec"
SURFACE = "#ffffff"
POS = "#2e7d32"  # positive values only
NEG = "#c62828"  # negative values only
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
OTHER = "#9aa0a6"
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ_RED = ["#fdf3f2", "#f6c9c4", "#ec8f86", "#d9534f", "#a8322d", "#6e1d1a"]
EVENT_COLORS = {
    "GEOPOLITICAL": SERIES[0],
    "MACROECONOMIC": SERIES[1],
    "CREDIT_EVENT": SERIES[5],
    "OPERATIONAL_ESG": SERIES[2],
}


def event_color(cls: str) -> str:
    return EVENT_COLORS.get(cls, OTHER)


def register_plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font={"family": "Inter, Helvetica Neue, Arial, sans-serif", "size": 12, "color": CHARCOAL},
        title={"font": {"size": 14, "color": NAVY}, "x": 0, "xanchor": "left"},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        xaxis={"gridcolor": GRID, "linecolor": GRID, "zeroline": False, "ticks": ""},
        yaxis={"gridcolor": GRID, "linecolor": GRID, "zeroline": False, "ticks": ""},
        legend={"orientation": "h", "y": -0.18, "font": {"size": 11}},
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        hoverlabel={"bgcolor": "white", "font": {"color": CHARCOAL}},
    )
    pio.templates["riskpulse"] = t
    pio.templates.default = "riskpulse"


CSS = f"""
<style>
  .block-container {{padding-top: 1.6rem; max-width: 1400px;}}
  h1, h2, h3 {{color: {NAVY}; font-weight: 600; letter-spacing: -0.01em;}}
  .rp-kpi {{border: 1px solid {GRID}; border-radius: 6px; padding: 10px 14px; background: #fafbfc;}}
  .rp-kpi .label {{color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;}}
  .rp-kpi .value {{color: {NAVY}; font-size: 1.45rem; font-weight: 600;}}
  .rp-kpi .delta-neg {{color: {NEG}; font-size: 0.85rem;}}
  .rp-kpi .delta-pos {{color: {POS}; font-size: 0.85rem;}}
  .rp-note {{color: {MUTED}; font-size: 0.82rem;}}
  .rp-badge {{display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.75rem;
             border:1px solid {GRID}; color:{CHARCOAL}; background:#f3f4f6;}}
  .rp-synth {{border-left: 3px solid {NEG}; padding: 6px 10px; background: #fff7f6; color: {CHARCOAL};}}
</style>
"""


def setup_page(title: str) -> None:
    st.set_page_config(page_title=f"RiskPulse | {title}", layout="wide")
    register_plotly_template()
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str | None = None, negative: bool | None = None) -> str:
    d = ""
    if delta is not None:
        cls = "delta-neg" if negative else "delta-pos" if negative is False else "rp-note"
        d = f'<div class="{cls}">{delta}</div>'
    return f'<div class="rp-kpi"><div class="label">{label}</div><div class="value">{value}</div>{d}</div>'


def usd(x: float, unit: str = "auto") -> str:
    """Format USD with units: 'USD 1.24 bn' / 'USD 312.5 m'."""
    a = abs(x)
    sign = "-" if x < 0 else ""
    if unit == "bn" or (unit == "auto" and a >= 1e9):
        return f"{sign}USD {a / 1e9:,.2f} bn"
    if unit == "m" or (unit == "auto" and a >= 1e6):
        return f"{sign}USD {a / 1e6:,.1f} m"
    return f"{sign}USD {a:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
