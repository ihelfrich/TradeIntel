"""
Visual theme system for Trade Network Intelligence.

Provides Plotly layout defaults, color scales, custom CSS,
and HTML helper functions for a dark, publication-quality aesthetic
inspired by NYT data visualization.
"""

# ── Color Palette ────────────────────────────────────────────────────────────

_BG_PRIMARY = "#0a0e1a"
_BG_CARD = "#0d1b2a"
_BG_CARD_ALT = "#1b2838"
_BG_SURFACE = "#111827"
_BORDER_SUBTLE = "#1a2235"
_BORDER_MID = "#2a3550"
_TEXT_PRIMARY = "#e8edf5"
_TEXT_SECONDARY = "#c8d6e5"
_TEXT_MUTED = "#7b8fa3"
_ACCENT_BLUE = "#64b5f6"
_ACCENT_BLUE_BRIGHT = "#90caf9"
_ACCENT_CYAN = "#4dd0e1"
_GRID_COLOR = "#1a2235"
_AXIS_LINE = "#2a3550"
_TICK_COLOR = "#4a5568"
_HOVER_BG = "#1a2235"

# ── Color Scales ─────────────────────────────────────────────────────────────

WELFARE_COLORSCALE = [
    [0.0, "#b71c1c"],
    [0.1, "#d32f2f"],
    [0.2, "#e57373"],
    [0.3, "#ef9a9a"],
    [0.4, "#ffcdd2"],
    [0.5, "#ffffff"],
    [0.6, "#c8e6c9"],
    [0.7, "#a5d6a7"],
    [0.8, "#66bb6a"],
    [0.85, "#43a047"],
    [0.95, "#2e7d32"],
    [1.0, "#1b5e20"],
]

SEQUENTIAL_COLORSCALE = [
    [0.0, "#0a0e1a"],
    [0.05, "#0d1b2a"],
    [0.15, "#0d2847"],
    [0.25, "#0e3564"],
    [0.35, "#0f4281"],
    [0.45, "#1565c0"],
    [0.55, "#1e88e5"],
    [0.65, "#42a5f5"],
    [0.75, "#64b5f6"],
    [0.85, "#90caf9"],
    [0.95, "#bbdefb"],
    [1.0, "#e3f2fd"],
]

CATEGORY_COLORS = [
    "#64b5f6",  # blue
    "#ef5350",  # red
    "#66bb6a",  # green
    "#ffa726",  # orange
    "#ab47bc",  # purple
    "#26c6da",  # cyan
    "#ffee58",  # yellow
    "#ec407a",  # pink
    "#8d6e63",  # brown
    "#78909c",  # blue-grey
    "#7e57c2",  # deep purple
    "#29b6f6",  # light blue
    "#d4e157",  # lime
    "#ff7043",  # deep orange
    "#26a69a",  # teal
    "#5c6bc0",  # indigo
]

# ── Plotly Dark Theme ────────────────────────────────────────────────────────

DARK_THEME = dict(
    # Canvas
    paper_bgcolor=_BG_PRIMARY,
    plot_bgcolor=_BG_PRIMARY,

    # Typography
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color=_TEXT_SECONDARY,
        size=13,
    ),

    # Title
    title=dict(
        font=dict(
            size=18,
            color="#ffffff",
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        ),
        x=0.0,
        xanchor="left",
        yanchor="top",
        pad=dict(l=10, t=10),
    ),

    # Legend
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(color=_TEXT_SECONDARY, size=12),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),

    # Color axis (for choropleth / heatmap color bars)
    coloraxis=dict(
        colorbar=dict(
            bgcolor="rgba(0,0,0,0)",
            tickfont=dict(color=_TEXT_MUTED, size=11),
            title_font=dict(color=_TEXT_SECONDARY, size=12),
            outlinewidth=0,
            thickness=14,
            len=0.6,
        ),
    ),

    # X axis
    xaxis=dict(
        showgrid=True,
        gridcolor=_GRID_COLOR,
        gridwidth=1,
        griddash="dot",
        linecolor=_AXIS_LINE,
        linewidth=1,
        tickcolor=_TICK_COLOR,
        tickfont=dict(color=_TEXT_MUTED, size=11),
        title_font=dict(color=_TEXT_SECONDARY, size=13),
        zeroline=False,
        showline=True,
        ticks="outside",
        ticklen=4,
    ),

    # Y axis
    yaxis=dict(
        showgrid=True,
        gridcolor=_GRID_COLOR,
        gridwidth=1,
        griddash="dot",
        linecolor=_AXIS_LINE,
        linewidth=1,
        tickcolor=_TICK_COLOR,
        tickfont=dict(color=_TEXT_MUTED, size=11),
        title_font=dict(color=_TEXT_SECONDARY, size=13),
        zeroline=False,
        showline=True,
        ticks="outside",
        ticklen=4,
    ),

    # Margins — compact but readable
    margin=dict(l=60, r=30, t=60, b=50),

    # Hover label
    hoverlabel=dict(
        bgcolor=_HOVER_BG,
        bordercolor=_BORDER_MID,
        font=dict(
            color="#e0e0e0",
            size=12,
            family="Inter, -apple-system, sans-serif",
        ),
    ),

    # Map / Geo defaults
    geo=dict(
        bgcolor=_BG_PRIMARY,
        lakecolor=_BG_PRIMARY,
        landcolor="#111827",
        subunitcolor=_BORDER_SUBTLE,
        showlakes=True,
        showland=True,
        showcountries=True,
        countrycolor="#2a3550",
        coastlinecolor="#2a3550",
        framecolor=_BORDER_SUBTLE,
        projection_type="natural earth",
    ),

    # Ternary / polar (if used)
    polar=dict(
        bgcolor=_BG_PRIMARY,
        radialaxis=dict(gridcolor=_GRID_COLOR, linecolor=_AXIS_LINE),
        angularaxis=dict(gridcolor=_GRID_COLOR, linecolor=_AXIS_LINE),
    ),

    # Modebar
    modebar=dict(
        bgcolor="rgba(0,0,0,0)",
        color=_TEXT_MUTED,
        activecolor=_ACCENT_BLUE,
    ),

    # Scene (3D plots)
    scene=dict(
        xaxis=dict(
            backgroundcolor=_BG_PRIMARY,
            gridcolor=_GRID_COLOR,
            linecolor=_AXIS_LINE,
            tickfont=dict(color=_TEXT_MUTED, size=10),
            title_font=dict(color=_TEXT_SECONDARY, size=12),
            showbackground=True,
            zerolinecolor=_GRID_COLOR,
        ),
        yaxis=dict(
            backgroundcolor=_BG_PRIMARY,
            gridcolor=_GRID_COLOR,
            linecolor=_AXIS_LINE,
            tickfont=dict(color=_TEXT_MUTED, size=10),
            title_font=dict(color=_TEXT_SECONDARY, size=12),
            showbackground=True,
            zerolinecolor=_GRID_COLOR,
        ),
        zaxis=dict(
            backgroundcolor=_BG_PRIMARY,
            gridcolor=_GRID_COLOR,
            linecolor=_AXIS_LINE,
            tickfont=dict(color=_TEXT_MUTED, size=10),
            title_font=dict(color=_TEXT_SECONDARY, size=12),
            showbackground=True,
            zerolinecolor=_GRID_COLOR,
        ),
    ),
)


def apply_theme(fig):
    """Apply the dark publication theme to any Plotly figure.

    Handles both standard 2D figures and figures with subplots
    (multiple x/y axes). Returns the figure for chaining.
    """
    fig.update_layout(**DARK_THEME)

    # Apply axis styling to all subplot axes (xaxis2, yaxis2, etc.)
    axis_keys_x = [k for k in fig.layout.to_plotly_json() if k.startswith("xaxis")]
    axis_keys_y = [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]

    x_defaults = DARK_THEME["xaxis"]
    y_defaults = DARK_THEME["yaxis"]

    for key in axis_keys_x:
        fig.update_layout(**{key: x_defaults})
    for key in axis_keys_y:
        fig.update_layout(**{key: y_defaults})

    return fig


# ── Custom CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   Trade Network Intelligence — Visual Theme
   A dark, publication-quality design system.
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Web Fonts ────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ────────────────────────────────────────────────────────── */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0d1b2a;
    --bg-surface: #111827;
    --bg-card: #0f1729;
    --bg-card-hover: #131d33;
    --border-subtle: rgba(42, 53, 80, 0.5);
    --border-glow: rgba(100, 181, 246, 0.25);
    --text-primary: #e8edf5;
    --text-secondary: #c8d6e5;
    --text-muted: #7b8fa3;
    --accent-blue: #64b5f6;
    --accent-blue-dim: rgba(100, 181, 246, 0.15);
    --accent-cyan: #4dd0e1;
    --accent-green: #66bb6a;
    --accent-red: #ef5350;
    --accent-orange: #ffa726;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-serif: 'Source Serif 4', 'Georgia', serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 18px;
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.35);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
    --shadow-glow: 0 0 20px rgba(100, 181, 246, 0.1);
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global Resets & Base ─────────────────────────────────────────────────── */
.stApp {
    font-family: var(--font-sans) !important;
    background: linear-gradient(168deg, #060a14 0%, var(--bg-primary) 30%, #0c1222 100%) !important;
    color: var(--text-secondary) !important;
}

.stApp > header {
    background: rgba(10, 14, 26, 0.92) !important;
    backdrop-filter: blur(12px) saturate(140%);
    -webkit-backdrop-filter: blur(12px) saturate(140%);
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #080c18 0%, #0e1528 40%, #131b30 70%, #0b1020 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
    font-family: var(--font-sans) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-family: var(--font-serif) !important;
    font-weight: 600 !important;
    font-size: 1.35rem !important;
    letter-spacing: -0.01em !important;
    color: #ffffff !important;
    margin-bottom: 2px !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
    font-style: italic !important;
    font-family: var(--font-serif) !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}

[data-testid="stSidebar"] .stRadio > div > label {
    padding: 7px 12px !important;
    border-radius: var(--radius-sm) !important;
    transition: all var(--transition-fast) !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(100, 181, 246, 0.07) !important;
    border-color: var(--border-glow) !important;
    color: var(--accent-blue) !important;
}

[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(100, 181, 246, 0.10) !important;
    border-color: rgba(100, 181, 246, 0.30) !important;
    box-shadow: inset 3px 0 0 var(--accent-blue) !important;
}

[data-testid="stSidebar"] .stRadio > div > label span[data-testid="stCaptionContainer"] {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    font-family: var(--font-sans) !important;
    font-style: normal !important;
    opacity: 0.7;
}

/* Sidebar select boxes & inputs */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(13, 27, 42, 0.7) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(13, 27, 42, 0.8), rgba(15, 23, 41, 0.6)) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 18px 20px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-glow) !important;
    box-shadow: var(--shadow-md), var(--shadow-glow), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transform: translateY(-1px);
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.02em !important;
    font-variant-numeric: tabular-nums !important;
}

[data-testid="stMetricDelta"] {
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

[data-testid="stMetricDelta"] svg {
    width: 12px !important;
    height: 12px !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid var(--border-subtle) !important;
    padding: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 22px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border: none !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    transition: all var(--transition-fast) !important;
    letter-spacing: 0.01em !important;
    white-space: nowrap !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    border-bottom-color: rgba(100, 181, 246, 0.35) !important;
    background: rgba(100, 181, 246, 0.04) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent-blue) !important;
    background: rgba(100, 181, 246, 0.06) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Expanders ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader,
[data-testid="stExpanderToggleIcon"],
details > summary {
    background: linear-gradient(135deg, rgba(13, 27, 42, 0.6), rgba(27, 40, 56, 0.4)) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    transition: all var(--transition-fast) !important;
    border: 1px solid var(--border-subtle) !important;
}

.streamlit-expanderHeader:hover,
details > summary:hover {
    background: linear-gradient(135deg, rgba(13, 27, 42, 0.8), rgba(27, 40, 56, 0.6)) !important;
    border-color: var(--border-glow) !important;
}

details[open] > summary {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-bottom-color: transparent !important;
}

[data-testid="stExpander"] details > div {
    background: rgba(13, 27, 42, 0.3) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 8px 24px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    font-family: var(--font-sans) !important;
    letter-spacing: 0.02em !important;
    transition: all var(--transition-base) !important;
    box-shadow: 0 2px 8px rgba(21, 101, 192, 0.3) !important;
    text-transform: none !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1e88e5, #42a5f5) !important;
    box-shadow: 0 4px 18px rgba(30, 136, 229, 0.45), 0 0 12px rgba(100, 181, 246, 0.2) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 1px 4px rgba(21, 101, 192, 0.25) !important;
}

/* Secondary / download buttons */
.stDownloadButton > button {
    background: rgba(13, 27, 42, 0.6) !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--border-glow) !important;
    box-shadow: none !important;
}

.stDownloadButton > button:hover {
    background: rgba(100, 181, 246, 0.1) !important;
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 12px rgba(100, 181, 246, 0.15) !important;
}

/* ── Dataframes ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"],
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
}

[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
.dvn-scroller {
    border-radius: var(--radius-md) !important;
}

/* Glide data editor header */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] .gdg-header {
    background: rgba(13, 27, 42, 0.8) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* ── Select boxes & inputs ────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(15, 23, 41, 0.7) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    transition: border-color var(--transition-fast) !important;
}

.stSelectbox > div > div:hover,
.stMultiSelect > div > div:hover {
    border-color: rgba(100, 181, 246, 0.35) !important;
}

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 1px rgba(100, 181, 246, 0.2) !important;
}

.stNumberInput > div > div > input,
.stTextInput > div > div > input,
textarea {
    background: rgba(15, 23, 41, 0.7) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

/* ── Slider ───────────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent-blue) !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 0 8px rgba(100, 181, 246, 0.35) !important;
}

.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
    background: var(--border-subtle) !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(100, 181, 246, 0.2) 20%,
        rgba(100, 181, 246, 0.2) 80%,
        transparent 100%
    ) !important;
    margin: 16px 0 !important;
}

/* ── Markdown / Text ──────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h1 {
    font-family: var(--font-serif) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}

[data-testid="stMarkdownContainer"] h2 {
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    font-size: 1.3rem !important;
    margin-top: 1.2em !important;
}

[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font-sans) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 1.1rem !important;
}

[data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
    line-height: 1.65 !important;
    font-size: 0.95rem !important;
}

[data-testid="stMarkdownContainer"] a {
    color: var(--accent-blue) !important;
    text-decoration: none !important;
    border-bottom: 1px solid rgba(100, 181, 246, 0.3) !important;
    transition: all var(--transition-fast) !important;
}

[data-testid="stMarkdownContainer"] a:hover {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

[data-testid="stMarkdownContainer"] code {
    font-family: var(--font-mono) !important;
    background: rgba(13, 27, 42, 0.7) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.85em !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ── Alerts / Info / Warning / Error ──────────────────────────────────────── */
.stAlert > div {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Plotly chart containers ──────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── Columns — reduce gap for tighter layout ──────────────────────────────── */
[data-testid="column"] {
    padding: 0 6px !important;
}

/* ── Spinner / loading ────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--accent-blue) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: rgba(100, 181, 246, 0.2);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(100, 181, 246, 0.35);
}

/* ── Toast / Notifications ────────────────────────────────────────────────── */
[data-testid="stToast"] {
    background: rgba(13, 27, 42, 0.95) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Custom metric card (injected via st.markdown) ────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, rgba(13, 27, 42, 0.85), rgba(15, 23, 41, 0.55));
    border: 1px solid rgba(42, 53, 80, 0.5);
    border-radius: 14px;
    padding: 20px 22px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);
    transition: all 250ms cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}

.metric-card.blue::before  { background: linear-gradient(90deg, transparent, #64b5f6, transparent); }
.metric-card.green::before { background: linear-gradient(90deg, transparent, #66bb6a, transparent); }
.metric-card.red::before   { background: linear-gradient(90deg, transparent, #ef5350, transparent); }
.metric-card.orange::before{ background: linear-gradient(90deg, transparent, #ffa726, transparent); }
.metric-card.cyan::before  { background: linear-gradient(90deg, transparent, #4dd0e1, transparent); }
.metric-card.purple::before{ background: linear-gradient(90deg, transparent, #ab47bc, transparent); }

.metric-card:hover {
    border-color: rgba(100, 181, 246, 0.25);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35), 0 0 20px rgba(100, 181, 246, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.06);
    transform: translateY(-2px);
}

.metric-card .mc-label {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7b8fa3;
    margin-bottom: 6px;
    font-family: 'Inter', sans-serif;
}

.metric-card .mc-value {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    font-family: 'Inter', sans-serif;
}

.metric-card.blue .mc-value  { color: #64b5f6; }
.metric-card.green .mc-value { color: #66bb6a; }
.metric-card.red .mc-value   { color: #ef5350; }
.metric-card.orange .mc-value{ color: #ffa726; }
.metric-card.cyan .mc-value  { color: #4dd0e1; }
.metric-card.purple .mc-value{ color: #ab47bc; }

.metric-card .mc-delta {
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 5px;
    font-family: 'Inter', sans-serif;
}

.metric-card .mc-delta.positive { color: #66bb6a; }
.metric-card .mc-delta.negative { color: #ef5350; }
.metric-card .mc-delta.neutral  { color: #7b8fa3; }

/* ── Section header (injected via st.markdown) ────────────────────────────── */
.section-header {
    margin: 28px 0 18px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(100, 181, 246, 0.15);
}

.section-header .sh-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin: 0;
}

.section-header .sh-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: #7b8fa3;
    margin-top: 4px;
    line-height: 1.45;
    font-weight: 400;
}

/* ── Animation utility ────────────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeInUp 0.4s ease-out both;
}
</style>
"""

# ── HTML Helper Functions ────────────────────────────────────────────────────

_COLOR_MAP = {
    "blue": "#64b5f6",
    "green": "#66bb6a",
    "red": "#ef5350",
    "orange": "#ffa726",
    "cyan": "#4dd0e1",
    "purple": "#ab47bc",
}


def metric_card(label: str, value: str, delta: str | None = None, color: str = "blue") -> str:
    """Return an HTML string for a custom-styled metric card.

    Usage::

        st.markdown(metric_card("Trade Volume", "$4.2T", "+12.3%", "blue"),
                    unsafe_allow_html=True)

    Parameters
    ----------
    label : str
        Short uppercase-style label above the value.
    value : str
        The main displayed value (pre-formatted).
    delta : str, optional
        Change indicator, e.g. "+12.3%" or "-0.5pp".  Prefix with ``+``
        for green, ``-`` for red, anything else for neutral.
    color : str
        One of "blue", "green", "red", "orange", "cyan", "purple".
    """
    color_class = color if color in _COLOR_MAP else "blue"

    delta_html = ""
    if delta is not None:
        delta_str = str(delta)
        if delta_str.startswith("+") or delta_str.startswith("▲"):
            delta_class = "positive"
            arrow = "&#9650; "
        elif delta_str.startswith("-") or delta_str.startswith("▼"):
            delta_class = "negative"
            arrow = "&#9660; "
        else:
            delta_class = "neutral"
            arrow = ""
        delta_html = f'<div class="mc-delta {delta_class}">{arrow}{delta_str}</div>'

    return (
        f'<div class="metric-card {color_class}">'
        f'  <div class="mc-label">{label}</div>'
        f'  <div class="mc-value">{value}</div>'
        f"  {delta_html}"
        f"</div>"
    )


def section_header(title: str, subtitle: str | None = None) -> str:
    """Return an HTML string for a styled section header.

    Usage::

        st.markdown(section_header("Network Topology", "Structural analysis of trade flows"),
                    unsafe_allow_html=True)
    """
    sub_html = ""
    if subtitle is not None:
        sub_html = f'<div class="sh-subtitle">{subtitle}</div>'

    return (
        f'<div class="section-header">'
        f'  <div class="sh-title">{title}</div>'
        f"  {sub_html}"
        f"</div>"
    )
