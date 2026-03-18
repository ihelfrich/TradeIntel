"""
Visual theme system for Trade Network Intelligence.

Light, warm, authoritative — designed to convey expertise and clarity.
Warm ivory paper, deep navy accents, soft shadows, smooth animations.
Inspired by Nature, Bloomberg, The Economist, Our World in Data.
"""

# ── Color Palette — Warm Paper + Deep Ink ────────────────────────────────

_BG_PAGE       = "#f7f6f3"      # warm ivory paper
_BG_CARD       = "#ffffff"      # white cards
_BG_SURFACE    = "#efedea"      # input fields, code blocks
_BG_SIDEBAR    = "#ffffff"      # clean sidebar
_BORDER        = "#ddd9d3"      # visible borders
_BORDER_SUBTLE = "#ece9e4"      # faint borders, dividers
_TEXT_PRIMARY  = "#1a1a2e"      # headings — near-black
_TEXT_BODY     = "#3d3d4e"      # body text
_TEXT_MUTED    = "#6b6b7b"      # captions, labels
_TEXT_DIM      = "#9b9ba7"      # least-important text
_ACCENT        = "#1e3a5f"      # deep navy — primary accent
_ACCENT_LIGHT  = "#2d5a8e"      # lighter navy (hover)
_ACCENT_DIM    = "rgba(30, 58, 95, 0.06)"
_ACCENT_BORDER = "rgba(30, 58, 95, 0.18)"
_GREEN         = "#16a34a"
_RED           = "#dc2626"
_BLUE          = "#2563eb"
_PURPLE        = "#7c3aed"
_CYAN          = "#0891b2"
_ORANGE        = "#c2410c"
_GOLD          = "#a16207"
_GRID          = "#e8e5e0"
_AXIS          = "#ccc8c2"

# ── Color Scales ─────────────────────────────────────────────────────────

WELFARE_COLORSCALE = [
    [0.0, "#991b1b"],
    [0.15, "#dc2626"],
    [0.3, "#f87171"],
    [0.45, "#fca5a5"],
    [0.5, "#f7f6f3"],
    [0.55, "#86efac"],
    [0.7, "#4ade80"],
    [0.85, "#16a34a"],
    [1.0, "#15803d"],
]

SEQUENTIAL_COLORSCALE = [
    [0.0, "#f7f6f3"],
    [0.1, "#dbeafe"],
    [0.2, "#bfdbfe"],
    [0.3, "#93c5fd"],
    [0.4, "#60a5fa"],
    [0.5, "#3b82f6"],
    [0.6, "#2563eb"],
    [0.7, "#1d4ed8"],
    [0.8, "#1e40af"],
    [0.9, "#1e3a8a"],
    [1.0, "#1e3a5f"],
]

CATEGORY_COLORS = [
    "#1e3a5f",  # navy (primary)
    "#dc2626",  # red
    "#16a34a",  # green
    "#7c3aed",  # purple
    "#0891b2",  # cyan
    "#c2410c",  # orange
    "#be185d",  # pink
    "#4338ca",  # indigo
    "#0d9488",  # teal
    "#a16207",  # amber
    "#6d28d9",  # violet
    "#64748b",  # slate
    "#b91c1c",  # dark red
    "#15803d",  # dark green
    "#0369a1",  # sky
    "#9333ea",  # purple-2
]

# ── Plotly Theme ─────────────────────────────────────────────────────────

DARK_THEME = dict(
    paper_bgcolor=_BG_PAGE,
    plot_bgcolor="#ffffff",

    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color=_TEXT_BODY,
        size=13,
    ),

    title=dict(
        font=dict(size=16, color=_TEXT_PRIMARY,
                  family="Inter, -apple-system, sans-serif"),
        x=0.0, xanchor="left", yanchor="top",
        pad=dict(l=4, t=8),
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(color=_TEXT_MUTED, size=12),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),

    coloraxis=dict(
        colorbar=dict(
            bgcolor="rgba(0,0,0,0)",
            tickfont=dict(color=_TEXT_MUTED, size=11),
            title_font=dict(color=_TEXT_BODY, size=12),
            outlinewidth=0, thickness=12, len=0.55,
        ),
    ),

    xaxis=dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=_AXIS, linewidth=1,
        tickcolor=_TEXT_DIM,
        tickfont=dict(color=_TEXT_MUTED, size=11),
        title_font=dict(color=_TEXT_BODY, size=12),
        zeroline=False, showline=True, ticks="outside", ticklen=4,
    ),

    yaxis=dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=_AXIS, linewidth=1,
        tickcolor=_TEXT_DIM,
        tickfont=dict(color=_TEXT_MUTED, size=11),
        title_font=dict(color=_TEXT_BODY, size=12),
        zeroline=False, showline=True, ticks="outside", ticklen=4,
    ),

    margin=dict(l=56, r=24, t=52, b=44),

    hoverlabel=dict(
        bgcolor=_BG_CARD, bordercolor=_BORDER,
        font=dict(color=_TEXT_PRIMARY, size=12, family="Inter, sans-serif"),
    ),

    geo=dict(
        bgcolor=_BG_PAGE,
        lakecolor="#dbeafe",
        landcolor="#e8e5e0",
        subunitcolor=_BORDER,
        showlakes=True, showland=True, showcountries=True,
        countrycolor=_AXIS, coastlinecolor="#b8b4ae",
        framecolor=_BORDER_SUBTLE,
        projection_type="natural earth",
        oceancolor="#eef4fb",
    ),

    polar=dict(
        bgcolor=_BG_CARD,
        radialaxis=dict(gridcolor=_GRID, linecolor=_AXIS),
        angularaxis=dict(gridcolor=_GRID, linecolor=_AXIS),
    ),

    modebar=dict(
        bgcolor="rgba(0,0,0,0)", color=_TEXT_DIM, activecolor=_ACCENT,
    ),

    scene=dict(
        xaxis=dict(backgroundcolor=_BG_PAGE, gridcolor=_GRID,
                   linecolor=_AXIS, tickfont=dict(color=_TEXT_MUTED, size=10),
                   title_font=dict(color=_TEXT_BODY, size=11),
                   showbackground=True, zerolinecolor=_GRID),
        yaxis=dict(backgroundcolor=_BG_PAGE, gridcolor=_GRID,
                   linecolor=_AXIS, tickfont=dict(color=_TEXT_MUTED, size=10),
                   title_font=dict(color=_TEXT_BODY, size=11),
                   showbackground=True, zerolinecolor=_GRID),
        zaxis=dict(backgroundcolor=_BG_PAGE, gridcolor=_GRID,
                   linecolor=_AXIS, tickfont=dict(color=_TEXT_MUTED, size=10),
                   title_font=dict(color=_TEXT_BODY, size=11),
                   showbackground=True, zerolinecolor=_GRID),
    ),
)


def apply_theme(fig):
    """Apply theme to any Plotly figure, including subplots."""
    fig.update_layout(**DARK_THEME)
    layout_json = fig.layout.to_plotly_json()
    x_defaults = DARK_THEME["xaxis"]
    y_defaults = DARK_THEME["yaxis"]
    for key in layout_json:
        if key.startswith("xaxis"):
            fig.update_layout(**{key: x_defaults})
        elif key.startswith("yaxis"):
            fig.update_layout(**{key: y_defaults})
    return fig


# ── Custom CSS ───────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   Trade Network Intelligence — Paper & Ink Theme
   Warm ivory, deep navy accent, soft shadows, smooth animations.
   Designed to convey expertise, clarity, and scholarly authority.
   ═══════════════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-page: #f7f6f3;
    --bg-card: #ffffff;
    --bg-surface: #efedea;
    --border: #ddd9d3;
    --border-subtle: #ece9e4;
    --text-primary: #1a1a2e;
    --text-body: #3d3d4e;
    --text-muted: #6b6b7b;
    --text-dim: #9b9ba7;
    --accent: #1e3a5f;
    --accent-light: #2d5a8e;
    --accent-dim: rgba(30, 58, 95, 0.06);
    --accent-border: rgba(30, 58, 95, 0.18);
    --green: #16a34a;
    --red: #dc2626;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-serif: 'Newsreader', Georgia, 'Times New Roman', serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    --radius: 10px;
    --radius-sm: 6px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.04);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.07), 0 2px 6px rgba(0,0,0,0.04);
    --shadow-hover: 0 12px 32px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
    --transition: 200ms ease;
    --transition-slow: 400ms ease;
}

/* ── Animations ──────────────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}

@keyframes accentPulse {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}

@keyframes widthExpand {
    from { width: 0; }
    to   { width: 60px; }
}

/* ── Base ────────────────────────────────────────────────────────────── */
.stApp {
    font-family: var(--font-sans) !important;
    background: var(--bg-page) !important;
    color: var(--text-body) !important;
}

.stApp > header {
    background: rgba(247, 246, 243, 0.88) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-body) !important;
    font-family: var(--font-sans) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-family: var(--font-serif) !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
    color: var(--text-primary) !important;
    margin-bottom: 2px !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
    font-family: var(--font-sans) !important;
    font-style: normal !important;
}

/* Sidebar nav items */
[data-testid="stSidebar"] .stRadio > div { gap: 1px !important; }

[data-testid="stSidebar"] .stRadio > div > label {
    padding: 7px 12px !important;
    border-radius: var(--radius-sm) !important;
    transition: all var(--transition) !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
}

[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: var(--accent-dim) !important;
    border-color: var(--accent-border) !important;
    box-shadow: inset 3px 0 0 var(--accent) !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] span:first-child,
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) span:first-child {
    color: var(--accent) !important;
}

[data-testid="stSidebar"] .stRadio > div > label span[data-testid="stCaptionContainer"] {
    font-size: 0.7rem !important;
    color: var(--text-dim) !important;
    opacity: 0.85;
}

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--bg-page) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}

/* ── Metric Cards (st.metric) ───────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 16px 18px !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
    font-variant-numeric: tabular-nums !important;
}

[data-testid="stMetricDelta"] { font-weight: 500 !important; font-size: 0.82rem !important; }

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 20px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    border: none !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    transition: all var(--transition) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-body) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Expanders ───────────────────────────────────────────────────────── */
.streamlit-expanderHeader,
[data-testid="stExpanderToggleIcon"],
details > summary {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-sm) !important;
}

details > summary:hover {
    border-color: var(--border) !important;
    box-shadow: var(--shadow-md) !important;
}

details[open] > summary {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-bottom-color: transparent !important;
}

[data-testid="stExpander"] details > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 9px 22px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    font-family: var(--font-sans) !important;
    box-shadow: 0 1px 3px rgba(30, 58, 95, 0.2) !important;
    transition: all var(--transition) !important;
}

.stButton > button:hover {
    background: var(--accent-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(30, 58, 95, 0.22) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 3px rgba(30, 58, 95, 0.15) !important;
}

.stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}

.stDownloadButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
}

/* ── Dataframes ──────────────────────────────────────────────────────── */
[data-testid="stDataFrame"], .stDataFrame {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] .gdg-header {
    background: var(--bg-surface) !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
}

.stSelectbox > div > div:hover,
.stMultiSelect > div > div:hover { border-color: #b8b4ae !important; }

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(30, 58, 95, 0.1) !important;
}

.stNumberInput > div > div > input,
.stTextInput > div > div > input,
textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

/* ── Slider ──────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg-card) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Dividers ────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border-subtle) !important;
    margin: 20px 0 !important;
}

/* ── Typography ──────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h1 {
    font-family: var(--font-serif) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}

[data-testid="stMarkdownContainer"] h2 {
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    margin-top: 1em !important;
}

[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font-sans) !important;
    color: var(--text-body) !important;
    font-weight: 500 !important;
    font-size: 1.05rem !important;
}

[data-testid="stMarkdownContainer"] p {
    color: var(--text-body) !important;
    line-height: 1.65 !important;
    font-size: 0.93rem !important;
}

[data-testid="stMarkdownContainer"] li {
    color: var(--text-body) !important;
    line-height: 1.6 !important;
}

[data-testid="stMarkdownContainer"] strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

[data-testid="stMarkdownContainer"] a {
    color: var(--accent) !important;
    text-decoration: none !important;
    border-bottom: 1px solid var(--accent-border) !important;
    transition: all var(--transition) !important;
}

[data-testid="stMarkdownContainer"] a:hover {
    color: var(--accent-light) !important;
    border-bottom-color: var(--accent-light) !important;
}

[data-testid="stMarkdownContainer"] code {
    font-family: var(--font-mono) !important;
    background: var(--bg-surface) !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    font-size: 0.84em !important;
    color: var(--accent) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────── */
.stAlert > div {
    border-radius: var(--radius) !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    border-left-width: 4px !important;
}

/* Info alert — scholarly navy accent */
.stAlert [data-testid="stNotificationContentInfo"],
.stAlert div[role="alert"] {
    background: linear-gradient(135deg, rgba(30,58,95,0.04) 0%, rgba(30,58,95,0.02) 100%) !important;
    border: 1px solid rgba(30, 58, 95, 0.10) !important;
    border-left: 4px solid rgba(30, 58, 95, 0.35) !important;
}

/* ── Progress bar ────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-light)) !important;
    border-radius: 4px !important;
}

.stProgress > div > div {
    background: var(--bg-surface) !important;
    border-radius: 4px !important;
}

/* ── Chart containers ────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-page); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #b8b4ae; }

/* ── Toast ───────────────────────────────────────────────────────────── */
[data-testid="stToast"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* ── Column gap ──────────────────────────────────────────────────────── */
[data-testid="column"] { padding: 0 6px !important; }

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ═══════════════════════════════════════════════════════════════════════
   Custom Components — Cards, Headers, Hero, Feature Panels
   ═══════════════════════════════════════════════════════════════════════ */

/* ── Metric Card ─────────────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    transition: all var(--transition);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}

.metric-card.navy::before   { background: #1e3a5f; }
.metric-card.blue::before   { background: #2563eb; }
.metric-card.green::before  { background: #16a34a; }
.metric-card.red::before    { background: #dc2626; }
.metric-card.orange::before { background: #c2410c; }
.metric-card.cyan::before   { background: #0891b2; }
.metric-card.purple::before { background: #7c3aed; }
.metric-card.gold::before   { background: #a16207; }

.metric-card:hover {
    border-color: var(--border);
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}

.metric-card .mc-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6b6b7b;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
}

.metric-card .mc-value {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    font-family: 'Inter', sans-serif;
}

.metric-card.navy .mc-value   { color: #1e3a5f; }
.metric-card.blue .mc-value   { color: #2563eb; }
.metric-card.green .mc-value  { color: #16a34a; }
.metric-card.red .mc-value    { color: #dc2626; }
.metric-card.orange .mc-value { color: #c2410c; }
.metric-card.cyan .mc-value   { color: #0891b2; }
.metric-card.purple .mc-value { color: #7c3aed; }
.metric-card.gold .mc-value   { color: #a16207; }

.metric-card .mc-delta {
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 6px;
    font-family: 'Inter', sans-serif;
}

.metric-card .mc-delta.positive { color: #16a34a; }
.metric-card .mc-delta.negative { color: #dc2626; }
.metric-card .mc-delta.neutral  { color: #9b9ba7; }

/* ── Section Header ──────────────────────────────────────────────────── */
.section-header {
    margin: 28px 0 18px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
    animation: fadeIn 0.5s ease-out;
}

.section-header .sh-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin: 0;
}

.section-header .sh-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: #6b6b7b;
    margin-top: 4px;
    line-height: 1.45;
    font-weight: 400;
}

/* ── Hero Section ────────────────────────────────────────────────────── */
.hero-section {
    text-align: center;
    padding: 48px 24px 28px 24px;
    animation: fadeInUp 0.7s ease-out;
}

.hero-kicker {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #1e3a5f;
    margin-bottom: 14px;
}

.hero-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 3rem;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
}

.hero-subtitle {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 300;
    font-style: italic;
    color: #6b6b7b;
    margin-top: 14px;
    max-width: 640px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.55;
}

.hero-accent-line {
    height: 3px;
    width: 60px;
    margin: 20px auto 0 auto;
    border-radius: 2px;
    background: linear-gradient(90deg, #1e3a5f, #2d5a8e, #1e3a5f);
    background-size: 200% 100%;
    animation: accentPulse 4s ease-in-out infinite, widthExpand 0.8s ease-out 0.3s backwards;
}

.hero-author {
    font-family: 'Inter', sans-serif;
    font-size: 0.84rem;
    color: #9b9ba7;
    margin-top: 18px;
    font-weight: 400;
}

.hero-author strong {
    color: #1e3a5f;
    font-weight: 600;
}

/* ── Feature Card (large, for home page) ─────────────────────────────── */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 28px 26px;
    box-shadow: var(--shadow-md);
    transition: all var(--transition);
    height: 100%;
    animation: fadeInUp 0.6s ease-out backwards;
}

.feature-card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-3px);
    border-color: var(--border);
}

.feature-card .fc-kicker {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif;
}

.feature-card .fc-kicker.navy   { color: #1e3a5f; }
.feature-card .fc-kicker.green  { color: #16a34a; }
.feature-card .fc-kicker.purple { color: #7c3aed; }
.feature-card .fc-kicker.gold   { color: #a16207; }

.feature-card .fc-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1.25;
    margin-bottom: 12px;
}

.feature-card .fc-body {
    font-size: 0.88rem;
    color: #3d3d4e;
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
}

.feature-card .fc-cta {
    margin-top: 16px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

.feature-card .fc-cta.navy   { color: #1e3a5f; }
.feature-card .fc-cta.green  { color: #16a34a; }
.feature-card .fc-cta.purple { color: #7c3aed; }
.feature-card .fc-cta.gold   { color: #a16207; }

/* Staggered animation for cards in columns */
[data-testid="column"]:nth-child(1) .feature-card { animation-delay: 0.1s; }
[data-testid="column"]:nth-child(2) .feature-card { animation-delay: 0.2s; }
[data-testid="column"]:nth-child(3) .feature-card { animation-delay: 0.3s; }
[data-testid="column"]:nth-child(4) .feature-card { animation-delay: 0.4s; }

/* ── Prose Block (for substantive text) ──────────────────────────────── */
.prose-block {
    max-width: 740px;
    font-size: 0.94rem;
    line-height: 1.72;
    color: #3d3d4e;
    font-family: 'Inter', sans-serif;
    animation: fadeIn 0.6s ease-out;
}

.prose-block p { margin-bottom: 14px; }

.prose-block strong { color: #1a1a2e; }

.prose-block em { color: #6b6b7b; font-style: italic; }

/* ── Data Currency Badge ─────────────────────────────────────────────── */
.data-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 6px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: #6b6b7b;
    box-shadow: var(--shadow-sm);
    animation: fadeIn 1s ease-out 0.5s backwards;
}

.data-badge .badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #16a34a;
    box-shadow: 0 0 6px rgba(22, 163, 74, 0.4);
}

.data-badge strong {
    color: #1a1a2e;
    font-weight: 600;
}

/* ── Key Insight Callout ─────────────────────────────────────────────── */
.key-insight {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-left: 4px solid #1e3a5f;
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 18px 22px;
    margin: 12px 0;
    box-shadow: var(--shadow-sm);
    font-size: 0.9rem;
    line-height: 1.6;
    color: #3d3d4e;
    font-family: 'Inter', sans-serif;
    animation: slideInLeft 0.5s ease-out;
}

.key-insight .ki-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #1e3a5f;
    margin-bottom: 6px;
    font-family: 'Inter', sans-serif;
}

/* ── Stat Row (compact inline stats) ─────────────────────────────────── */
.stat-row {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin: 12px 0;
    padding: 14px 0;
    border-top: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
}

.stat-row .sr-item {
    font-family: 'Inter', sans-serif;
}

.stat-row .sr-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9b9ba7;
    margin-bottom: 2px;
}

.stat-row .sr-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    font-variant-numeric: tabular-nums;
}
</style>
"""

# ── HTML Helpers ─────────────────────────────────────────────────────────

_COLOR_MAP = {
    "navy": "#1e3a5f",
    "blue": "#2563eb",
    "green": "#16a34a",
    "red": "#dc2626",
    "orange": "#c2410c",
    "cyan": "#0891b2",
    "purple": "#7c3aed",
    "gold": "#a16207",
}


def metric_card(label: str, value: str, delta: str | None = None, color: str = "navy") -> str:
    """Return HTML for a styled metric card.

    Colors: navy, blue, green, red, orange, cyan, purple, gold.
    """
    color_class = color if color in _COLOR_MAP else "navy"

    delta_html = ""
    if delta is not None:
        delta_str = str(delta)
        if delta_str.startswith("+") or delta_str.startswith("▲"):
            delta_class = "positive"
        elif delta_str.startswith("-") or delta_str.startswith("▼"):
            delta_class = "negative"
        else:
            delta_class = "neutral"
        delta_html = f'<div class="mc-delta {delta_class}">{delta_str}</div>'

    return (
        f'<div class="metric-card {color_class}">'
        f'  <div class="mc-label">{label}</div>'
        f'  <div class="mc-value">{value}</div>'
        f"  {delta_html}"
        f"</div>"
    )


def section_header(title: str, subtitle: str | None = None) -> str:
    """Return HTML for a styled section header."""
    sub_html = ""
    if subtitle is not None:
        sub_html = f'<div class="sh-subtitle">{subtitle}</div>'
    return (
        f'<div class="section-header">'
        f'  <div class="sh-title">{title}</div>'
        f"  {sub_html}"
        f"</div>"
    )


def data_badge(label: str, value: str) -> str:
    """Return HTML for a data-currency badge (e.g., 'BACI · 2002–2022')."""
    return (
        f'<div class="data-badge">'
        f'  <span class="badge-dot"></span>'
        f"  {label} &ensp;<strong>{value}</strong>"
        f"</div>"
    )


def key_insight(text: str, label: str = "Key Insight") -> str:
    """Return HTML for a styled key-insight callout."""
    return (
        f'<div class="key-insight">'
        f'  <div class="ki-label">{label}</div>'
        f"  {text}"
        f"</div>"
    )


def stat_row(items: list[tuple[str, str]]) -> str:
    """Return HTML for a compact inline stat row.

    items: list of (label, value) tuples.
    """
    parts = []
    for label, value in items:
        parts.append(
            f'<div class="sr-item">'
            f'  <div class="sr-label">{label}</div>'
            f'  <div class="sr-value">{value}</div>'
            f"</div>"
        )
    return f'<div class="stat-row">{"".join(parts)}</div>'
