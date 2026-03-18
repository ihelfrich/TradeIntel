"""
Trade Network Intelligence — Interactive Global Trade Visualization
by Ian Helfrich, PhD

An interactive tool for exploring global trade networks,
simulating tariff impacts, and analyzing trade dependencies.
Built on BACI bilateral trade data (2002–2022).
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

from utils.data_loader import (
    load_bilateral_trade, load_country_codes, load_sector_flows,
    get_year_network_data, get_top_n_flows, get_country_summary,
    get_coords, SECTOR_LABELS, COUNTRY_COORDS,
)
from utils.network import (
    build_trade_graph, compute_centrality_measures,
    detect_communities, compute_dependency_metrics,
    compute_network_stats_over_time, graph_from_link_data,
    compute_3d_force_layout, compute_comprehensive_network_invariants,
)
from utils.tariff_sim import (
    simulate_tariff_shock, simulate_multi_country_tariff,
    SCENARIOS, DEFAULT_ELASTICITY,
)
from utils.gravity_model import (
    estimate_ols_gravity, estimate_ppml_gravity, estimate_sector_elasticities,
)
from utils.product_analysis import (
    get_strategic_product_flows, get_product_evolution, STRATEGIC_PRODUCTS,
)
from utils.welfare import (
    compute_welfare_impact, compute_multi_country_welfare,
)
from utils.topology import (
    trade_to_distance_matrix, compute_persistent_homology,
    compute_topological_evolution, compute_clique_complex_stats,
    topological_tariff_sensitivity,
    compute_attributed_persistent_homology,
    compute_topological_null_models,
)
from utils.mapper_analysis import compute_trade_mapper
from utils.ge_counterfactual import (
    load_trade_data, solve_counterfactual, solve_nash_equilibrium,
    solve_optimal_tariff, get_available_datasets, ELASTICITY_REGISTRY,
)
from utils.theme import (
    DARK_THEME, apply_theme, CUSTOM_CSS, WELFARE_COLORSCALE,
    SEQUENTIAL_COLORSCALE, CATEGORY_COLORS, metric_card, section_header,
    data_badge, key_insight, stat_row,
)
from utils.research_pipeline import (
    run_optimal_tariff_survey, run_elasticity_sensitivity,
    run_tariff_rate_sweep, run_retaliation_comparison,
)
from utils.topo_counterfactual import (
    compare_topology_factual_vs_counterfactual,
    topological_laffer_curve,
)
import hashlib

# ── Auto-theme: wrap st.plotly_chart so every figure gets the dark theme ──
_original_plotly_chart = st.plotly_chart

def _themed_plotly_chart(figure_or_data, *args, **kwargs):
    """Apply dark theme to every Plotly figure before rendering."""
    if hasattr(figure_or_data, "update_layout"):
        apply_theme(figure_or_data)
    return _original_plotly_chart(figure_or_data, *args, **kwargs)

st.plotly_chart = _themed_plotly_chart

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Network Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 8px 0 4px 0;">
    <div style="font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem;
                font-weight: 600; color: #1a1a2e; letter-spacing: -0.01em; line-height: 1.2;">
        Trade Network Intelligence
    </div>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: #6b6b7b;
                margin-top: 4px; letter-spacing: 0.02em;">
        Dr. Ian Helfrich — Georgia Institute of Technology
    </div>
</div>
""", unsafe_allow_html=True)

# Section labels are non-selectable dividers rendered inline
_PAGES = [
    # Section: Overview
    "Home",
    # Section: Explore
    "── EXPLORE ──",
    "3D Trade Globe",
    "Network Evolution",
    "Country Dependencies",
    "US Trade Exposure",
    "Supply Chain Deep Dive",
    # Section: Policy & Counterfactuals
    "── POLICY ──",
    "Tariff Impact Simulator",
    "Welfare Analysis",
    "GE Counterfactual Lab",
    "Research Lab",
    # Section: Methods
    "── METHODS ──",
    "Gravity Model Lab",
    "Network Invariants",
    "3D Network Topology",
    # Section: Topology (TDA)
    "── TOPOLOGY ──",
    "Topology-Counterfactual",
    "Persistent Homology",
    "Feature Explorer",
    "Statistical Significance",
    "Mapper Lens",
    "Topological Evolution",
    "Topological Sensitivity",
]

_CAPTIONS = {
    "Home": "",
    "3D Trade Globe": "Interactive bilateral flow map",
    "Network Evolution": "Structural trends, 2002–2022",
    "Country Dependencies": "Concentration risk & exposure",
    "US Trade Exposure": "Import vulnerability profile",
    "Supply Chain Deep Dive": "HS-code product mapping",
    "Tariff Impact Simulator": "Bilateral & multilateral shocks",
    "Welfare Analysis": "Gains from trade & policy costs",
    "GE Counterfactual Lab": "Multi-sector general equilibrium",
    "Research Lab": "Optimal tariffs & Laffer curves",
    "Gravity Model Lab": "OLS & PPML structural gravity",
    "Network Invariants": "Centrality, clustering, spectra",
    "3D Network Topology": "Force-directed 3D layout",
    "Topology-Counterfactual": "How tariffs reshape structure",
    "Persistent Homology": "Rips filtration & Betti numbers",
    "Feature Explorer": "Map features to countries & flows",
    "Statistical Significance": "Null model benchmarks",
    "Mapper Lens": "Compressed structural summary",
    "Topological Evolution": "Structural change over time",
    "Topological Sensitivity": "Structural tipping points",
}

page = st.sidebar.radio(
    "Navigate",
    _PAGES,
    label_visibility="collapsed",
    captions=[_CAPTIONS.get(p, "") for p in _PAGES],
)

# Load shared data
trade_full = load_bilateral_trade()
cc = load_country_codes()
all_years = sorted(trade_full["year"].unique())
iso_to_name = cc.set_index("country_iso3")["country_name"].to_dict()
name_to_iso = {v: k for k, v in iso_to_name.items()}


def fmt_billions(val):
    if val >= 1000:
        return f"${val/1000:,.1f}T"
    return f"${val:,.1f}B"


def fmt_millions(val):
    if val >= 1000:
        return f"${val/1000:,.1f}B"
    return f"${val:,.0f}M"


# ═════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═════════════════════════════════════════════════════════════════════════════
if page == "Home" or page.startswith("──"):
    # Compute live headline stats from the loaded data
    _latest_year = all_years[-1]
    _latest = trade_full[trade_full["year"] == _latest_year]
    _earliest = trade_full[trade_full["year"] == all_years[0]]
    _total_trade_T = _latest["trade_value_usd_millions"].sum() / 1e6
    _n_countries = len(set(_latest["iso_o"].unique()) | set(_latest["iso_d"].unique()))
    _n_flows = len(_latest)
    _growth_x = _latest["trade_value_usd_millions"].sum() / max(_earliest["trade_value_usd_millions"].sum(), 1)
    _top_flow = _latest.nlargest(1, "trade_value_usd_millions").iloc[0]
    _top_flow_B = _top_flow["trade_value_usd_millions"] / 1e3

    # ── Hero ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-kicker">Research Platform</div>
        <div class="hero-title">Trade Network Intelligence</div>
        <div class="hero-subtitle">
            Quantitative analysis of global trade architecture — structure, vulnerability,
            and the propagation of policy shocks across 226 economies.
        </div>
        <div class="hero-accent-line"></div>
        <div class="hero-author">
            <strong>Dr. Ian Helfrich</strong> &ensp;·&ensp; Georgia Institute of Technology
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data currency badge ──
    st.markdown(
        '<div style="text-align: center; margin-bottom: 8px;">'
        + data_badge("BACI bilateral trade data", f"{all_years[0]}–{all_years[-1]} · {_n_countries} economies")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Headline metrics ──
    h1, h2, h3, h4 = st.columns(4)
    h1.markdown(metric_card("Global Trade ({})".format(_latest_year),
                f"${_total_trade_T:.1f}T",
                f"{_growth_x:.1f}× since {all_years[0]}", "navy"), unsafe_allow_html=True)
    h2.markdown(metric_card("Economies Covered", str(_n_countries),
                f"{_n_flows:,} bilateral flows", "blue"), unsafe_allow_html=True)
    h3.markdown(metric_card("Largest Bilateral Flow",
                f"${_top_flow_B:.0f}B",
                f"{_top_flow['iso_o']} → {_top_flow['iso_d']}", "green"), unsafe_allow_html=True)
    h4.markdown(metric_card("Time Span",
                f"{all_years[0]}–{all_years[-1]}",
                f"{len(all_years)} years of BACI data", "gold"), unsafe_allow_html=True)

    st.markdown("")

    # ── The Challenge ──
    st.markdown(section_header("The Challenge"), unsafe_allow_html=True)
    st.markdown("""
    <div class="prose-block">
    <p>
    Global trade is not a collection of bilateral relationships — it is a network. The more than
    $23 trillion in goods that cross borders each year flows through a complex web of dependencies
    where a tariff imposed in Washington reverberates through supply chains in Shenzhen, assembly
    lines in Guadalajara, and commodity markets in São Paulo. Understanding this network — its
    architecture, its fragilities, and how it responds to policy intervention — requires tools that
    go beyond bilateral trade balances and static country rankings.
    </p>
    <p>
    The global trade system is undergoing its most significant structural transformation since
    the postwar era. The US-China trade war, COVID-19 supply chain disruptions, the CHIPS and
    Science Act, and the resurgence of industrial policy across advanced economies have made trade
    network analysis essential for policymakers, analysts, and researchers. Understanding not just
    <em>how much</em> countries trade, but <em>how the network is organized</em> — its cycles,
    bottlenecks, and structural vulnerabilities — is critical for anticipating how policy changes
    propagate through the system.
    </p>
    <p>
    This platform provides those tools. I integrate twenty years of bilateral trade data with
    structural gravity models, multi-sector general equilibrium analysis, and topological data
    analysis to offer a unified analytical framework for understanding how global trade works,
    where it is vulnerable, and what happens when policy disrupts it.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Capabilities (4 feature cards) ──
    st.markdown(section_header("Analytical Capabilities"), unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("""
        <div class="feature-card">
            <div class="fc-kicker navy">NETWORK ARCHITECTURE</div>
            <div class="fc-title">Map the Global Trade Network</div>
            <div class="fc-body">
                Visualize the complete bilateral trade network across 226 economies from
                2002 to 2022. Compute centrality, identify concentration risk, and track how
                the network's structure has shifted — including the emergence of China as the
                system's most connected node and the evolving position of the United States.
                Identify which import dependencies create strategic vulnerability and which
                sectors face dangerous supplier concentration.
            </div>
            <div class="fc-cta navy">→ Global Trade Map &ensp;·&ensp; Structural Trends &ensp;·&ensp; Dependency Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
        <div class="feature-card">
            <div class="fc-kicker green">POLICY COUNTERFACTUALS</div>
            <div class="fc-title">Simulate Tariff Scenarios in General Equilibrium</div>
            <div class="fc-body">
                Run "what-if" experiments through a multi-sector general equilibrium model
                based on Lashkaripour (2021, <em>Journal of International Economics</em>).
                Impose tariffs unilaterally or multilaterally, compute Nash equilibrium tariff
                vectors, find optimal unilateral tariffs, and trace the welfare implications for
                every country in the system. Results are compared across eight different trade
                elasticity specifications to quantify how sensitive policy conclusions are to
                modeling assumptions.
            </div>
            <div class="fc-cta green">→ Tariff Scenario Builder &ensp;·&ensp; General Equilibrium Lab &ensp;·&ensp; Policy Research</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    fc3, fc4 = st.columns(2)
    with fc3:
        st.markdown("""
        <div class="feature-card">
            <div class="fc-kicker purple">TOPOLOGICAL STRUCTURE</div>
            <div class="fc-title">Reveal Higher-Order Trade Patterns</div>
            <div class="fc-body">
                Apply persistent homology — a technique from topological data analysis — to
                the trade distance matrix. This reveals structural patterns invisible to
                pairwise analysis: H₁ cycles (triangular trade relationships that resist
                bilateral disruption), H₀ components (trade bloc formation dynamics), and
                H₂ voids (missing multilateral integration). Each topological feature is
                attributed to specific countries and dollar-denominated trade flows.
            </div>
            <div class="fc-cta purple">→ Persistent Homology &ensp;·&ensp; Feature Attribution &ensp;·&ensp; Statistical Testing</div>
        </div>
        """, unsafe_allow_html=True)
    with fc4:
        st.markdown("""
        <div class="feature-card">
            <div class="fc-kicker gold">STRUCTURAL POLICY ANALYSIS</div>
            <div class="fc-title">How Tariffs Reshape the Topology of Trade</div>
            <div class="fc-body">
                This platform's central analytical contribution: persistent homology applied
                to counterfactual trade networks — networks that do not exist in the observed
                data but that general equilibrium models predict would emerge under alternative
                tariff regimes. This makes it possible to ask: How does a 25% tariff on Chinese
                goods change the cycle structure of global trade? At what tariff rate does the
                network undergo a structural phase transition?
            </div>
            <div class="fc-cta gold">→ Topological Policy Analysis &ensp;·&ensp; Structural Tipping Points</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Methodology ──
    st.markdown(section_header("Data & Methodology"), unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        **Data Sources**
        - **BACI** (CEPII) — Bilateral trade flows, 226 economies, 2002–2022
        - **OECD ICIO** — 81 countries × 28 sectors, input-output tables with tariff data, 2011–2022
        - **WIOD** — 44 countries × 16 sectors, 2000–2014
        - **CEPII Gravity** — Bilateral distance, contiguity, language, colonial history, RTAs
        - **Teti Global Tariff Database** — Applied MFN and preferential tariff rates
        """)
    with d2:
        st.markdown("""
        **Methods**
        - **Structural Gravity:** OLS and PPML estimation (Santos Silva & Tenreyro, 2006)
        - **Welfare:** ACR sufficient statistics (Arkolakis, Costinot, & Rodríguez-Clare, 2012)
        - **General Equilibrium:** Lashkaripour (2021, JIE) sufficient-statistics hat algebra with CES preferences and input-output linkages
        - **Trade Elasticities:** Eight specifications — IS, CP, U4, BSY, GYY, Shapiro, FGO, LL
        - **Topology:** Persistent homology via Vietoris-Rips filtration (Carlsson, 2009); Mapper algorithm (Singh, Mémoli, & Carlsson, 2007)
        """)

    st.markdown("")

    # ── Guided paths ──
    st.markdown(section_header("Where to Start",
                "Select a path based on your analytical interest"), unsafe_allow_html=True)

    with st.expander("Policy Analysis — Tariff impacts and trade vulnerability", expanded=False):
        st.markdown("""
        **Recommended path:**

        1. **US Trade Exposure** — Identify where US imports are most concentrated and which sectors
           depend on a limited number of source countries.
        2. **Tariff Impact Simulator** — Model a specific scenario (e.g., 10% across-the-board tariffs on
           Chinese imports) and observe how trade flows reallocate across partners.
        3. **GE Counterfactual Lab** — Run the same scenario through a rigorous multi-sector general
           equilibrium model with eight different elasticity specifications to quantify the range of
           possible welfare outcomes.
        4. **Policy Research** — Compare welfare impacts with and without retaliation; examine Laffer
           curve dynamics to identify the tariff rate that maximizes national welfare.

        **Key finding:** The welfare impact of tariffs depends critically on the assumed trade elasticity.
        The eight specifications in this platform span a wide range — from near-zero costs to substantial
        welfare losses for the same tariff. Communicating this uncertainty honestly is essential for
        responsible policy analysis.
        """)

    with st.expander("Economic Modeling — Gravity, GE counterfactuals, elasticity estimation", expanded=False):
        st.markdown("""
        **Recommended path:**

        1. **Gravity Estimation** — OLS and PPML structural gravity estimates with sector-level
           elasticities. Establish benchmarks before running counterfactual analysis.
        2. **Welfare Impact** — ACR (2012) sufficient-statistics welfare computation at the country level.
        3. **GE Counterfactual Lab** — Full Lashkaripour (2021, JIE) implementation: CES preferences,
           input-output linkages, 81 countries × 28 sectors (ICIO) or 44 × 16 (WIOD).
           Compute Nash equilibrium tariffs and optimal unilateral tariffs for any country.
        4. **Policy Research** — Batch analyses: optimal tariff surveys across every country and
           elasticity specification, Laffer curves, and cross-elasticity sensitivity tests.

        The GE solver uses `scipy.optimize.root` with multi-start (hybr + Levenberg-Marquardt fallback).
        Convergence is tracked per-scenario. The eight elasticity specifications span Caliendo & Parro (2015),
        Simonovska & Waugh (2014), Shapiro (2016), and others.
        """)

    with st.expander("Topological Analysis — Persistent homology and structural inference", expanded=False):
        st.markdown("""
        **Recommended path:**

        1. **Persistent Homology** — Compute Vietoris-Rips persistent homology on the trade distance
           matrix. Observe H₀ (connected component formation), H₁ (trade cycles), and H₂ (higher-order voids).
        2. **Feature Attribution** — Map each topological feature to specific countries and dollar-denominated
           trade flows. Identify which economies form the most persistent H₁ cycle.
        3. **Statistical Testing** — Benchmark the observed Betti numbers against Erdos-Renyi,
           configuration model, and gravity-predicted null networks to test significance.
        4. **Topological Policy Analysis** — Compute persistent homology on counterfactual trade
           networks predicted by the GE model under alternative tariff regimes. The "topological
           Laffer curve" shows how the number of trade cycles varies as a function of tariff rate —
           revealing structural phase transitions where small policy changes trigger large network
           reorganizations.
        5. **Topological Summary** — Mapper algorithm visualization for a compressed structural overview.

        This analysis applies persistent homology to counterfactual trade networks — networks that exist
        only as predictions of the general equilibrium model. Standard applications of TDA to trade
        (Topaz et al., 2015; Feng et al., 2019) use observed data, so their topology largely reflects
        GDP and geographic proximity. By running TDA on GE-predicted counterfactual flows, I isolate
        how *policy* reshapes the network's topological structure.
        """)

    st.divider()
    st.markdown("""
    <div style="text-align: center; font-size: 0.82rem; color: #6b6b7b; padding: 12px 0; line-height: 1.6;">
        Select a page from the sidebar to begin. All computations run in real time on the underlying data.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# EXPLORE: GLOBAL TRADE MAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "3D Trade Globe":
    st.markdown(section_header("Global Trade Flow Map",
                "Bilateral trade flows between countries, sized by value — BACI (CEPII)"), unsafe_allow_html=True)
    st.info("**Why this matters:** Global trade exceeded $23 trillion in 2022. This map shows where that money flows — the thickness and color of each arc reveals which bilateral relationships dominate world commerce. Look for the US-China-EU triangle that accounts for over 40% of global flows.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        year = st.select_slider("Year", options=all_years, value=max(all_years))
    with col2:
        n_flows = st.slider("Top N flows to display", 50, 500, 150, 25)
    with col3:
        min_value = st.number_input(
            "Min trade value ($M)", value=0, step=100,
            help="Filter out flows below this threshold"
        )

    top_flows = get_top_n_flows(year, n_flows)
    if min_value > 0:
        top_flows = top_flows[top_flows["trade_value_usd_millions"] >= min_value]

    # Build arc map
    fig = go.Figure()

    # Add flow lines
    lons = []
    lats = []
    values = []
    hover_texts = []

    for _, row in top_flows.iterrows():
        o_lat, o_lon = get_coords(row["iso_o"])
        d_lat, d_lon = get_coords(row["iso_d"])
        if (o_lat, o_lon) == (0, 0) or (d_lat, d_lon) == (0, 0):
            continue
        val = row["trade_value_usd_millions"]
        lons.extend([o_lon, d_lon, None])
        lats.extend([o_lat, d_lat, None])
        exp_name = row.get("exporter_name", row["iso_o"])
        imp_name = row.get("importer_name", row["iso_d"])
        hover_texts.extend([
            f"{exp_name} → {imp_name}: {fmt_millions(val)}",
            f"{exp_name} → {imp_name}: {fmt_millions(val)}",
            None,
        ])
        values.append(val)

    max_val = max(values) if values else 1
    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats,
        mode="lines",
        line=dict(width=1.0, color="rgba(100, 181, 246, 0.4)"),
        hoverinfo="text",
        text=hover_texts,
    ))

    # Add country nodes
    summary = get_country_summary(year)
    node_lats = []
    node_lons = []
    node_sizes = []
    node_texts = []
    node_colors = []

    for iso3, row in summary.head(80).iterrows():
        lat, lon = get_coords(iso3)
        if (lat, lon) == (0, 0):
            continue
        node_lats.append(lat)
        node_lons.append(lon)
        trade_b = row["total_trade"] / 1000
        node_sizes.append(max(4, min(40, trade_b ** 0.4 * 3)))
        node_colors.append(row["trade_balance"])
        name = row.get("country_name", iso3)
        node_texts.append(
            f"<b>{name}</b><br>"
            f"Total trade: {fmt_billions(row['total_trade']/1000)}<br>"
            f"Exports: {fmt_billions(row['total_exports']/1000)}<br>"
            f"Imports: {fmt_billions(row['total_imports']/1000)}<br>"
            f"Balance: {fmt_billions(row['trade_balance']/1000)}<br>"
            f"Partners: {int(row['export_partners'])} exp / {int(row['import_partners'])} imp"
        )

    fig.add_trace(go.Scattergeo(
        lon=node_lons, lat=node_lats,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="RdYlGn",
            colorbar=dict(title="Trade Balance<br>($M)", thickness=12, len=0.4),
            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
        ),
        hoverinfo="text",
        text=node_texts,
    ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#e8e5e0",
        showocean=True, oceancolor="#eef4fb",
        showcountries=True, countrycolor="#b8b4ae",
        showcoastlines=True, coastlinecolor="#b8b4ae",
        showframe=False,
        showlakes=True, lakecolor="#eef4fb",
    )
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        title=dict(text=f"Global Trade Flows — {year}", x=0.5,
                   font=dict(size=18, color="#1a1a2e")),
        paper_bgcolor="#f7f6f3",
        geo=dict(bgcolor="#f7f6f3"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    total = get_year_network_data(year)["trade_value_usd_millions"].sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Global Trade", fmt_billions(total / 1000))
    col2.metric("Active Countries", f"{summary.shape[0]}")
    col3.metric("Trade Links", f"{len(get_year_network_data(year)):,}")
    col4.metric("Flows Displayed", f"{len(top_flows):,}")

    # Top traders table
    st.subheader("Top 20 Trading Nations")
    top20 = summary.head(20).copy()
    top20["total_trade_B"] = (top20["total_trade"] / 1000).round(1)
    top20["exports_B"] = (top20["total_exports"] / 1000).round(1)
    top20["imports_B"] = (top20["total_imports"] / 1000).round(1)
    top20["balance_B"] = (top20["trade_balance"] / 1000).round(1)
    display_cols = ["country_name", "total_trade_B", "exports_B", "imports_B", "balance_B", "export_partners", "import_partners"]
    rename_map = {
        "country_name": "Country", "total_trade_B": "Total Trade ($B)",
        "exports_B": "Exports ($B)", "imports_B": "Imports ($B)",
        "balance_B": "Balance ($B)", "export_partners": "Export Partners",
        "import_partners": "Import Partners",
    }
    st.dataframe(
        top20[display_cols].rename(columns=rename_map).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()
    st.caption("**Next →** See how this network has changed over 20 years on the **Network Evolution** page. Or drill into a specific country's vulnerabilities on the **Country Dependencies** page.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1B: 3D FORCE-DIRECTED NETWORK
# ═════════════════════════════════════════════════════════════════════════════
elif page == "3D Network Topology":
    st.title("3D Force-Directed Trade Network")
    st.markdown(
        "Interactive 3D graph where **distance encodes trade intensity** (Fruchterman-Reingold spring embedding). "
        "Nodes are colored by **Louvain community** and sized by **PageRank centrality**."
    )
    st.info(
        "**Why this matters:** Force-directed layouts reveal community structure that adjacency matrices hide. "
        "Countries that cluster together trade more intensely — and the 3D layout exposes which economies "
        "bridge otherwise separate trade blocs."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        year_3d = st.select_slider("Year", options=all_years, value=max(all_years), key="3d_year")
    with col2:
        n_countries_3d = st.slider("Countries", 20, 120, 60, 5, key="3d_n")
    with col3:
        edge_threshold_pct = st.slider("Edge visibility (top %)", 1, 100, 15, 1, key="3d_edges",
                                        help="Show only the top X% of edges by trade value")

    trade_3d = get_year_network_data(year_3d)
    graph_data_3d = build_trade_graph(trade_3d)

    with st.spinner("Computing 3D spring layout..."):
        layout = compute_3d_force_layout(graph_data_3d, top_n=n_countries_3d)

    if layout["nodes"]:
        nodes_df = pd.DataFrame(layout["nodes"])
        edges_df = pd.DataFrame(layout["edges"])

        # Community color palette — rich, vibrant colors
        community_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#D7BDE2",
        ]

        # Build 3D node scatter
        node_colors = [community_colors[n["community"] % len(community_colors)] for n in layout["nodes"]]
        pr_vals = nodes_df["pagerank"].values
        node_sizes = 8 + 60 * (pr_vals / pr_vals.max()) ** 0.6

        fig_3d = go.Figure()

        # Edges — filter to top % by weight
        if len(edges_df) > 0:
            weight_threshold = edges_df["weight"].quantile(1 - edge_threshold_pct / 100)
            visible_edges = edges_df[edges_df["weight"] >= weight_threshold]

            for _, e in visible_edges.iterrows():
                opacity = min(0.6, 0.1 + 0.5 * (e["weight"] / edges_df["weight"].max()))
                fig_3d.add_trace(go.Scatter3d(
                    x=[e["x0"], e["x1"]], y=[e["y0"], e["y1"]], z=[e["z0"], e["z1"]],
                    mode="lines",
                    line=dict(
                        color=f"rgba(100, 181, 246, {opacity:.2f})",
                        width=max(1, 4 * (e["weight"] / edges_df["weight"].max()) ** 0.3),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ))

        # Nodes
        hover_texts = [
            f"<b>{n['id']}</b> ({iso_to_name.get(n['id'], n['id'])})<br>"
            f"Community: {n['community']}<br>"
            f"PageRank: {n['pagerank']:.4f}<br>"
            f"Trade: {fmt_billions(n['strength']/1000)}"
            for n in layout["nodes"]
        ]

        fig_3d.add_trace(go.Scatter3d(
            x=nodes_df["x"], y=nodes_df["y"], z=nodes_df["z"],
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.92,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            text=[n["id"] for n in layout["nodes"]],
            textposition="top center",
            textfont=dict(size=9, color="#1a1a2e"),
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False,
        ))

        fig_3d.update_layout(
            height=700,
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title="", zeroline=False,
                          backgroundcolor="#f7f6f3"),
                yaxis=dict(showgrid=False, showticklabels=False, title="", zeroline=False,
                          backgroundcolor="#f7f6f3"),
                zaxis=dict(showgrid=False, showticklabels=False, title="", zeroline=False,
                          backgroundcolor="#f7f6f3"),
                bgcolor="#f7f6f3",
            ),
            paper_bgcolor="#f7f6f3",
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f"Trade Network — {year_3d} ({n_countries_3d} countries, {layout['n_communities']} communities)",
                font=dict(color="#1a1a2e", size=16),
                x=0.5,
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Community breakdown
        st.subheader("Community Structure (Louvain)")
        n_comm = layout["n_communities"]
        comm_cols = st.columns(min(n_comm, 6))
        for i, size in enumerate(layout["community_sizes"][:6]):
            members = [n["id"] for n in layout["nodes"] if n["community"] == i]
            with comm_cols[i % len(comm_cols)]:
                st.markdown(
                    f"<div style='background: {community_colors[i % len(community_colors)]}22; "
                    f"border-left: 4px solid {community_colors[i % len(community_colors)]}; "
                    f"padding: 10px; border-radius: 8px; margin-bottom: 8px;'>"
                    f"<b style='color: {community_colors[i % len(community_colors)]}'>Community {i}</b><br>"
                    f"<small>{size} countries</small><br>"
                    f"<small>{', '.join(members[:8])}{' ...' if len(members) > 8 else ''}</small></div>",
                    unsafe_allow_html=True,
                )

        # Top nodes by PageRank
        st.subheader("Network Centrality Rankings")
        top_nodes = nodes_df.nlargest(20, "pagerank").copy()
        top_nodes["country"] = top_nodes["id"].map(iso_to_name)
        top_nodes["total_trade_B"] = (top_nodes["strength"] / 1000).round(1)
        top_nodes["pagerank_pct"] = (top_nodes["pagerank"] * 100).round(3)
        top_nodes["community_color"] = top_nodes["community"].apply(
            lambda c: community_colors[c % len(community_colors)]
        )

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(
            y=top_nodes["country"], x=top_nodes["pagerank_pct"],
            orientation="h",
            marker=dict(
                color=top_nodes["community_color"].tolist(),
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
            ),
            hovertemplate="%{y}: PageRank=%{x:.3f}%<extra></extra>",
        ))
        fig_pr.update_layout(
            height=500, xaxis_title="PageRank (%)", yaxis_title="",
            title="Top 20 Countries by PageRank Centrality",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#1a1a2e"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()
    st.caption(
        "Quantify this community structure with graph invariants → **Network Invariants**. "
        "Or detect communities using algebraic topology → **Persistent Homology**."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: TARIFF IMPACT SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Tariff Impact Simulator":
    st.markdown(section_header("Tariff Impact Simulator",
                "Model how tariff shocks ripple through global trade networks"), unsafe_allow_html=True)
    st.markdown(
        "Model how tariff shocks ripple through global trade networks. "
        "Based on structural gravity trade elasticities "
        "(Caliendo & Parro 2015, Head & Mayer 2014)."
    )
    st.info("**Why this matters:** When one country imposes tariffs, trade doesn't disappear — it redirects. This simulator models how bilateral flows reorganize under tariff shocks, revealing winners (trade diversion beneficiaries) and losers (directly targeted exporters).")

    tab1, tab2 = st.tabs(["Single Country Shock", "Multi-Country Scenario"])

    # ── Tab 1: Single shock ──
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            imposing_name = st.selectbox(
                "Country imposing tariff",
                sorted(iso_to_name.values()),
                index=sorted(iso_to_name.values()).index("USA") if "USA" in name_to_iso else 0,
                key="single_imposing",
            )
            imposing_iso = name_to_iso[imposing_name]

        with col2:
            target_name = st.selectbox(
                "Country being tariffed",
                sorted(iso_to_name.values()),
                index=sorted(iso_to_name.values()).index("China") if "China" in name_to_iso else 0,
                key="single_target",
            )
            target_iso = name_to_iso[target_name]

        col3, col4, col5 = st.columns([1, 1, 1])
        with col3:
            tariff_pct = st.slider("Tariff increase (%)", 1, 200, 25, 1)
        with col4:
            elasticity = st.slider("Trade elasticity (ε)", 1.0, 12.0, DEFAULT_ELASTICITY, 0.5,
                                   help="Standard estimate: ~5. Higher = more trade-sensitive.")
        with col5:
            year = st.select_slider("Base year", options=all_years, value=max(all_years), key="tariff_year")

        bilateral = st.checkbox("Include retaliation (symmetric tariff)", value=True)

        trade_year = get_year_network_data(year)
        result = simulate_tariff_shock(
            trade_year, target_iso, imposing_iso, tariff_pct, elasticity, bilateral=bilateral
        )

        # Display results
        st.divider()
        st.subheader(f"Impact of {tariff_pct}% tariff: {imposing_name} ↔ {target_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bilateral Trade Before", fmt_millions(result["total_trade_before"]))
        c2.metric("Bilateral Trade After", fmt_millions(result["total_trade_after"]))
        c3.metric("Trade Destroyed", fmt_millions(result["total_bilateral_loss"]),
                  delta=f"-{result['pct_reduction']:.1f}%", delta_color="inverse")
        c4.metric(
            "Import Loss",
            fmt_millions(result["import_loss"]),
            delta=f"exports: -{fmt_millions(result['export_loss'])}" if bilateral else None,
            delta_color="inverse",
        )

        # Diversion charts
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(f"**Import Diversion** — Who replaces {target_name}?")
            div_imp = result["import_diversion"].head(10).copy()
            if len(div_imp) > 0:
                div_imp["country_name"] = div_imp["country"].map(iso_to_name)
                fig_div = px.bar(
                    div_imp, x="trade_gain", y="country_name",
                    orientation="h", color="trade_gain",
                    color_continuous_scale="Greens",
                    labels={"trade_gain": "Trade Gain ($M)", "country_name": ""},
                )
                fig_div.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_div, use_container_width=True)
            else:
                st.info("No import diversion data")

        with col_right:
            if bilateral:
                st.markdown(f"**Export Diversion** — Where do {imposing_name}'s exports go instead?")
                div_exp = result["export_diversion"].head(10).copy()
                if len(div_exp) > 0:
                    div_exp["country_name"] = div_exp["country"].map(iso_to_name)
                    fig_div2 = px.bar(
                        div_exp, x="trade_gain", y="country_name",
                        orientation="h", color="trade_gain",
                        color_continuous_scale="Blues",
                        labels={"trade_gain": "Trade Gain ($M)", "country_name": ""},
                    )
                    fig_div2.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig_div2, use_container_width=True)
                else:
                    st.info("No export diversion data")

    # ── Tab 2: Multi-country scenario ──
    with tab2:
        st.markdown("### Pre-built Tariff Scenarios")

        scenario_name = st.selectbox("Select scenario", list(SCENARIOS.keys()))
        scenario = SCENARIOS[scenario_name]

        col1, col2 = st.columns([1, 1])
        with col1:
            imposing_name_multi = st.selectbox(
                "Country imposing tariffs",
                sorted(iso_to_name.values()),
                index=sorted(iso_to_name.values()).index("USA") if "USA" in name_to_iso else 0,
                key="multi_imposing",
            )
            imposing_iso_multi = name_to_iso[imposing_name_multi]
        with col2:
            elasticity_multi = st.slider("Trade elasticity", 1.0, 12.0, DEFAULT_ELASTICITY, 0.5,
                                         key="multi_elast")

        year_multi = st.select_slider("Base year", options=all_years, value=max(all_years), key="multi_year")

        # Show scenario table
        scenario_df = pd.DataFrame([
            {"Country": iso_to_name.get(k, k), "Tariff (%)": v}
            for k, v in scenario.items()
        ])
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        trade_year_multi = get_year_network_data(year_multi)
        multi_result = simulate_multi_country_tariff(
            trade_year_multi, imposing_iso_multi, scenario, elasticity_multi
        )

        total_loss = multi_result["trade_loss"].sum()
        total_before = multi_result["old_bilateral_trade"].sum()

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Trade at Risk", fmt_billions(total_before / 1000))
        c2.metric("Total Trade Destroyed", fmt_billions(total_loss / 1000),
                  delta=f"-{total_loss/total_before*100:.1f}%" if total_before > 0 else None,
                  delta_color="inverse")
        c3.metric("Countries Affected", f"{len(multi_result)}")

        multi_result["target_name"] = multi_result["target"].map(iso_to_name)
        fig_multi = px.bar(
            multi_result.sort_values("trade_loss"),
            x="trade_loss_billions", y="target_name",
            orientation="h",
            color="tariff_pct",
            color_continuous_scale="Reds",
            labels={
                "trade_loss_billions": "Trade Loss ($B)",
                "target_name": "",
                "tariff_pct": "Tariff %",
            },
            title=f"Trade Losses by Country — {scenario_name}",
        )
        fig_multi.update_layout(height=450)
        st.plotly_chart(fig_multi, use_container_width=True)

        # Detail table
        display = multi_result[["target_name", "tariff_pct", "old_bilateral_trade", "new_bilateral_trade", "trade_loss", "pct_reduction"]].copy()
        display.columns = ["Country", "Tariff %", "Trade Before ($M)", "Trade After ($M)", "Trade Loss ($M)", "% Reduction"]
        display["Trade Before ($M)"] = display["Trade Before ($M)"].round(0)
        display["Trade After ($M)"] = display["Trade After ($M)"].round(0)
        display["Trade Loss ($M)"] = display["Trade Loss ($M)"].round(0)
        display["% Reduction"] = display["% Reduction"].round(1)
        st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.divider()
    st.caption("**Next →** For rigorous welfare estimates with GE effects on the **GE Counterfactual Lab** page. For a comparison across elasticity specifications → **Research Lab** → Elasticity Sensitivity.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: COUNTRY DEPENDENCIES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Country Dependencies":
    st.markdown(section_header("Country Trade Dependency Analysis",
                "Concentration risk, sector breakdown, and partner diversification"), unsafe_allow_html=True)
    st.info("**Why this matters:** A country with an HHI above 0.25 in critical imports is dangerously concentrated — a single supply disruption can cascade. This page reveals which countries are most vulnerable and which sectors create that vulnerability.")

    col1, col2 = st.columns([1, 1])
    with col1:
        country_name = st.selectbox(
            "Select country",
            sorted(iso_to_name.values()),
            index=sorted(iso_to_name.values()).index("USA") if "USA" in name_to_iso else 0,
        )
        country_iso = name_to_iso[country_name]
    with col2:
        year = st.select_slider("Year", options=all_years, value=max(all_years), key="dep_year")

    trade_year = get_year_network_data(year)
    deps = compute_dependency_metrics(trade_year, country_iso)

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Exports", fmt_billions(deps["total_exports"] / 1000))
    c2.metric("Total Imports", fmt_billions(deps["total_imports"] / 1000))
    c3.metric("Trade Balance", fmt_billions(deps["trade_balance"] / 1000))
    c4.metric("Export HHI", f"{deps['export_hhi']:.4f}",
              help="Herfindahl-Hirschman Index: 0=diversified, 1=concentrated")
    c5.metric("Import HHI", f"{deps['import_hhi']:.4f}")

    # Concentration interpretation
    for metric_name, hhi_val in [("Export", deps["export_hhi"]), ("Import", deps["import_hhi"])]:
        if hhi_val > 0.25:
            st.warning(f"{metric_name} markets are highly concentrated (HHI={hhi_val:.3f}). "
                      f"Significant vulnerability to trade disruptions.")
        elif hhi_val > 0.15:
            st.info(f"{metric_name} markets are moderately concentrated (HHI={hhi_val:.3f}).")

    # Export and import dependency charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Export Dependencies")
        exp_dep = deps["export_dependency"].head(15).copy()
        if len(exp_dep) > 0:
            exp_dep["partner_name"] = exp_dep["partner"].map(iso_to_name)
            exp_dep["share_pct"] = (exp_dep["export_share"] * 100).round(1)
            fig_exp = px.bar(
                exp_dep, x="export_value", y="partner_name",
                orientation="h", color="share_pct",
                color_continuous_scale="YlOrRd",
                labels={"export_value": "Export Value ($M)", "partner_name": "", "share_pct": "Share %"},
            )
            fig_exp.update_layout(height=450, coloraxis_showscale=True)
            st.plotly_chart(fig_exp, use_container_width=True)

    with col_right:
        st.subheader("Import Dependencies")
        imp_dep = deps["import_dependency"].head(15).copy()
        if len(imp_dep) > 0:
            imp_dep["partner_name"] = imp_dep["partner"].map(iso_to_name)
            imp_dep["share_pct"] = (imp_dep["import_share"] * 100).round(1)
            fig_imp = px.bar(
                imp_dep, x="import_value", y="partner_name",
                orientation="h", color="share_pct",
                color_continuous_scale="YlGnBu",
                labels={"import_value": "Import Value ($M)", "partner_name": "", "share_pct": "Share %"},
            )
            fig_imp.update_layout(height=450, coloraxis_showscale=True)
            st.plotly_chart(fig_imp, use_container_width=True)

    # Treemap of trade partners
    st.subheader("Trade Partner Composition")
    exp_data = deps["export_dependency"].head(25).copy()
    imp_data = deps["import_dependency"].head(25).copy()
    if len(exp_data) > 0 and len(imp_data) > 0:
        exp_data["type"] = "Exports"
        exp_data["value"] = exp_data["export_value"]
        exp_data["partner_name"] = exp_data["partner"].map(iso_to_name)
        imp_data["type"] = "Imports"
        imp_data["value"] = imp_data["import_value"]
        imp_data["partner_name"] = imp_data["partner"].map(iso_to_name)
        treemap_data = pd.concat([
            exp_data[["type", "partner_name", "value"]],
            imp_data[["type", "partner_name", "value"]],
        ])
        fig_tree = px.treemap(
            treemap_data, path=["type", "partner_name"], values="value",
            color="value", color_continuous_scale="Viridis",
        )
        fig_tree.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)

    # Sector breakdown
    st.subheader("Sector Breakdown (2019)")
    sector_flows = load_sector_flows()
    country_sectors_exp = sector_flows[sector_flows["iso_o"] == country_iso].groupby("oecd_sector")["value"].sum()
    country_sectors_imp = sector_flows[sector_flows["iso_d"] == country_iso].groupby("oecd_sector")["value"].sum()

    if len(country_sectors_exp) > 0:
        sec_df = pd.DataFrame({
            "Exports": country_sectors_exp,
            "Imports": country_sectors_imp,
        }).fillna(0)
        sec_df.index = sec_df.index.map(lambda x: SECTOR_LABELS.get(x, x))
        sec_df = sec_df.sort_values("Exports", ascending=True)

        fig_sec = go.Figure()
        fig_sec.add_trace(go.Bar(y=sec_df.index, x=sec_df["Exports"], name="Exports",
                                 orientation="h", marker_color="#1e3a5f"))
        fig_sec.add_trace(go.Bar(y=sec_df.index, x=sec_df["Imports"], name="Imports",
                                 orientation="h", marker_color="#2563eb"))
        fig_sec.update_layout(
            barmode="group", height=600,
            xaxis_title="Trade Value ($M)", yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sec, use_container_width=True)
    st.divider()
    st.caption("**Next →** Explore the US case in depth on the **US Trade Exposure** page. Or see how tariffs would reshape these dependencies on the **Tariff Impact Simulator** page.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: NETWORK EVOLUTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Network Evolution":
    st.markdown(section_header("Trade Network Evolution",
                "20 years of structural change in global trade (2002–2022)"), unsafe_allow_html=True)
    st.markdown("How the structure of global trade has changed over two decades.")
    st.info("**Why this matters:** Trade networks aren't static — they rewire after financial crises, pandemics, and geopolitical shifts. These 20-year trends reveal whether global trade is becoming more or less concentrated, resilient, and interconnected.")

    with st.spinner("Computing network statistics over time..."):
        stats = compute_network_stats_over_time(all_years, trade_full)

    # Key metrics over time
    fig_stats = go.Figure()
    fig_stats.add_trace(go.Scatter(
        x=stats["year"], y=stats["total_trade_billions"],
        mode="lines+markers", name="Total Trade ($B)",
        line=dict(color="#1e3a5f", width=3),
    ))
    fig_stats.update_layout(
        title="Total Global Trade Volume",
        height=350, xaxis_title="Year", yaxis_title="$ Billions",
    )
    st.plotly_chart(fig_stats, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_density = go.Figure()
        fig_density.add_trace(go.Scatter(
            x=stats["year"], y=stats["density"],
            mode="lines+markers", name="Network Density",
            line=dict(color="#2563eb", width=2),
        ))
        fig_density.update_layout(
            title="Network Density (Connectedness)",
            height=300, xaxis_title="Year", yaxis_title="Density",
        )
        st.plotly_chart(fig_density, use_container_width=True)

    with col2:
        fig_conc = go.Figure()
        fig_conc.add_trace(go.Scatter(
            x=stats["year"], y=stats["top10_concentration"],
            mode="lines+markers", name="Top 10 Concentration",
            line=dict(color="#dc2626", width=2),
        ))
        fig_conc.update_layout(
            title="Top 10 Countries Share of Global Trade",
            height=300, xaxis_title="Year", yaxis_title="Share",
        )
        st.plotly_chart(fig_conc, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_links = go.Figure()
        fig_links.add_trace(go.Scatter(
            x=stats["year"], y=stats["trade_links"],
            mode="lines+markers", name="Trade Links",
            line=dict(color="#16a34a", width=2),
        ))
        fig_links.update_layout(
            title="Number of Active Trade Links",
            height=300, xaxis_title="Year", yaxis_title="Bilateral Links",
        )
        st.plotly_chart(fig_links, use_container_width=True)

    with col4:
        fig_clust = go.Figure()
        fig_clust.add_trace(go.Scatter(
            x=stats["year"], y=stats["avg_clustering"],
            mode="lines+markers", name="Clustering",
            line=dict(color="#7c3aed", width=2),
        ))
        fig_clust.update_layout(
            title="Average Clustering Coefficient",
            height=300, xaxis_title="Year", yaxis_title="Clustering",
        )
        st.plotly_chart(fig_clust, use_container_width=True)

    # Network centrality comparison: pick two years
    st.divider()
    st.subheader("Centrality Shift: Compare Two Years")
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        year1 = st.selectbox("Year 1", all_years, index=0)
    with col_y2:
        year2 = st.selectbox("Year 2", all_years, index=len(all_years) - 1)

    with st.spinner("Computing centrality..."):
        g1_data = build_trade_graph(get_year_network_data(year1))
        g2_data = build_trade_graph(get_year_network_data(year2))
        cent1 = compute_centrality_measures(g1_data)
        cent2 = compute_centrality_measures(g2_data)

    # PageRank comparison
    if len(cent1) > 0 and len(cent2) > 0:
        merged = cent1[["pagerank"]].rename(columns={"pagerank": f"pagerank_{year1}"}).join(
            cent2[["pagerank"]].rename(columns={"pagerank": f"pagerank_{year2}"}),
            how="outer",
        ).fillna(0)
        merged["change"] = merged[f"pagerank_{year2}"] - merged[f"pagerank_{year1}"]
        merged["country_name"] = merged.index.map(iso_to_name)

        top_risers = merged.nlargest(10, "change")
        top_fallers = merged.nsmallest(10, "change")

        col_rise, col_fall = st.columns(2)
        with col_rise:
            st.markdown(f"**Biggest Risers ({year1}→{year2})**")
            fig_rise = px.bar(
                top_risers, x="change", y="country_name",
                orientation="h", color="change",
                color_continuous_scale="Greens",
                labels={"change": "PageRank Change", "country_name": ""},
            )
            fig_rise.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_rise, use_container_width=True)

        with col_fall:
            st.markdown(f"**Biggest Fallers ({year1}→{year2})**")
            fig_fall = px.bar(
                top_fallers, x="change", y="country_name",
                orientation="h", color="change",
                color_continuous_scale="Reds_r",
                labels={"change": "PageRank Change", "country_name": ""},
            )
            fig_fall.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_fall, use_container_width=True)
    st.divider()
    st.caption("**Next →** Investigate a specific country's exposure on the **Country Dependencies** page. Or see the topological signature of these structural changes on the **Topological Evolution** page.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4B: NETWORK INVARIANTS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Network Invariants":
    st.title("Comprehensive Network Invariants")
    st.markdown(
        "A complete suite of graph-theoretic measures for the global trade network. "
        "These invariants characterize the network's structure, hierarchy, and vulnerability."
    )
    st.info(
        "**Why this matters:** Standard trade statistics (total volume, top partners) miss structural "
        "properties. Graph invariants — clustering, betweenness, spectral gap — reveal whether the trade "
        "network is resilient, hierarchical, or fragile. These are the same measures used in infrastructure "
        "and epidemiological network analysis."
    )

    year_inv = st.select_slider("Year", options=all_years, value=max(all_years), key="inv_year")
    trade_inv = get_year_network_data(year_inv)
    graph_inv = build_trade_graph(trade_inv)

    with st.spinner("Computing comprehensive network invariants..."):
        inv = compute_comprehensive_network_invariants(graph_inv)
        cent = compute_centrality_measures(graph_inv)
        comms = detect_communities(graph_inv)

    if inv:
        # ── Section 1: Structural Overview ──
        st.subheader("Structural Overview")
        cols = st.columns(6)
        labels_vals = [
            ("Nodes (Countries)", inv["n_nodes"]),
            ("Edges (Trade Links)", f"{inv['n_edges']:,}"),
            ("Network Density", f"{inv['density']:.4f}"),
            ("Reciprocity", f"{inv['reciprocity']:.3f}"),
            ("Transitivity", f"{inv['transitivity']:.3f}"),
            ("Max Core Number", inv["max_core_number"]),
        ]
        for i, (label, val) in enumerate(labels_vals):
            cols[i].metric(label, val)

        # ── Section 2: Connectivity ──
        st.subheader("Connectivity & Paths")
        cols2 = st.columns(5)
        cols2[0].metric("Strongly Connected Components", inv["n_strongly_connected"])
        cols2[1].metric("Weakly Connected Components", inv["n_weakly_connected"])
        cols2[2].metric("Largest SCC Size", inv["largest_scc_size"])
        cols2[3].metric("Avg Path Length (SCC)", f"{inv['avg_path_length']:.2f}" if inv['avg_path_length'] != float('inf') else "N/A")
        cols2[4].metric("Diameter", inv["diameter"] if inv["diameter"] > 0 else "N/A")

        # ── Section 3: Degree Distribution ──
        st.subheader("Degree & Strength Distribution")
        cols3 = st.columns(5)
        cols3[0].metric("Avg In-Degree", f"{inv['avg_in_degree']:.1f}")
        cols3[1].metric("Avg Out-Degree", f"{inv['avg_out_degree']:.1f}")
        cols3[2].metric("Max In-Degree", inv["max_in_degree"])
        cols3[3].metric("Max Out-Degree", inv["max_out_degree"])
        cols3[4].metric("Degree Assortativity", f"{inv['degree_assortativity']:.3f}",
                        help="Positive = hubs connect to hubs; Negative = hubs connect to periphery")

        # ── Section 4: Inequality ──
        st.subheader("Trade Inequality & Hierarchy")
        cols4 = st.columns(4)
        cols4[0].metric("Trade Gini Coefficient", f"{inv['trade_gini']:.3f}",
                        help="0=perfect equality, 1=one country has all trade")
        cols4[1].metric("Mean Trade Strength", fmt_billions(inv["mean_strength"] / 1000))
        cols4[2].metric("Median Trade Strength", fmt_billions(inv["median_strength"] / 1000))
        cols4[3].metric("Avg Clustering", f"{inv['avg_clustering']:.4f}")

        # ── Interpretation panel ──
        st.divider()
        st.subheader("Interpretation")

        interpretations = []
        if inv["reciprocity"] > 0.5:
            interpretations.append("**High reciprocity** — most trade links are bidirectional, indicating balanced partnerships.")
        else:
            interpretations.append("**Moderate/low reciprocity** — many one-directional trade flows, suggesting asymmetric dependencies.")

        if inv["degree_assortativity"] < -0.1:
            interpretations.append("**Disassortative network** — large trading nations tend to trade with smaller ones (hub-and-spoke structure).")
        elif inv["degree_assortativity"] > 0.1:
            interpretations.append("**Assortative network** — large trading nations preferentially trade with each other (rich club).")

        if inv["trade_gini"] > 0.7:
            interpretations.append(f"**Very high trade inequality** (Gini={inv['trade_gini']:.3f}) — trade is heavily concentrated among a few nations.")

        if inv["transitivity"] > 0.5:
            interpretations.append(f"**High transitivity** ({inv['transitivity']:.3f}) — if A trades with B and C, B and C likely also trade. Strong multilateral structure.")

        if inv["n_strongly_connected"] == 1:
            interpretations.append("**Fully connected** — every country can reach every other through directed trade paths.")

        for interp in interpretations:
            st.markdown(f"- {interp}")

        # ── Centrality distributions (visual) ──
        st.divider()
        st.subheader("Centrality Distributions")

        if len(cent) > 0:
            cent_reset = cent.reset_index()
            cent_reset["country"] = cent_reset["iso3"].map(iso_to_name)
            cent_reset["community"] = cent_reset["iso3"].map(comms)

            col_left, col_right = st.columns(2)

            with col_left:
                # Eigenvector vs PageRank scatter
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=cent_reset["eigenvector_centrality"],
                    y=cent_reset["pagerank"],
                    mode="markers+text",
                    text=cent_reset["iso3"].where(cent_reset["pagerank"] > cent_reset["pagerank"].quantile(0.85)),
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(
                        size=8 + 30 * (cent_reset["total_strength"] / cent_reset["total_strength"].max()) ** 0.5,
                        color=cent_reset["community"],
                        colorscale="Turbo",
                        opacity=0.8,
                        line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                    ),
                    hovertext=cent_reset["country"],
                    hoverinfo="text",
                ))
                fig_scatter.update_layout(
                    height=450, title="Eigenvector vs PageRank Centrality",
                    xaxis_title="Eigenvector Centrality",
                    yaxis_title="PageRank",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(color="#1a1a2e"),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_right:
                # Betweenness centrality — who are the bridges?
                top_between = cent_reset.nlargest(20, "betweenness_centrality")
                fig_btwn = go.Figure()
                fig_btwn.add_trace(go.Bar(
                    y=top_between["country"], x=top_between["betweenness_centrality"],
                    orientation="h",
                    marker=dict(
                        color=top_between["betweenness_centrality"],
                        colorscale="Plasma",
                    ),
                ))
                fig_btwn.update_layout(
                    height=450, title="Top 20 Bridge Nations (Betweenness)",
                    xaxis_title="Betweenness Centrality", yaxis_title="",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(color="#1a1a2e"),
                )
                st.plotly_chart(fig_btwn, use_container_width=True)

            # Strength distribution (log-log)
            st.subheader("Trade Strength Distribution")
            st.markdown("Power-law? Log-log plot of node strength (total trade) vs. rank reveals the hierarchy.")
            strengths_sorted = cent_reset["total_strength"].sort_values(ascending=False).values
            fig_zipf = go.Figure()
            fig_zipf.add_trace(go.Scatter(
                x=list(range(1, len(strengths_sorted) + 1)),
                y=strengths_sorted / 1000,
                mode="markers+lines",
                marker=dict(size=5, color="#2563eb"),
                line=dict(color="rgba(100, 181, 246, 0.5)", width=1),
            ))
            fig_zipf.update_layout(
                height=400,
                xaxis=dict(type="log", title="Rank"),
                yaxis=dict(type="log", title="Total Trade ($B)"),
                title="Zipf Plot: Trade Strength Distribution",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(color="#1a1a2e"),
            )
            st.plotly_chart(fig_zipf, use_container_width=True)

        # Full data export
        with st.expander("Full centrality data (all countries)"):
            if len(cent) > 0:
                export_df = cent_reset.copy()
                export_df["total_trade_B"] = (export_df["total_strength"] / 1000).round(1)
                st.dataframe(
                    export_df[["country", "iso3", "pagerank", "eigenvector_centrality",
                               "betweenness_centrality", "degree_centrality", "total_trade_B", "community"]].round(5),
                    use_container_width=True, hide_index=True,
                )

    st.divider()
    st.caption(
        "See these measures evolve over time → **Network Evolution**. "
        "Or explore the network's topology beyond pairwise connections → **Persistent Homology**."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: US TRADE EXPOSURE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "US Trade Exposure":
    st.markdown(section_header("US Trade Exposure Analysis",
                "Import concentration, sector vulnerabilities, and the 2025–26 tariff context"), unsafe_allow_html=True)
    st.markdown(
        "Deep dive into US trade dependencies and vulnerability to tariff shocks. "
        "Critical context for understanding the 2025–26 tariff regime."
    )
    st.info("**Why this matters:** Critical context for the 2025–26 tariff regime. The US imports over $3 trillion annually — this page shows where that exposure is concentrated, which sectors depend on single-source suppliers, and where diversification is weakest.")

    year = st.select_slider("Year", options=all_years, value=max(all_years), key="us_year")
    trade_year = get_year_network_data(year)
    us_deps = compute_dependency_metrics(trade_year, "USA")

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("US Exports", fmt_billions(us_deps["total_exports"] / 1000))
    c2.metric("US Imports", fmt_billions(us_deps["total_imports"] / 1000))
    c3.metric("Trade Deficit", fmt_billions(abs(us_deps["trade_balance"]) / 1000))
    c4.metric("Import Concentration (HHI)", f"{us_deps['import_hhi']:.4f}")

    st.divider()

    # US imports by country — where the US depends
    st.subheader("Where US Imports Come From")
    imp = us_deps["import_dependency"].head(20).copy()
    imp["partner_name"] = imp["partner"].map(iso_to_name)
    imp["share_pct"] = (imp["import_share"] * 100).round(1)

    fig_us_imp = px.bar(
        imp, x="import_value", y="partner_name",
        orientation="h", color="share_pct",
        color_continuous_scale="YlOrRd",
        labels={"import_value": "Import Value ($M)", "partner_name": "", "share_pct": "Share %"},
        title="Top 20 US Import Sources",
    )
    fig_us_imp.update_layout(height=500)
    st.plotly_chart(fig_us_imp, use_container_width=True)

    # Tariff scenario comparison
    st.divider()
    st.subheader("Tariff Scenario Impact on US Trade")

    scenario_results = {}
    for name, scenario in SCENARIOS.items():
        result = simulate_multi_country_tariff(trade_year, "USA", scenario)
        scenario_results[name] = {
            "total_loss_B": result["trade_loss"].sum() / 1000,
            "avg_reduction": result["pct_reduction"].mean(),
        }

    scenario_comp = pd.DataFrame(scenario_results).T
    scenario_comp.columns = ["Total Trade Loss ($B)", "Avg % Reduction"]

    fig_scenario = px.bar(
        scenario_comp.reset_index().rename(columns={"index": "Scenario"}),
        x="Total Trade Loss ($B)", y="Scenario",
        orientation="h", color="Total Trade Loss ($B)",
        color_continuous_scale="Reds",
        title="Estimated Trade Losses by Scenario",
    )
    fig_scenario.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig_scenario, use_container_width=True)

    st.dataframe(
        scenario_comp.round(1).reset_index().rename(columns={"index": "Scenario"}),
        use_container_width=True, hide_index=True,
    )

    # China-specific deep dive
    st.divider()
    st.subheader("China Dependency Deep Dive")

    china_exports_to_us = trade_year[
        (trade_year["iso_o"] == "CHN") & (trade_year["iso_d"] == "USA")
    ]["trade_value_usd_millions"].sum()
    us_exports_to_china = trade_year[
        (trade_year["iso_o"] == "USA") & (trade_year["iso_d"] == "CHN")
    ]["trade_value_usd_millions"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("US Imports from China", fmt_billions(china_exports_to_us / 1000))
    c2.metric("US Exports to China", fmt_billions(us_exports_to_china / 1000))
    c3.metric("US-China Deficit", fmt_billions((china_exports_to_us - us_exports_to_china) / 1000))

    # Simulate range of China tariffs
    china_tariff_range = list(range(0, 105, 5))
    china_impacts = []
    for t in china_tariff_range:
        if t == 0:
            china_impacts.append({"tariff": t, "trade_loss": 0, "bilateral_trade": china_exports_to_us + us_exports_to_china})
        else:
            r = simulate_tariff_shock(trade_year, "CHN", "USA", t)
            china_impacts.append({
                "tariff": t,
                "trade_loss": r["total_bilateral_loss"],
                "bilateral_trade": r["total_trade_after"],
            })

    china_df = pd.DataFrame(china_impacts)
    fig_china = go.Figure()
    fig_china.add_trace(go.Scatter(
        x=china_df["tariff"], y=china_df["bilateral_trade"] / 1000,
        mode="lines+markers", name="Remaining Bilateral Trade ($B)",
        line=dict(color="#dc2626", width=3),
    ))
    fig_china.add_trace(go.Scatter(
        x=china_df["tariff"], y=china_df["trade_loss"] / 1000,
        mode="lines+markers", name="Cumulative Trade Loss ($B)",
        line=dict(color="#1e3a5f", width=2, dash="dash"),
    ))
    fig_china.update_layout(
        title="US-China Trade vs. Tariff Level",
        xaxis_title="Tariff (%)", yaxis_title="$ Billions",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    # Add markers for key tariff levels
    for level, label in [(25, "25%"), (60, "~Current"), (100, "Decoupling")]:
        row = china_df[china_df["tariff"] == level].iloc[0]
        fig_china.add_annotation(
            x=level, y=row["bilateral_trade"] / 1000,
            text=label, showarrow=True, arrowhead=2,
        )
    st.plotly_chart(fig_china, use_container_width=True)

    # Sector exposure
    st.divider()
    st.subheader("US Sector Exposure to Tariffs (2019)")
    sector_flows = load_sector_flows()
    us_imp_sectors = sector_flows[
        (sector_flows["iso_d"] == "USA")
    ].groupby(["iso_o", "oecd_sector"])["value"].sum().reset_index()

    # Top sources for each sector
    tariffed_countries = list(SCENARIOS["Trump 2025-26 Tariffs (Approximate)"].keys())
    us_imp_tariffed = us_imp_sectors[us_imp_sectors["iso_o"].isin(tariffed_countries)]
    us_imp_tariffed_by_sector = us_imp_tariffed.groupby("oecd_sector")["value"].sum().sort_values(ascending=False)
    us_imp_total_by_sector = us_imp_sectors.groupby("oecd_sector")["value"].sum()

    if len(us_imp_tariffed_by_sector) > 0:
        exposure = pd.DataFrame({
            "tariffed_imports": us_imp_tariffed_by_sector,
            "total_imports": us_imp_total_by_sector,
        }).fillna(0)
        exposure["exposure_pct"] = (exposure["tariffed_imports"] / exposure["total_imports"] * 100).round(1)
        exposure.index = exposure.index.map(lambda x: SECTOR_LABELS.get(x, x))
        exposure = exposure.sort_values("tariffed_imports", ascending=True)

        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(
            y=exposure.index, x=exposure["tariffed_imports"],
            orientation="h", name="From Tariffed Countries",
            marker_color="#dc2626",
        ))
        fig_exp.add_trace(go.Bar(
            y=exposure.index, x=exposure["total_imports"] - exposure["tariffed_imports"],
            orientation="h", name="From Other Countries",
            marker_color="#b8b4ae",
        ))
        fig_exp.update_layout(
            barmode="stack", height=600,
            title="US Imports by Sector: Tariffed vs. Non-Tariffed Sources",
            xaxis_title="Import Value ($M)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_exp, use_container_width=True)
    st.divider()
    st.caption("**Next →** Model specific tariff scenarios on the **Tariff Impact Simulator** page. Or compute welfare impacts with a full GE model on the **GE Counterfactual Lab** page.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6: GRAVITY MODEL LAB
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Gravity Model Lab":
    st.title("Structural Gravity Model Lab")
    st.markdown(
        "Estimate trade elasticities directly from data using the structural gravity framework. "
        "Compare OLS (log-linear) vs. PPML (Santos Silva & Tenreyro 2006) estimators."
    )
    st.info(
        "**Why this matters:** The gravity model is the workhorse of trade economics — it explains "
        "bilateral flows using GDP, distance, language, colonial ties, and trade agreements. The PPML "
        "estimates here are the econometrically correct specification (Santos Silva & Tenreyro, 2006). "
        "The sector-level elasticities feed directly into the GE counterfactual model."
    )

    year_grav = st.select_slider("Estimation year", options=all_years, value=2019, key="grav_year")

    tab_ols, tab_ppml, tab_sector = st.tabs(["OLS Estimation", "PPML Estimation", "Sector Elasticities"])

    with tab_ols:
        st.markdown("### OLS on log-linearized gravity")
        st.latex(r"\ln X_{ij} = \alpha + \beta_1 \ln GDP_i + \beta_2 \ln GDP_j - \beta_3 \ln d_{ij} + \gamma Z_{ij} + \varepsilon_{ij}")
        st.caption("Note: OLS drops zero trade flows, creating a selection bias. PPML is preferred.")

        with st.spinner("Estimating OLS gravity model..."):
            ols = estimate_ols_gravity(year_grav)

        c1, c2, c3 = st.columns(3)
        c1.metric("R-squared", f"{ols['r_squared']:.4f}")
        c2.metric("Observations", f"{ols['n_obs']:,}")
        c3.metric("Distance Elasticity", f"{ols['distance_elasticity']:.3f}")

        results_df = ols["results"].copy()
        results_df["coefficient"] = results_df["coefficient"].round(4)
        results_df["std_error"] = results_df["std_error"].round(4)
        results_df["t_statistic"] = results_df["t_statistic"].round(2)
        results_df["stars"] = results_df["t_statistic"].abs().apply(
            lambda t: "***" if t > 2.576 else ("**" if t > 1.96 else ("*" if t > 1.645 else ""))
        )
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Coefficient plot
        plot_df = results_df[results_df["variable"] != "constant"].copy()
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            y=plot_df["variable"], x=plot_df["coefficient"],
            orientation="h",
            error_x=dict(type="data", array=plot_df["std_error"].values * 1.96),
            marker_color=["#dc2626" if c < 0 else "#1e3a5f" for c in plot_df["coefficient"]],
        ))
        fig_coef.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_coef.update_layout(
            title="OLS Coefficient Estimates (95% CI)", height=350,
            xaxis_title="Coefficient", yaxis_title="",
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with tab_ppml:
        st.markdown("### Poisson Pseudo-Maximum Likelihood (PPML)")
        st.latex(r"E[X_{ij}|Z] = \exp(Z_{ij} \cdot \beta)")
        st.markdown(
            "PPML includes zero trade flows and is consistent under heteroskedasticity. "
            "Santos Silva & Tenreyro (2006) show OLS on log trade is biased."
        )

        with st.spinner("Estimating PPML gravity model (IRLS)..."):
            ppml = estimate_ppml_gravity(year_grav)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pseudo R²", f"{ppml['pseudo_r_squared']:.4f}")
        c2.metric("Observations", f"{ppml['n_obs']:,}")
        c3.metric("Distance Elasticity", f"{ppml['distance_elasticity']:.3f}")
        c4.metric("Converged", f"{'Yes' if ppml['converged'] else 'No'} ({ppml['iterations']} iter)")

        ppml_df = ppml["results"].copy()
        ppml_df["coefficient"] = ppml_df["coefficient"].round(4)
        ppml_df["std_error"] = ppml_df["std_error"].round(4)
        ppml_df["t_statistic"] = ppml_df["t_statistic"].round(2)
        ppml_df["stars"] = ppml_df["t_statistic"].abs().apply(
            lambda t: "***" if t > 2.576 else ("**" if t > 1.96 else ("*" if t > 1.645 else ""))
        )
        st.dataframe(ppml_df, use_container_width=True, hide_index=True)

        # Side by side comparison
        st.subheader("OLS vs. PPML Comparison")
        comparison = ols["results"][["variable", "coefficient"]].rename(columns={"coefficient": "OLS"}).merge(
            ppml["results"][["variable", "coefficient"]].rename(columns={"coefficient": "PPML"}),
            on="variable",
        )
        comparison = comparison[comparison["variable"] != "constant"]
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="OLS", y=comparison["variable"], x=comparison["OLS"],
                                   orientation="h", marker_color="#1e3a5f"))
        fig_comp.add_trace(go.Bar(name="PPML", y=comparison["variable"], x=comparison["PPML"],
                                   orientation="h", marker_color="#2563eb"))
        fig_comp.update_layout(barmode="group", height=400, title="Coefficient Comparison")
        st.plotly_chart(fig_comp, use_container_width=True)

    with tab_sector:
        st.markdown("### Sector-Specific Trade Elasticities")
        st.markdown(
            "Estimated from sectoral tariff data (CEPII tau) merged with bilateral sector flows. "
            "These elasticities capture how responsive each sector is to trade cost changes."
        )

        with st.spinner("Estimating sector-level elasticities..."):
            sec_elast = estimate_sector_elasticities()

        if len(sec_elast) > 0:
            fig_sec = px.bar(
                sec_elast.sort_values("tariff_elasticity"),
                x="tariff_elasticity", y="sector_name",
                orientation="h", color="tariff_elasticity",
                color_continuous_scale="RdYlGn",
                labels={"tariff_elasticity": "Trade-Tariff Elasticity", "sector_name": ""},
                title="Sector Trade Elasticities (higher = more tariff-sensitive)",
            )
            fig_sec.update_layout(height=600, coloraxis_showscale=False)
            st.plotly_chart(fig_sec, use_container_width=True)

            st.dataframe(
                sec_elast[["sector_name", "tariff_elasticity", "distance_elasticity", "r_squared", "n_obs"]].round(3),
                use_container_width=True, hide_index=True,
            )
        else:
            st.warning("Could not estimate sector elasticities. Check data availability.")

    st.divider()
    st.caption(
        "These elasticities power the GE model → **GE Counterfactual Lab**. "
        "Or explore how well gravity predicts network structure → **Statistical Significance** (gravity null model)."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7: SUPPLY CHAIN DEEP DIVE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Supply Chain Deep Dive":
    st.title("Strategic Supply Chain Analysis")
    st.markdown(
        "Product-level (HS02) analysis of strategic supply chains. "
        "Identifies concentration risks and import dependencies at the commodity level."
    )
    st.info("**Why this matters:** Not all trade is equal — semiconductors, rare earths, and pharmaceuticals have outsized strategic importance. This page lets you trace specific HS product codes through global supply chains and identify chokepoints.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        product_group = st.selectbox(
            "Strategic Product Group",
            list(STRATEGIC_PRODUCTS.keys()),
        )
    with col2:
        focus_name = st.selectbox(
            "Focus Country",
            sorted(iso_to_name.values()),
            index=sorted(iso_to_name.values()).index("USA") if "USA" in name_to_iso else 0,
            key="supply_focus",
        )
        focus_iso = name_to_iso[focus_name]
    with col3:
        hs_years = sorted([
            int(f.stem.split("Y")[1])
            for f in (Path.home() / "trade_data_warehouse" / "baci" / "hs_by_dyad").iterdir()
            if f.suffix == ".parquet"
        ])
        year_hs = st.select_slider("Year", options=hs_years, value=max(hs_years), key="hs_year")

    st.markdown(f"**{STRATEGIC_PRODUCTS[product_group]['description']}**")

    with st.spinner(f"Analyzing {product_group} supply chain..."):
        result = get_strategic_product_flows(year_hs, product_group, focus_iso)

    if not result:
        st.error("No data available for this combination.")
    else:
        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Total Imports to {focus_name}", fmt_millions(result["total_imports"]))
        c2.metric("Import HHI (Concentration)", f"{result['import_hhi']:.4f}")
        c3.metric("Top Source", f"{iso_to_name.get(result['top_source'], result['top_source'])} ({result['top_source_share']*100:.1f}%)")
        c4.metric("Source Countries", f"{result['n_sources']}")

        if result["import_hhi"] > 0.25:
            st.error(f"CRITICAL: {product_group} imports are highly concentrated (HHI={result['import_hhi']:.3f}). "
                     "Severe supply chain vulnerability.")
        elif result["import_hhi"] > 0.15:
            st.warning(f"ELEVATED RISK: {product_group} imports are moderately concentrated.")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Import Sources")
            imp_src = result["imports_by_source"]
            if len(imp_src) > 0:
                imp_src.columns = ["country", "value"]
                imp_src["country_name"] = imp_src["country"].map(iso_to_name)
                imp_src = imp_src.head(15)
                fig_src = px.bar(
                    imp_src, x="value", y="country_name",
                    orientation="h", color="value",
                    color_continuous_scale="Reds",
                    labels={"value": "Import Value ($M)", "country_name": ""},
                )
                fig_src.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_src, use_container_width=True)

        with col_right:
            st.subheader("By Product")
            imp_prod = result["imports_by_product"]
            if len(imp_prod) > 0:
                imp_prod.columns = ["hs4", "product", "value"]
                fig_prod = px.bar(
                    imp_prod.head(15), x="value", y="product",
                    orientation="h", color="value",
                    color_continuous_scale="Blues",
                    labels={"value": "Import Value ($M)", "product": ""},
                )
                fig_prod.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_prod, use_container_width=True)

        # Global dominance map
        st.subheader("Global Export Leaders")
        global_exp = result["global_by_exporter"]
        if len(global_exp) > 0:
            global_exp.columns = ["country", "value"]
            global_exp["country_name"] = global_exp["country"].map(iso_to_name)
            global_exp["share_pct"] = (global_exp["value"] / global_exp["value"].sum() * 100).round(1)
            fig_global = px.treemap(
                global_exp.head(30), path=["country_name"], values="value",
                color="share_pct", color_continuous_scale="YlOrRd",
                title=f"Who Produces {product_group}? (Global Export Share)",
            )
            fig_global.update_layout(height=450)
            st.plotly_chart(fig_global, use_container_width=True)

        # Time evolution
        st.subheader(f"{product_group} Import Evolution")
        with st.spinner("Loading historical product data..."):
            evolution = get_product_evolution(product_group, focus_iso)

        if len(evolution) > 0:
            # Melt for plotting
            id_cols = ["year", "total_imports"]
            source_cols = [c for c in evolution.columns if c not in id_cols]
            fig_evo = go.Figure()
            fig_evo.add_trace(go.Scatter(
                x=evolution["year"], y=evolution["total_imports"],
                mode="lines+markers", name="Total",
                line=dict(width=3, color="black"),
            ))
            for col in source_cols[:5]:
                fig_evo.add_trace(go.Scatter(
                    x=evolution["year"], y=evolution[col],
                    mode="lines+markers",
                    name=iso_to_name.get(col, col),
                ))
            fig_evo.update_layout(
                title=f"{focus_name} {product_group} Imports Over Time",
                height=400, xaxis_title="Year", yaxis_title="Import Value ($M)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_evo, use_container_width=True)
    st.divider()
    st.caption("**Next →** See how tariffs affect these supply chains on the **Tariff Impact Simulator** page. Or explore the gravity model that explains why trade flows the way it does on the **Gravity Model Lab** page.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 8: WELFARE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Welfare Analysis":
    st.markdown(section_header("Welfare Impact Analysis",
                "Sufficient-statistics welfare — ACR (Arkolakis, Costinot, Rodríguez-Clare, 2012)"), unsafe_allow_html=True)
    st.markdown(
        "Welfare calculations using the **Arkolakis, Costinot & Rodriguez-Clare (2012)** "
        "sufficient statistic: $\\hat{W}_j = \\hat{\\lambda}_{jj}^{-1/\\varepsilon}$"
    )
    st.markdown(
        "This result holds across Armington, Eaton-Kortum, and Melitz models. "
        "Welfare change depends only on the change in domestic expenditure share "
        "and the trade elasticity."
    )
    st.info("**Why this matters:** The ACR (2012) formula gives a quick welfare estimate from trade shares and elasticities alone — no structural model needed. But it assumes a single sector and no tariff revenue. Compare these results with the multi-sector GE approach on the GE Counterfactual Lab.")

    st.divider()
    st.subheader("Scenario Welfare Comparison")

    col1, col2 = st.columns([1, 1])
    with col1:
        scenario_name = st.selectbox("Tariff Scenario", list(SCENARIOS.keys()), key="welfare_scenario")
    with col2:
        welfare_elast = st.slider("Trade Elasticity (ε)", 2.0, 12.0, 5.0, 0.5, key="welfare_elast")

    year_w = st.select_slider("Base Year", options=all_years, value=max(all_years), key="welfare_year")

    trade_w = get_year_network_data(year_w)

    with st.spinner("Computing welfare impacts across countries..."):
        welfare_multi = compute_multi_country_welfare(
            trade_w, SCENARIOS[scenario_name], welfare_elast,
        )

    # Summary for USA
    us_row = welfare_multi[welfare_multi["country"] == "USA"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("US Welfare Change", f"{us_row['welfare_change_pct']:.3f}%")
    c2.metric("US Dollar Impact", f"${us_row['welfare_dollars_M']/1000:,.1f}B" if pd.notna(us_row['welfare_dollars_M']) else "N/A",
              delta_color="inverse")
    c3.metric("US Domestic Share Change",
              f"{us_row['domestic_share_before']:.3f} → {us_row['domestic_share_after']:.3f}")

    # All countries
    welfare_multi["country_name"] = welfare_multi["country"].map(iso_to_name)
    welfare_multi["welfare_dollars_B"] = welfare_multi["welfare_dollars_M"] / 1000

    fig_welfare = px.bar(
        welfare_multi.sort_values("welfare_change_pct"),
        x="welfare_change_pct", y="country_name",
        orientation="h", color="role",
        color_discrete_map={"Imposing": "#dc2626", "Targeted": "#1e3a5f"},
        labels={"welfare_change_pct": "Welfare Change (%)", "country_name": "", "role": "Role"},
        title=f"Welfare Impact by Country — {scenario_name}",
    )
    fig_welfare.update_layout(height=450)
    st.plotly_chart(fig_welfare, use_container_width=True)

    # Dollar terms
    fig_dollars = px.bar(
        welfare_multi.dropna(subset=["welfare_dollars_B"]).sort_values("welfare_dollars_B"),
        x="welfare_dollars_B", y="country_name",
        orientation="h", color="welfare_dollars_B",
        color_continuous_scale="RdYlGn",
        labels={"welfare_dollars_B": "Welfare Impact ($B)", "country_name": ""},
        title="Welfare Impact in Dollar Terms",
    )
    fig_dollars.update_layout(height=450, coloraxis_showscale=False)
    st.plotly_chart(fig_dollars, use_container_width=True)

    # Elasticity sensitivity analysis
    st.divider()
    st.subheader("Sensitivity to Trade Elasticity")
    st.markdown(
        "The welfare impact is highly sensitive to the assumed trade elasticity. "
        "Lower elasticity → larger welfare losses (harder to substitute)."
    )

    elast_range = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    sensitivity = []
    for e in elast_range:
        us_welfare = compute_welfare_impact(
            trade_w, "USA", SCENARIOS[scenario_name], e, gdp=25_000_000
        )
        sensitivity.append({
            "elasticity": e,
            "welfare_pct": us_welfare["welfare_change_pct"],
            "welfare_B": us_welfare["welfare_change_dollars_M"] / 1000 if us_welfare["welfare_change_dollars_M"] else 0,
        })

    sens_df = pd.DataFrame(sensitivity)
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["elasticity"], y=sens_df["welfare_pct"],
        mode="lines+markers", name="Welfare Change (%)",
        line=dict(color="#dc2626", width=3),
    ))
    fig_sens.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sens.update_layout(
        title=f"US Welfare Loss vs. Trade Elasticity — {scenario_name}",
        xaxis_title="Trade Elasticity (ε)", yaxis_title="Welfare Change (%)",
        height=350,
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    # Detail table
    st.subheader("Detailed Results")
    display_w = welfare_multi[[
        "country_name", "role", "welfare_change_pct",
        "welfare_dollars_B", "domestic_share_before", "domestic_share_after"
    ]].copy()
    display_w.columns = ["Country", "Role", "Welfare Δ (%)", "Welfare Δ ($B)",
                         "λ_jj Before", "λ_jj After"]
    display_w["Welfare Δ (%)"] = display_w["Welfare Δ (%)"].round(3)
    display_w["Welfare Δ ($B)"] = display_w["Welfare Δ ($B)"].round(1)
    display_w["λ_jj Before"] = display_w["λ_jj Before"].round(4)
    display_w["λ_jj After"] = display_w["λ_jj After"].round(4)
    st.dataframe(display_w, use_container_width=True, hide_index=True)

    st.caption(
        "**Methodology**: ACR (2012) sufficient statistic. Welfare Ŵ = λ̂_jj^(-1/ε). "
        "Domestic share λ_jj = 1 - (imports/GDP). Trade reduction computed via "
        "gravity-based elasticity model. Approximate GDPs used for dollar conversion."
    )
    st.divider()
    st.caption("**Next →** For multi-sector welfare with I-O linkages on the **GE Counterfactual Lab** page. For batch results across all countries and elasticities on the **Research Lab** page.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 9: PERSISTENT HOMOLOGY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Persistent Homology":
    st.markdown(section_header("Persistent Homology of the Global Trade Network",
                "Vietoris-Rips filtration reveals multi-scale topological structure"), unsafe_allow_html=True)
    st.markdown(
        r"""
        The trade network $G=(V,E,w)$ induces a filtered **Vietoris-Rips complex** $\mathcal{K}(\varepsilon)$
        where trade intensity is converted to distance (high trade $\to$ small distance).
        As the filtration parameter $\varepsilon$ increases, the filtration tracks the birth and death of
        topological features via **persistent homology** $H_k(\mathcal{K})$:

        | Homology | Geometric meaning | Economic interpretation |
        |----------|-------------------|------------------------|
        | $H_0$ | Connected components | **Trade blocs** — groups of tightly-linked economies |
        | $H_1$ | 1-cycles (loops) | **Triangular trade** — circular flow patterns (A→B→C→A) |
        | $H_2$ | 2-voids (cavities) | **Structural holes** — missing multilateral linkages |

        Features that persist across a wide filtration range represent **robust structural properties**
        of the trade network (stability theorem, Cohen-Steiner et al. 2007).
        """
    )
    st.info(
        "**Why this matters:** Persistent homology detects structure at every scale simultaneously. "
        "H₀ features are trade blocs (connected components that merge as the filtration threshold rises). "
        "H₁ features are triangular trade cycles (A→B→C→A patterns where no single bilateral link is dominant). "
        "H₂ features are higher-order voids indicating incomplete multilateral integration."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        year_ph = st.select_slider("Year", options=all_years, value=max(all_years), key="ph_year")
    with col2:
        top_n_ph = st.slider("Countries (top N by trade)", 20, 100, 50, 5, key="ph_topn")
    with col3:
        dist_method = st.selectbox("Distance metric", ["negative_log", "inverse", "log_inverse"], key="ph_dist")

    trade_ph = get_year_network_data(year_ph)

    # Serialize for caching
    trade_bytes = trade_ph.to_json().encode()
    trade_hash = hashlib.md5(trade_bytes).hexdigest()

    with st.spinner("Computing persistent homology via Rips filtration..."):
        ph_result = compute_persistent_homology(trade_hash, trade_bytes, max_dim=2, top_n=top_n_ph, distance_method=dist_method)

    # Summary metrics
    pd_data = ph_result["persistence_data"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", ph_result["n_countries"])
    c2.metric("H₀ features (blocs)", pd_data["0"]["n_features"] + pd_data["0"]["n_infinite"])
    c3.metric("H₁ features (cycles)", pd_data["1"]["n_features"] + pd_data["1"].get("n_infinite", 0))
    c4.metric("H₂ features (voids)", pd_data["2"]["n_features"] + pd_data["2"].get("n_infinite", 0))

    # Persistence Diagram
    st.subheader("Persistence Diagram")
    st.markdown("Each point $(b, d)$ represents a topological feature born at filtration $b$ and dying at $d$. Distance from the diagonal = persistence = significance.")

    fig_pd = go.Figure()
    colors = {0: "#1e3a5f", 1: "#dc2626", 2: "#16a34a"}
    names = {0: "H₀ (components)", 1: "H₁ (cycles)", 2: "H₂ (voids)"}

    for dim in range(3):
        dgm = ph_result["diagrams_json"][str(dim)]
        births = dgm["births"]
        deaths = dgm["deaths"]
        if len(births) > 0:
            fig_pd.add_trace(go.Scatter(
                x=births, y=deaths, mode="markers",
                marker=dict(size=8, color=colors[dim], opacity=0.7),
                name=names[dim],
                hovertemplate=f"H_{dim}: birth=%{{x:.3f}}, death=%{{y:.3f}}<extra></extra>",
            ))

    # Diagonal line
    fig_pd.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    fig_pd.update_layout(
        height=500, xaxis_title="Birth", yaxis_title="Death",
        title="Persistence Diagram — Rips Filtration",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1a1a2e"),
    )
    st.plotly_chart(fig_pd, use_container_width=True)

    # Betti Curves
    col_left, col_right = st.columns(2)
    filt_vals = ph_result["filtration_values"]

    with col_left:
        st.subheader("Betti Curves β_k(ε)")
        st.markdown("Number of k-dimensional features alive at filtration value ε.")
        fig_betti = go.Figure()
        for dim in range(3):
            fig_betti.add_trace(go.Scatter(
                x=filt_vals, y=ph_result["betti_curves"][str(dim)],
                mode="lines", name=names[dim],
                line=dict(color=colors[dim], width=2),
            ))
        fig_betti.update_layout(
            height=400, xaxis_title="Filtration ε", yaxis_title="Betti number",
            title="Betti Curves",
        )
        st.plotly_chart(fig_betti, use_container_width=True)

    with col_right:
        st.subheader("Euler Characteristic χ(ε)")
        st.markdown(r"$\chi(\varepsilon) = \beta_0(\varepsilon) - \beta_1(\varepsilon) + \beta_2(\varepsilon)$")
        fig_euler = go.Figure()
        fig_euler.add_trace(go.Scatter(
            x=filt_vals, y=ph_result["euler_curve"],
            mode="lines", line=dict(color="#7c3aed", width=3),
            name="χ(ε)",
        ))
        fig_euler.update_layout(
            height=400, xaxis_title="Filtration ε", yaxis_title="Euler characteristic",
            title="Euler Characteristic Curve",
        )
        st.plotly_chart(fig_euler, use_container_width=True)

    # Persistence barcode
    st.subheader("Persistence Barcode")
    st.markdown("Each bar represents the lifetime [birth, death) of a topological feature. Longer bars = more robust.")
    fig_barcode = go.Figure()
    y_offset = 0
    for dim in range(3):
        dgm = ph_result["diagrams_json"][str(dim)]
        births = dgm["births"]
        deaths = dgm["deaths"]
        pairs = list(zip(births, deaths))
        pairs.sort(key=lambda x: x[1] - x[0], reverse=True)
        for i, (b, d) in enumerate(pairs[:30]):  # top 30 per dimension
            d_plot = min(d, 1.05)
            fig_barcode.add_trace(go.Scatter(
                x=[b, d_plot], y=[y_offset, y_offset],
                mode="lines", line=dict(color=colors[dim], width=3),
                showlegend=(i == 0),
                name=names[dim] if i == 0 else None,
                hovertemplate=f"H_{dim}: [{b:.3f}, {d_plot:.3f})<extra></extra>",
            ))
            y_offset += 1
        y_offset += 2

    fig_barcode.update_layout(
        height=max(400, y_offset * 5), xaxis_title="Filtration",
        yaxis=dict(showticklabels=False), title="Persistence Barcode",
    )
    st.plotly_chart(fig_barcode, use_container_width=True)

    # Clique complex statistics
    st.divider()
    st.subheader("Clique Complex (Flag Complex) Statistics")
    st.markdown(
        r"""
        The **clique complex** $\mathcal{K}(G)$ has a $k$-simplex for every $(k+1)$-clique in $G$.
        The **f-vector** $(f_0, f_1, f_2, \ldots)$ counts simplices by dimension.
        The **Euler characteristic** $\chi = \sum (-1)^k f_k$ measures topological complexity.
        """
    )

    threshold_pct = st.slider("Edge threshold (percentile of distances)", 10, 90, 50, 5, key="clique_thresh")
    with st.spinner("Computing clique complex..."):
        clique_stats = compute_clique_complex_stats(trade_ph, top_n=top_n_ph, threshold_percentile=threshold_pct)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vertices (0-simplices)", clique_stats["n_vertices"])
    c2.metric("Edges (1-simplices)", f"{clique_stats['n_edges']:,}")
    c3.metric("Triangles (2-simplices)", f"{clique_stats['n_triangles']:,}")
    c4.metric("Tetrahedra (3-simplices)", f"{clique_stats['n_tetrahedra']:,}")
    c5.metric("Euler Characteristic χ", f"{clique_stats['euler_characteristic']:,}")

    st.markdown(f"**f-vector**: {clique_stats['f_vector']}")
    st.markdown(f"**Betti numbers**: {clique_stats['betti_numbers']}")

    st.markdown("**Strongest Trilateral Trade Groups** (earliest-forming 2-simplices):")
    tri_data = []
    for tri, filt in clique_stats["top_triangles"][:15]:
        tri_names = [iso_to_name.get(c, c) for c in tri]
        tri_data.append({"Countries": " — ".join(tri_names), "Filtration": round(filt, 4), "ISO": " — ".join(tri)})
    st.dataframe(pd.DataFrame(tri_data), use_container_width=True, hide_index=True)

    # ── Distance Matrix Heatmap ──
    st.divider()
    st.subheader("Trade Distance Matrix")
    st.markdown("The filtered distance matrix that feeds the Rips complex. Dark = close (high trade), bright = far (low trade).")

    D_viz, countries_viz = trade_to_distance_matrix(trade_ph, method=dist_method, top_n_countries=min(top_n_ph, 40))
    max_d_viz = D_viz[D_viz > 0].max() if np.any(D_viz > 0) else 1
    D_viz_norm = D_viz / max_d_viz

    country_labels_viz = [iso_to_name.get(c, c)[:12] for c in countries_viz]
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=D_viz_norm,
        x=country_labels_viz,
        y=country_labels_viz,
        colorscale=[
            [0, "#1e3a5f"],        # Close — deep navy
            [0.2, "#2563eb"],      # Medium-close — blue
            [0.4, "#60a5fa"],      # Moderate — light blue
            [0.6, "#f59e0b"],      # Far — amber
            [0.8, "#dc2626"],      # Very far — red
            [1.0, "#fde68a"],      # Farthest — light yellow
        ],
        colorbar=dict(title="Distance", thickness=12),
        hovertemplate="%{y} → %{x}<br>Distance: %{z:.3f}<extra></extra>",
    ))
    fig_heatmap.update_layout(
        height=600, width=600,
        title="Trade Distance Matrix (Rips filtration input)",
        paper_bgcolor="#f7f6f3",
        plot_bgcolor="#f7f6f3",
        font=dict(color="#1a1a2e", size=8),
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Persistence Landscape ──
    if ph_result["landscapes_h1"]:
        st.divider()
        st.subheader("Persistence Landscape (H₁)")
        st.markdown(
            r"The **persistence landscape** $\lambda_k(t)$ is a functional summary of the persistence diagram. "
            r"Peak height indicates the most persistent 1-cycle feature at each filtration scale."
        )
        landscape_data = np.array(ph_result["landscapes_h1"])
        n_landscapes = landscape_data.shape[0] // 100 if len(landscape_data) > 0 else 0
        if n_landscapes > 0:
            fig_land = go.Figure()
            land_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFEAA7", "#DDA0DD"]
            for k in range(min(n_landscapes, 5)):
                land_slice = landscape_data[k * 100:(k + 1) * 100]
                fig_land.add_trace(go.Scatter(
                    x=np.linspace(0, 1, len(land_slice)),
                    y=land_slice,
                    mode="lines",
                    name=f"λ_{k+1}",
                    line=dict(width=2, color=land_colors[k % len(land_colors)]),
                    fill="tozeroy" if k == 0 else None,
                    fillcolor=f"rgba(255, 107, 107, 0.15)" if k == 0 else None,
                ))
            fig_land.update_layout(
                height=350,
                xaxis_title="Filtration parameter t",
                yaxis_title="λ_k(t)",
                title="H₁ Persistence Landscape — Trade Cycle Prominence",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(color="#1a1a2e"),
            )
            st.plotly_chart(fig_land, use_container_width=True)

    st.divider()
    st.caption(
        "Map each feature back to specific countries and dollar values → **Feature Explorer**. "
        "Test significance against null models → **Statistical Significance**."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TOPOLOGICAL FEATURE EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Feature Explorer":
    st.title("Topological Feature Explorer")
    st.markdown(
        r"""
        **Map every topological feature back to specific countries and dollar values.**

        Each persistence point in $H_k$ corresponds to a concrete economic relationship:
        - **$H_0$ merges**: Which trade blocs merged, triggered by which bilateral link?
        - **$H_1$ cycles**: Which 3 countries form each triangular trade pattern? What's the dollar value?
        - **$H_2$ voids**: Where is multilateral integration incomplete?
        """
    )
    st.info(
        "**Why this matters:** A persistence diagram is only useful if you can say *which countries* form "
        "each feature. This page attributes every H₁ cycle to its constituent countries, with the dollar "
        "value of each edge. The most persistent cycles reveal the most structurally important multilateral "
        "trade relationships."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        year_fe = st.select_slider("Year", options=all_years, value=max(all_years), key="fe_year")
    with col2:
        top_n_fe = st.slider("Countries", 20, 80, 50, 5, key="fe_topn")
    with col3:
        dist_fe = st.selectbox("Distance", ["negative_log", "inverse"], key="fe_dist")

    trade_fe = get_year_network_data(year_fe)
    trade_fe_bytes = trade_fe.to_json().encode()
    trade_fe_hash = hashlib.md5(trade_fe_bytes).hexdigest()

    with st.spinner("Computing attributed persistent homology..."):
        attr_ph = compute_attributed_persistent_homology(
            trade_fe_hash, trade_fe_bytes, top_n=top_n_fe, distance_method=dist_fe,
        )

    # Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", attr_ph["n_countries"])
    c2.metric("H₁ Cycles (triangles)", attr_ph["n_h1_cycles"])
    c3.metric("H₂ Voids", attr_ph["n_h2_voids"])
    c4.metric("H₀ Merge Events", len(attr_ph["h0_merges"]))

    # ── H₁ Trade Cycles ──
    st.divider()
    st.subheader("H₁ Trade Cycles — Triangular Trade Patterns")
    st.markdown("Each H₁ feature corresponds to a **triangular trade relationship** where all 3 bilateral links exist. Sorted by persistence (most robust first).")

    h1 = attr_ph["h1_cycles"]
    if h1:
        h1_table = []
        for i, cyc in enumerate(h1[:25]):
            names = [iso_to_name.get(c, c) for c in cyc["countries"]]
            h1_table.append({
                "#": i + 1,
                "Countries": " — ".join(names),
                "Total Trade ($M)": f"{cyc['total_trade_M']:,.0f}",
                "Persistence": f"{cyc['persistence']:.4f}",
                "Type": cyc["classification"],
                "Regions": ", ".join(cyc["regions"]),
            })
        st.dataframe(pd.DataFrame(h1_table), use_container_width=True, hide_index=True)

        # Feature detail selector
        feature_options = [
            f"#{i+1}: {' — '.join(iso_to_name.get(c, c) for c in cyc['countries'])} "
            f"(persistence={cyc['persistence']:.4f}, ${cyc['total_trade_M']:,.0f}M)"
            for i, cyc in enumerate(h1[:15])
        ]
        selected = st.selectbox("Explore a specific cycle", feature_options, key="fe_select")
        sel_idx = feature_options.index(selected)
        cyc = h1[sel_idx]

        col_detail, col_globe = st.columns([1, 1])
        with col_detail:
            st.markdown(f"### Cycle #{sel_idx + 1} Detail")
            for (c_a, c_b), val in zip(cyc["edges"], cyc["trade_values_M"]):
                name_a = iso_to_name.get(c_a, c_a)
                name_b = iso_to_name.get(c_b, c_b)
                st.markdown(f"- **{name_a} ↔ {name_b}**: ${val:,.0f}M")
            st.markdown(f"- **Total cycle trade**: ${cyc['total_trade_M']:,.0f}M")
            st.markdown(f"- **Persistence**: {cyc['persistence']:.4f} (birth={cyc['birth']:.4f}, death={cyc['death']:.4f})")
            st.markdown(f"- **Classification**: {cyc['classification']}")

        with col_globe:
            # Mini globe showing the cycle
            fig_cycle = go.Figure()
            cycle_countries = cyc["countries"]
            # Draw cycle edges
            for c_a, c_b in cyc["edges"]:
                lat_a, lon_a = get_coords(c_a)
                lat_b, lon_b = get_coords(c_b)
                if (lat_a, lon_a) != (0, 0) and (lat_b, lon_b) != (0, 0):
                    fig_cycle.add_trace(go.Scattergeo(
                        lon=[lon_a, lon_b], lat=[lat_a, lat_b],
                        mode="lines", line=dict(width=3, color="#dc2626"),
                        showlegend=False,
                    ))
            # Draw cycle nodes
            for c in cycle_countries:
                lat, lon = get_coords(c)
                if (lat, lon) != (0, 0):
                    fig_cycle.add_trace(go.Scattergeo(
                        lon=[lon], lat=[lat], mode="markers+text",
                        marker=dict(size=12, color="#dc2626"),
                        text=[c], textposition="top center", textfont=dict(color="#1a1a2e", size=11),
                        showlegend=False,
                    ))
            fig_cycle.update_geos(
                projection_type="orthographic", showland=True,
                landcolor="#e8e5e0", oceancolor="#eef4fb",
                showcountries=True, countrycolor="#b8b4ae",
                showframe=False,
            )
            fig_cycle.update_layout(
                height=350, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#f7f6f3",
                geo=dict(bgcolor="#f7f6f3"),
            )
            st.plotly_chart(fig_cycle, use_container_width=True)

    # ── H₀ Merge Timeline ──
    st.divider()
    st.subheader("H₀ Trade Bloc Formation History")
    st.markdown("How the trade network assembled: each merge event shows two groups of countries becoming connected. Earlier merges = strongest bilateral links.")

    merges = attr_ph["h0_merges"]
    if merges:
        # Show first 20 significant merges (where both components have size > 1, or early merges)
        sig_merges = [m for m in merges if m["component_a_size"] > 1 or m["component_b_size"] > 1][:20]
        if not sig_merges:
            sig_merges = merges[:20]
        merge_table = []
        for m in sig_merges:
            trigger = f"{iso_to_name.get(m['edge'][0], m['edge'][0])} ↔ {iso_to_name.get(m['edge'][1], m['edge'][1])}"
            merge_table.append({
                "Filtration ε": f"{m['filtration']:.4f}",
                "Triggered By": trigger,
                "Trade ($M)": f"{m['trade_value_M']:,.0f}",
                "Group A": ", ".join(m["component_a"][:5]) + ("..." if len(m["component_a"]) > 5 else ""),
                "Group B": ", ".join(m["component_b"][:5]) + ("..." if len(m["component_b"]) > 5 else ""),
            })
        st.dataframe(pd.DataFrame(merge_table), use_container_width=True, hide_index=True)

    # ── H₂ Structural Voids ──
    if attr_ph["h2_voids"]:
        st.divider()
        st.subheader("H₂ Structural Voids — Incomplete Multilateral Integration")
        st.markdown("Four countries where all 6 bilateral links exist but the tetrahedron is hollow — incomplete multilateral integration.")
        for v in attr_ph["h2_voids"][:10]:
            names = [iso_to_name.get(c, c) for c in v["countries"]]
            st.markdown(f"- **{' — '.join(names)}** (persistence={v['persistence']:.4f})")

    st.divider()
    st.caption(
        "Are these features statistically significant? → **Statistical Significance**. "
        "See how they change under tariffs → **Topology-Counterfactual**."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: STATISTICAL SIGNIFICANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Statistical Significance":
    st.title("Statistical Significance of Trade Topology")
    st.markdown(
        r"""
        **Is the observed topology real, or could it arise by chance?**

        This analysis tests each Betti number against three null models:

        | Null Model | Question |
        |---|---|
        | **Erdős-Rényi** | Is $\beta_k$ explained by **edge density** alone? |
        | **Configuration** | Is $\beta_k$ explained by the **degree sequence**? |
        | **Gravity** | Is $\beta_k$ explained by **GDP, distance, institutions**? |

        The **gravity null** is the most economically meaningful: it tests whether the
        observed topology contains structure beyond what the gravity model predicts.
        """
    )
    st.info(
        "**Why this matters:** Observing β₁ = 15 cycles means nothing without a null hypothesis. Is that "
        "more than expected from edge density alone (Erdos-Renyi)? From the degree sequence (configuration "
        "model)? From what gravity predicts? Each null model answers a different question about what drives "
        "the observed topology."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        year_sig = st.select_slider("Year", options=all_years, value=max(all_years), key="sig_year")
    with col2:
        top_n_sig = st.slider("Countries", 20, 60, 40, 5, key="sig_topn")
    with col3:
        n_sims = st.slider("Simulations", 50, 500, 100, 50, key="sig_sims")

    trade_sig = get_year_network_data(year_sig)
    trade_sig_bytes = trade_sig.to_json().encode()
    trade_sig_hash = hashlib.md5(trade_sig_bytes).hexdigest()

    with st.spinner(f"Running {n_sims} null model simulations..."):
        null_result = compute_topological_null_models(
            trade_sig_hash, trade_sig_bytes,
            n_simulations=n_sims, top_n=top_n_sig,
        )

    obs = null_result["observed"]
    grav = null_result["gravity_topology"]

    # Summary table
    st.subheader("Significance Summary")
    sig_rows = []
    for dim in range(3):
        key = f"beta_{dim}"
        row = {
            "Invariant": f"β_{dim}",
            "Observed": obs[key],
            "Gravity Predicted": grav.get(key, "—"),
        }
        for null_type in ["erdos_renyi", "configuration"]:
            vals = null_result["null_distributions"][null_type][key]
            p = null_result["p_values"][null_type][key]
            z = null_result["z_scores"][null_type][key]
            row[f"{null_type} mean±std"] = f"{np.mean(vals):.1f} ± {np.std(vals):.1f}"
            row[f"{null_type} p-value"] = f"{p:.4f}" + (" ***" if p < 0.01 else " **" if p < 0.05 else " *" if p < 0.1 else "")
            row[f"{null_type} z-score"] = f"{z:+.2f}"
        sig_rows.append(row)

    sig_df = pd.DataFrame(sig_rows)
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

    # Null distribution histograms
    st.subheader("Null Distributions vs. Observed")
    betti_names = {0: "β₀ (components)", 1: "β₁ (cycles)", 2: "β₂ (voids)"}
    null_colors = {"erdos_renyi": "rgba(100, 181, 246, 0.6)", "configuration": "rgba(255, 183, 77, 0.6)"}
    null_labels = {"erdos_renyi": "Erdős-Rényi", "configuration": "Configuration"}

    for dim in range(3):
        key = f"beta_{dim}"
        fig_hist = go.Figure()
        for null_type, color in null_colors.items():
            vals = null_result["null_distributions"][null_type][key]
            fig_hist.add_trace(go.Histogram(
                x=vals, name=null_labels[null_type],
                marker_color=color, opacity=0.7,
                nbinsx=20,
            ))
        # Observed line
        fig_hist.add_vline(x=obs[key], line_color="#dc2626", line_width=3,
                           annotation_text=f"Observed: {obs[key]}", annotation_position="top right")
        # Gravity line
        if grav.get(key):
            fig_hist.add_vline(x=grav[key], line_color="#16a34a", line_width=2,
                               line_dash="dash",
                               annotation_text=f"Gravity: {grav[key]}", annotation_position="top left")

        fig_hist.update_layout(
            height=300, title=f"{betti_names[dim]} — Null Distribution",
            xaxis_title=f"β_{dim}", yaxis_title="Count",
            barmode="overlay",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#1a1a2e"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Gravity vs Observed persistence diagrams side by side
    st.divider()
    st.subheader("Observed vs. Gravity-Predicted Persistence Diagrams")
    st.markdown(
        "**Left**: What the trade network actually looks like topologically. "
        "**Right**: What gravity predicts it should look like. "
        "Differences reveal topologically unexpected structure."
    )
    col_obs, col_grav = st.columns(2)

    # Get observed diagrams from the existing PH
    trade_sig_ph = compute_persistent_homology(
        trade_sig_hash, trade_sig_bytes, top_n=top_n_sig,
    )

    colors_ph = {0: "#1e3a5f", 1: "#dc2626", 2: "#16a34a"}
    names_ph = {0: "H₀", 1: "H₁", 2: "H₂"}

    with col_obs:
        fig_obs = go.Figure()
        for dim in range(3):
            dgm = trade_sig_ph["diagrams_json"][str(dim)]
            if dgm["births"]:
                fig_obs.add_trace(go.Scatter(
                    x=dgm["births"], y=dgm["deaths"], mode="markers",
                    marker=dict(size=8, color=colors_ph[dim], opacity=0.7),
                    name=names_ph[dim],
                ))
        fig_obs.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      line=dict(color="gray", dash="dash"), showlegend=False))
        fig_obs.update_layout(
            height=400, title="Observed Trade Network",
            xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.02]),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(color="#1a1a2e"),
        )
        st.plotly_chart(fig_obs, use_container_width=True)

    with col_grav:
        fig_grav = go.Figure()
        grav_dgms = null_result["gravity_diagrams"]
        for dim in range(3):
            dgm = grav_dgms[str(dim)]
            if dgm["births"]:
                fig_grav.add_trace(go.Scatter(
                    x=dgm["births"], y=dgm["deaths"], mode="markers",
                    marker=dict(size=8, color=colors_ph[dim], opacity=0.7),
                    name=names_ph[dim],
                ))
        fig_grav.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                       line=dict(color="gray", dash="dash"), showlegend=False))
        fig_grav.update_layout(
            height=400, title="Gravity-Predicted Network",
            xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.02]),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(color="#1a1a2e"),
        )
        st.plotly_chart(fig_grav, use_container_width=True)

    # Interpretation
    st.divider()
    st.subheader("Interpretation")
    for dim in range(3):
        key = f"beta_{dim}"
        for null_type in ["erdos_renyi", "configuration"]:
            p = null_result["p_values"][null_type][key]
            z = null_result["z_scores"][null_type][key]
            label = null_labels[null_type]
            if p < 0.05:
                st.markdown(f"- **β_{dim} is significant** vs. {label} (p={p:.4f}, z={z:+.2f}). "
                           f"The observed {betti_names[dim].split('(')[1].rstrip(')')} are NOT explained by {'edge density' if null_type == 'erdos_renyi' else 'the degree sequence'} alone.")
            else:
                st.markdown(f"- β_{dim} is **not significant** vs. {label} (p={p:.4f}). "
                           f"The {betti_names[dim].split('(')[1].rstrip(')')} could arise from the null model.")

    obs_b1 = obs["beta_1"]
    grav_b1 = grav.get("beta_1", 0)
    if obs_b1 > grav_b1:
        st.markdown(f"- **Gravity comparison**: Observed β₁={obs_b1} vs gravity-predicted β₁={grav_b1}. "
                   f"The trade network has **{obs_b1 - grav_b1} more cycles** than gravity explains — "
                   f"these represent topologically unexpected trade patterns.")

    st.divider()
    st.caption(
        "Explore the features driving the observed topology → **Feature Explorer**. "
        "See a compressed topological summary → **Mapper Lens**."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MAPPER LENS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Mapper Lens":
    st.title("Mapper Algorithm — Topological Lens on Trade")
    st.markdown(
        r"""
        The **Mapper algorithm** (Singh, Mémoli, Carlsson 2007) constructs a
        compressed topological summary of the trade network. Choose a **lens function**
        to reveal different structural views:

        - **Total trade**: Groups countries by economic size
        - **Trade balance**: Separates surplus vs. deficit nations
        - **Gravity residual**: Groups by unexplained trade intensity
        - **Eigenvector centrality**: Groups by network influence
        - **Geographic remoteness**: Groups by physical distance to partners
        """
    )
    st.info(
        "**Why this matters:** Mapper produces a simplified topological summary of the trade network by "
        "clustering countries along a chosen 'lens' (total trade, centrality, gravity residual). The "
        "resulting graph reveals which countries group together and which bridge separate clusters — at a glance."
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        year_map = st.select_slider("Year", options=all_years, value=max(all_years), key="map_year")
    with col2:
        top_n_map = st.slider("Countries", 20, 80, 50, 5, key="map_topn")
    with col3:
        lens_fn = st.selectbox("Lens function", [
            "total_trade", "trade_balance", "eigenvector_centrality",
            "geographic_remoteness", "gravity_residual",
        ], key="map_lens")
    with col4:
        n_cubes = st.slider("Resolution (cubes)", 5, 20, 10, 1, key="map_cubes")

    trade_map = get_year_network_data(year_map)
    trade_map_bytes = trade_map.to_json().encode()
    trade_map_hash = hashlib.md5(trade_map_bytes).hexdigest()

    with st.spinner("Computing Mapper graph..."):
        mapper_result = compute_trade_mapper(
            trade_map_hash, trade_map_bytes,
            filter_function=lens_fn, top_n=top_n_map,
            n_cubes=n_cubes, perc_overlap=0.3,
        )

    if mapper_result["n_nodes"] == 0:
        st.warning("Mapper produced no nodes. Try adjusting resolution or overlap.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mapper Nodes", mapper_result["n_nodes"])
        c2.metric("Mapper Edges", mapper_result["n_edges"])
        c3.metric("Components", mapper_result["graph_stats"]["n_components"])

        # Build Plotly visualization
        fig_mapper = go.Figure()

        nodes = mapper_result["nodes"]
        edges = mapper_result["edges"]

        # Draw edges
        for e in edges:
            src = nodes.get(e["source"], {})
            tgt = nodes.get(e["target"], {})
            if "x" in src and "x" in tgt:
                fig_mapper.add_trace(go.Scatter(
                    x=[src["x"], tgt["x"]], y=[src["y"], tgt["y"]],
                    mode="lines",
                    line=dict(width=1 + e["n_shared"], color="rgba(100, 181, 246, 0.3)"),
                    showlegend=False, hoverinfo="skip",
                ))

        # Draw nodes
        node_x = [n["x"] for n in nodes.values()]
        node_y = [n["y"] for n in nodes.values()]
        node_sizes = [8 + 3 * n["size"] for n in nodes.values()]
        node_colors = [n["avg_filter"] for n in nodes.values()]
        node_texts = [
            f"<b>Cluster ({n['size']} countries)</b><br>"
            f"Members: {', '.join(n['members'][:6])}"
            f"{'...' if len(n['members']) > 6 else ''}<br>"
            f"Avg {lens_fn}: {n['avg_filter']:.2f}<br>"
            f"Total trade: {fmt_billions(n['total_trade_M']/1000)}"
            for n in nodes.values()
        ]

        fig_mapper.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(
                size=node_sizes, color=node_colors,
                colorscale="Turbo", showscale=True,
                colorbar=dict(title=lens_fn.replace("_", " ").title(), thickness=12),
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            text=node_texts, hoverinfo="text",
            showlegend=False,
        ))

        fig_mapper.update_layout(
            height=600,
            title=f"Mapper Graph — {lens_fn.replace('_', ' ').title()} Lens ({year_map})",
            plot_bgcolor="#f7f6f3",
            paper_bgcolor="#f7f6f3",
            font=dict(color="#1a1a2e"),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        )
        st.plotly_chart(fig_mapper, use_container_width=True)

        # Node explorer
        st.subheader("Cluster Explorer")
        node_options = [
            f"Cluster {i+1}: {', '.join(n['members'][:4])}{'...' if len(n['members']) > 4 else ''} ({n['size']} countries)"
            for i, n in enumerate(nodes.values())
        ]
        if node_options:
            selected_node = st.selectbox("Explore a cluster", node_options, key="map_node")
            sel_idx = node_options.index(selected_node)
            sel_node = list(nodes.values())[sel_idx]

            col_members, col_stats = st.columns([2, 1])
            with col_members:
                st.markdown("**Member Countries:**")
                for c in sel_node["members"]:
                    name = iso_to_name.get(c, c)
                    fv = mapper_result["filter_values"].get(c, 0)
                    st.markdown(f"- {name} ({c}) — {lens_fn}: {fv:.2f}")
            with col_stats:
                st.metric("Cluster Size", sel_node["size"])
                st.metric("Total Trade", fmt_billions(sel_node["total_trade_M"] / 1000))
                st.metric(f"Avg {lens_fn}", f"{sel_node['avg_filter']:.2f}")

        # Structural interpretation
        st.divider()
        st.subheader("Topological Interpretation")
        stats = mapper_result["graph_stats"]
        if stats["n_components"] > 1:
            st.markdown(f"- **{stats['n_components']} disconnected components** — the trade network has distinct clusters under this lens that do not overlap.")
        if stats["has_cycles"]:
            st.markdown("- **Cycles detected** in the Mapper graph — smooth transitions between groups of countries exist (no hard boundaries).")
        else:
            st.markdown("- **Tree-like structure** — the Mapper graph has no cycles, suggesting a hierarchical organization.")

    st.divider()
    st.caption(
        "Understand the underlying persistent homology → **Persistent Homology**. "
        "See how topology responds to tariff policy → **Topology-Counterfactual**."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 10: TOPOLOGICAL EVOLUTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Topological Evolution":
    st.title("Topological Evolution of Global Trade (2002–2022)")
    st.markdown(
        r"""
        Track how the **topological structure** of the trade network changes over 20 years.
        Key events visible in the topology:
        - **2008**: Global financial crisis — trade bloc fragmentation
        - **2020**: COVID-19 — supply chain disruption
        - **2018-2022**: US-China trade tensions — network restructuring

        The **bottleneck distance** $d_B(\mathcal{D}_t, \mathcal{D}_{t+1})$ between persistence diagrams
        quantifies the minimum structural change between years (stability theorem guarantees
        small perturbations yield small distances).
        """
    )
    st.info(
        "**Why this matters:** How does the trade network's topology change over two decades? Tracking "
        "Betti numbers and bottleneck distances year-by-year reveals topological phase transitions — "
        "moments when the network's fundamental structure shifts (e.g., 2008 crisis, COVID-19, US-China decoupling)."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        top_n_evo = st.slider("Countries", 30, 80, 50, 5, key="evo_topn")
    with col2:
        dist_method_evo = st.selectbox("Distance", ["negative_log", "inverse"], key="evo_dist")

    trade_bytes_evo = trade_full.to_json().encode()
    trade_hash_evo = hashlib.md5(trade_bytes_evo).hexdigest()

    with st.spinner("Computing persistent homology for all years (this takes a moment)..."):
        evo_result = compute_topological_evolution(
            trade_hash_evo, trade_bytes_evo, all_years,
            top_n=top_n_evo, distance_method=dist_method_evo,
        )

    evo_df = pd.read_json(evo_result["evolution"])
    bn_df = pd.read_json(evo_result["bottleneck"])

    # Betti numbers over time
    st.subheader("Betti Numbers Over Time")
    fig_betti_time = go.Figure()
    for dim, color, name in [(0, "#1e3a5f", "β₀ (components)"),
                               (1, "#dc2626", "β₁ (cycles)"),
                               (2, "#16a34a", "β₂ (voids)")]:
        if f"beta_{dim}" in evo_df.columns:
            fig_betti_time.add_trace(go.Scatter(
                x=evo_df["year"], y=evo_df[f"beta_{dim}"],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
            ))
    # Event markers
    for yr, label in [(2008, "Financial Crisis"), (2020, "COVID-19"), (2018, "US-China Tariffs")]:
        if yr in evo_df["year"].values:
            fig_betti_time.add_vline(x=yr, line_dash="dot", line_color="gray", opacity=0.5,
                                      annotation_text=label, annotation_position="top")
    fig_betti_time.update_layout(height=400, xaxis_title="Year", yaxis_title="Betti Number")
    st.plotly_chart(fig_betti_time, use_container_width=True)

    # Total persistence over time
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Total Persistence")
        st.markdown("Sum of all feature lifetimes — measures overall topological complexity.")
        fig_tp = go.Figure()
        for dim, color, name in [(0, "#1e3a5f", "H₀"), (1, "#dc2626", "H₁"), (2, "#16a34a", "H₂")]:
            if f"total_persistence_{dim}" in evo_df.columns:
                fig_tp.add_trace(go.Scatter(
                    x=evo_df["year"], y=evo_df[f"total_persistence_{dim}"],
                    mode="lines+markers", name=name,
                    line=dict(color=color, width=2),
                ))
        fig_tp.update_layout(height=350, xaxis_title="Year", yaxis_title="Total Persistence")
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_r:
        st.subheader("Euler Characteristic")
        st.markdown(r"$\chi = \beta_0 - \beta_1 + \beta_2$ — topological complexity invariant.")
        fig_ec = go.Figure()
        fig_ec.add_trace(go.Scatter(
            x=evo_df["year"], y=evo_df["euler_char"],
            mode="lines+markers", line=dict(color="#7c3aed", width=3),
        ))
        fig_ec.update_layout(height=350, xaxis_title="Year", yaxis_title="Euler Characteristic χ")
        st.plotly_chart(fig_ec, use_container_width=True)

    # Bottleneck distances
    st.divider()
    st.subheader("Bottleneck Distance Between Consecutive Years")
    st.markdown(
        r"""
        $d_B(\mathcal{D}_t, \mathcal{D}_{t+1})$ measures the **minimum structural change**
        required to transform one year's persistence diagram into the next.
        Spikes indicate **topological phase transitions** in the trade network.
        """
    )

    fig_bn = go.Figure()
    for dim in range(3):
        dim_data = bn_df[bn_df["dimension"] == dim]
        fig_bn.add_trace(go.Scatter(
            x=dim_data["year_from"].astype(str) + "→" + dim_data["year_to"].astype(str),
            y=dim_data["bottleneck_distance"],
            mode="lines+markers",
            name=f"H_{dim}",
            line=dict(width=2),
        ))

    fig_bn.update_layout(
        height=400, xaxis_title="Year Transition",
        yaxis_title="Bottleneck Distance",
        title="When Did the Trade Network's Topology Change Most?",
    )
    st.plotly_chart(fig_bn, use_container_width=True)

    # Data table
    with st.expander("Raw topological evolution data"):
        st.dataframe(evo_df.round(4), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "See the economic context for these transitions → **Network Evolution**. "
        "Ask whether tariffs would cause similar transitions → **Topology-Counterfactual**."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 11: TOPOLOGICAL SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Topological Sensitivity":
    st.title("Topological Sensitivity to Tariff Shocks")
    st.markdown(
        r"""
        **Question**: At what tariff level does the trade network undergo a **topological phase transition**?

        The platform systematically applies tariff shocks, reconstructs the filtered simplicial complex,
        and tracks how topological invariants change. Key signatures:

        - **$\beta_0$ increasing**: Trade blocs fragmenting (new connected components)
        - **$\beta_1$ appearing**: Circular trade patterns emerging as bilateral links break
        - **Euler characteristic shifting**: Fundamental structural reorganization
        - **Total persistence changing**: Network complexity increasing/decreasing

        This reveals whether tariff impacts are **continuous** (gradual degradation)
        or exhibit **critical thresholds** (topological bifurcations).
        """
    )
    st.info(
        "**Why this matters:** How robust is the network's topology to perturbation? This page applies "
        "controlled shocks and measures how Betti numbers respond, revealing which topological features "
        "are fragile and which are structurally robust."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        imposing_name_ts = st.selectbox(
            "Country imposing tariff",
            sorted(iso_to_name.values()),
            index=sorted(iso_to_name.values()).index("USA") if "USA" in name_to_iso else 0,
            key="ts_imposing",
        )
        imposing_iso_ts = name_to_iso[imposing_name_ts]
    with col2:
        target_name_ts = st.selectbox(
            "Country being tariffed",
            sorted(iso_to_name.values()),
            index=sorted(iso_to_name.values()).index("China") if "China" in name_to_iso else 0,
            key="ts_target",
        )
        target_iso_ts = name_to_iso[target_name_ts]

    col3, col4, col5 = st.columns([1, 1, 1])
    with col3:
        year_ts = st.select_slider("Base year", options=all_years, value=max(all_years), key="ts_year")
    with col4:
        top_n_ts = st.slider("Countries", 20, 80, 40, 5, key="ts_topn")
    with col5:
        max_tariff = st.slider("Max tariff (%)", 50, 300, 150, 10, key="ts_max")

    trade_ts = get_year_network_data(year_ts)
    tariff_range = list(range(0, max_tariff + 1, 10))  # Step by 10 for speed

    # Serialize for caching
    trade_ts_bytes = trade_ts.to_json().encode()
    trade_ts_hash = hashlib.md5(trade_ts_bytes).hexdigest()

    with st.spinner(f"Computing topological invariants for {len(tariff_range)} tariff levels..."):
        sens_df = topological_tariff_sensitivity(
            trade_ts_hash, trade_ts_bytes,
            imposing_iso_ts, target_iso_ts, tariff_range,
            top_n=top_n_ts,
        )

    # Betti numbers vs tariff
    st.subheader(f"Topological Invariants: {imposing_name_ts} tariffs on {target_name_ts}")

    fig_sens_betti = go.Figure()
    for dim, color, name in [(0, "#1e3a5f", "β₀ (components)"),
                               (1, "#dc2626", "β₁ (cycles)"),
                               (2, "#16a34a", "β₂ (voids)")]:
        fig_sens_betti.add_trace(go.Scatter(
            x=sens_df["tariff_pct"], y=sens_df[f"beta_{dim}"],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
        ))
    fig_sens_betti.update_layout(
        height=400, xaxis_title="Tariff (%)", yaxis_title="Betti Number",
        title="Betti Numbers vs. Tariff Level",
    )
    st.plotly_chart(fig_sens_betti, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_tp_sens = go.Figure()
        for dim, color in [(0, "#1e3a5f"), (1, "#dc2626"), (2, "#16a34a")]:
            fig_tp_sens.add_trace(go.Scatter(
                x=sens_df["tariff_pct"], y=sens_df[f"total_persistence_{dim}"],
                mode="lines+markers", name=f"H_{dim}",
                line=dict(color=color, width=2),
            ))
        fig_tp_sens.update_layout(
            height=350, xaxis_title="Tariff (%)", yaxis_title="Total Persistence",
            title="Total Persistence vs. Tariff",
        )
        st.plotly_chart(fig_tp_sens, use_container_width=True)

    with col_r:
        fig_euler_sens = go.Figure()
        fig_euler_sens.add_trace(go.Scatter(
            x=sens_df["tariff_pct"], y=sens_df["euler_char"],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=3),
        ))
        fig_euler_sens.update_layout(
            height=350, xaxis_title="Tariff (%)", yaxis_title="Euler Characteristic χ",
            title="Euler Characteristic vs. Tariff",
        )
        st.plotly_chart(fig_euler_sens, use_container_width=True)

    # Phase transition detection
    st.divider()
    st.subheader("Phase Transition Detection")
    st.markdown(
        "Looking for **discontinuities** in topological invariants — "
        "tariff levels where the network structure changes qualitatively."
    )

    # Compute derivatives
    if len(sens_df) > 2:
        for dim in range(3):
            col_name = f"beta_{dim}"
            sens_df[f"d_beta_{dim}"] = sens_df[col_name].diff()

        # Find biggest jumps
        for dim in range(3):
            max_jump_idx = sens_df[f"d_beta_{dim}"].abs().idxmax()
            if pd.notna(max_jump_idx):
                jump_row = sens_df.loc[max_jump_idx]
                jump_val = jump_row[f"d_beta_{dim}"]
                if abs(jump_val) > 0:
                    st.markdown(
                        f"**H_{dim}**: Largest change at **{jump_row['tariff_pct']:.0f}% tariff** "
                        f"(Δβ_{dim} = {jump_val:+.0f})"
                    )

    with st.expander("Raw sensitivity data"):
        st.dataframe(sens_df.round(4), use_container_width=True, hide_index=True)

    st.caption(
        "**Methodology**: For each tariff level, bilateral trade is reduced via gravity elasticity "
        "(ε=5), the distance matrix is reconstructed, and persistent homology is recomputed "
        "via Vietoris-Rips filtration (ripser). Topological invariants are extracted from the "
        "resulting persistence diagrams."
    )

    st.divider()
    st.caption(
        "Explore specific fragile features → **Feature Explorer**. "
        "See real counterfactual perturbations → **Topology-Counterfactual**."
    )


elif page == "GE Counterfactual Lab":
    st.markdown(section_header(
        "General Equilibrium Counterfactual Lab",
        "Multi-sector GE model — Lashkaripour (2021, JIE) with extensions"
    ), unsafe_allow_html=True)
    st.markdown("""
    Multi-country, multi-sector trade model with **CES preferences**, **input-output linkages**,
    and **8 trade elasticity specifications** from the literature.

    **What this does that ACR cannot:** Full general equilibrium with endogenous wages, tariff revenue,
    multi-sector price indices, and optimal tariff computation for *any* country.
    """)
    st.markdown(key_insight(
        "This general equilibrium model solves for counterfactual wages, prices, and trade flows "
        "simultaneously across all countries and sectors. Unlike reduced-form approaches, it captures "
        "the feedback loops that determine real-world outcomes — tariff revenue redistribution, "
        "input-output price propagation, and endogenous wage adjustment.",
        "General Equilibrium"
    ), unsafe_allow_html=True)

    # ── Dataset & elasticity selection ──
    available = get_available_datasets()
    datasets_available = sorted(set(ds for ds, yr in available))

    col_ds, col_yr, col_el = st.columns(3)
    with col_ds:
        ge_dataset = st.selectbox("Dataset", ["icio", "wiod"],
            format_func=lambda x: {"icio": "OECD ICIO (81 countries, 28 sectors)",
                                    "wiod": "WIOD (44 countries, 16 sectors)"}.get(x, x))
    with col_yr:
        years_for_ds = sorted(yr for ds, yr in available if ds == ge_dataset)
        ge_year = st.selectbox("Year", years_for_ds, index=len(years_for_ds)-1 if years_for_ds else 0)
    with col_el:
        el_options = list(ELASTICITY_REGISTRY.keys())
        ge_elasticity = st.selectbox("Trade Elasticity",
            el_options, index=el_options.index("IS"),
            format_func=lambda x: f"{x} — {ELASTICITY_REGISTRY[x]['name']}")

    # Load data
    try:
        ge_data = load_trade_data(ge_dataset, ge_year)
    except FileNotFoundError:
        st.error(f"Data file not found for {ge_dataset.upper()} {ge_year}. Check mat/ directory.")
        st.stop()

    st.markdown(f"**Loaded:** {ge_data['N']} countries × {ge_data['S']} sectors, "
                f"elasticity source: {ELASTICITY_REGISTRY[ge_elasticity]['paper']}")

    # ── Scenario selection ──
    st.divider()
    scenario_tab, nash_tab, optimal_tab = st.tabs([
        "Tariff Scenarios", "Nash Equilibrium", "Optimal Tariff"
    ])

    # ── Tab 1: Exogenous tariff scenarios ──
    with scenario_tab:
        st.subheader("Counterfactual Tariff Scenarios")

        scen_col1, scen_col2, scen_col3 = st.columns(3)
        with scen_col1:
            scenario_type = st.selectbox("Scenario Type",
                ["uniform", "targeted", "free_trade"],
                format_func=lambda x: {"uniform": "Uniform Tariff",
                                        "targeted": "Targeted (Country/Bloc)",
                                        "free_trade": "Free Trade (Remove All Tariffs)"}.get(x, x))
        with scen_col2:
            imposer = st.selectbox("Who Imposes?", ge_data["countries"],
                index=ge_data["countries"].index("USA") if "USA" in ge_data["countries"] else 0)
        with scen_col3:
            retaliation = st.selectbox("Retaliation", ["none", "reciprocal"],
                format_func=lambda x: {"none": "Unilateral", "reciprocal": "Reciprocal Retaliation"}.get(x, x))

        scenario = {"type": scenario_type, "country": imposer}
        if scenario_type == "uniform":
            rate_pct = st.slider("Tariff Rate (%)", 1, 100, 10)
            scenario["rate"] = rate_pct / 100
        elif scenario_type == "targeted":
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                target = st.selectbox("Target Partner",
                    ["CHN", "EU", "MEX", "CAN", "JPN", "KOR", "IND", "BRA", "GBR", "DEU"] +
                    [c for c in ge_data["countries"] if c not in
                     ["CHN", "EU", "MEX", "CAN", "JPN", "KOR", "IND", "BRA", "GBR", "DEU", imposer]])
            with tcol2:
                rate_pct = st.slider("Tariff Rate (%)", 1, 100, 25)
            scenario["partner"] = target
            scenario["rate"] = rate_pct / 100

        if st.button("Solve Counterfactual", type="primary", key="solve_cf"):
            with st.spinner(f"Solving {ge_data['N']}-country, {ge_data['S']}-sector GE model..."):
                result = solve_counterfactual(ge_data, ge_elasticity, scenario, retaliation)

            if not result["converged"]:
                st.warning(f"Solver did not fully converge (max residual: {result['max_residual']:.2e}). Results may be approximate.")

            welfare = result["welfare"]
            countries = result["countries"]

            # Summary metrics
            imposer_idx = countries.index(imposer) if imposer in countries else None
            worst_idx = np.argmin(welfare)
            best_idx = np.argmax(welfare)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                if imposer_idx is not None:
                    _w = welfare[imposer_idx]
                    m1.markdown(metric_card(f"{imposer} Welfare", f"{_w:+.3f}%",
                                color="green" if _w >= 0 else "red"), unsafe_allow_html=True)
            with m2:
                _wa = float(np.mean(welfare))
                m2.markdown(metric_card("World Average", f"{_wa:+.3f}%",
                            color="blue"), unsafe_allow_html=True)
            with m3:
                m3.markdown(metric_card(f"Worst: {countries[worst_idx]}",
                            f"{welfare[worst_idx]:+.3f}%", color="red"), unsafe_allow_html=True)
            with m4:
                m4.markdown(metric_card(f"Best: {countries[best_idx]}",
                            f"{welfare[best_idx]:+.3f}%", color="green"), unsafe_allow_html=True)

            # Choropleth
            from utils.data_loader import COUNTRY_COORDS
            map_data = []
            for i, c in enumerate(countries):
                coords = COUNTRY_COORDS.get(c)
                if coords and coords != (0, 0):
                    map_data.append({"iso3": c, "welfare_pct": welfare[i],
                                     "lat": coords[0], "lon": coords[1]})
            if map_data:
                map_df = pd.DataFrame(map_data)
                fig_map = go.Figure(go.Choropleth(
                    locations=map_df["iso3"],
                    z=map_df["welfare_pct"],
                    colorscale=WELFARE_COLORSCALE,
                    zmid=0,
                    colorbar_title="Welfare %",
                    text=[f"{r['iso3']}: {r['welfare_pct']:+.3f}%" for _, r in map_df.iterrows()],
                    hoverinfo="text",
                ))
                fig_map.update_layout(
                    geo=dict(showframe=False, showcoastlines=True,
                             projection_type="natural earth",
                             bgcolor="rgba(0,0,0,0)",
                             landcolor="#e8e5e0", oceancolor="#eef4fb",
                             coastlinecolor="#b8b4ae"),
                    height=500, margin=dict(l=0, r=0, t=30, b=0),
                    title=f"Welfare Impact: {imposer} {scenario_type} tariff"
                          + (f" {rate_pct}%" if scenario_type != "free_trade" else "")
                          + f" ({retaliation})",
                )
                st.plotly_chart(fig_map, use_container_width=True)

            # Bar chart of top winners/losers
            order = np.argsort(welfare)
            top_losers = order[:15]
            top_winners = order[-15:][::-1]
            show_idx = np.concatenate([top_losers, top_winners[::-1]])
            bar_df = pd.DataFrame({
                "Country": [countries[i] for i in show_idx],
                "Welfare (%)": [welfare[i] for i in show_idx],
            })
            fig_bar = px.bar(bar_df, x="Country", y="Welfare (%)",
                             color="Welfare (%)",
                             color_continuous_scale=WELFARE_COLORSCALE,
                             color_continuous_midpoint=0)
            fig_bar.update_layout(
                height=400,
                title="Top 15 Winners & Losers",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Full results table
            with st.expander("Full Country Results"):
                results_df = pd.DataFrame({
                    "Country": countries,
                    "Welfare (%)": [f"{w:+.4f}" for w in welfare],
                }).sort_values("Welfare (%)")
                st.dataframe(results_df, use_container_width=True, height=400)

    # ── Tab 2: Nash Equilibrium ──
    with nash_tab:
        st.subheader("Nash Equilibrium Tariff War")
        st.markdown("""
        Solves for the **Nash equilibrium** where all countries simultaneously optimize their
        tariff rates. Each country sets its tariff to maximize national welfare, taking others'
        tariffs as given. This is a **3N system** (wages + incomes + tariffs for N countries).

        *This computation is intensive — may take 30-60 seconds.*
        """)

        if st.button("Solve Nash Equilibrium", type="primary", key="solve_nash"):
            with st.spinner(f"Solving {ge_data['N']}-country Nash equilibrium (3×{ge_data['N']} = {3*ge_data['N']} unknowns)..."):
                nash = solve_nash_equilibrium(ge_data, ge_elasticity, max_retries=5)

            if not nash["converged"]:
                st.warning(f"Solver residual: {nash['max_residual']:.2e}. Results may be approximate.")

            countries = nash["countries"]
            welfare = nash["welfare"]
            tariffs = nash["optimal_tariffs"]

            us_idx = countries.index("USA") if "USA" in countries else 0
            m1, m2, m3 = st.columns(3)
            m1.markdown(metric_card("USA Welfare", f"{welfare[us_idx]:+.3f}%",
                        color="green" if welfare[us_idx] >= 0 else "red"), unsafe_allow_html=True)
            m2.markdown(metric_card("World Average", f"{np.mean(welfare):+.3f}%",
                        color="blue"), unsafe_allow_html=True)
            m3.markdown(metric_card("Avg Nash Tariff", f"{np.mean(tariffs):.1f}%",
                        color="orange"), unsafe_allow_html=True)

            # Tariff distribution
            fig_tariff = go.Figure()
            order_t = np.argsort(tariffs)[::-1]
            fig_tariff.add_trace(go.Bar(
                x=[countries[i] for i in order_t[:30]],
                y=[tariffs[i] for i in order_t[:30]],
                marker_color="#ef4444",
            ))
            fig_tariff.update_layout(
                title="Nash Optimal Tariffs by Country (Top 30)",
                yaxis_title="Optimal Tariff (%)",
                height=400,
            )
            st.plotly_chart(fig_tariff, use_container_width=True)

            # Welfare under Nash
            fig_w = go.Figure()
            order_w = np.argsort(welfare)
            fig_w.add_trace(go.Bar(
                x=[countries[i] for i in order_w],
                y=[welfare[i] for i in order_w],
                marker_color=[("#22c55e" if welfare[i] >= 0 else "#ef4444") for i in order_w],
            ))
            fig_w.update_layout(
                title="Welfare Under Nash Equilibrium (All Countries)",
                yaxis_title="Welfare Change (%)",
                height=400,
            )
            st.plotly_chart(fig_w, use_container_width=True)

            with st.expander("Full Nash Results"):
                nash_df = pd.DataFrame({
                    "Country": countries,
                    "Optimal Tariff (%)": [f"{t:.1f}" for t in tariffs],
                    "Welfare (%)": [f"{w:+.4f}" for w in welfare],
                }).sort_values("Welfare (%)")
                st.dataframe(nash_df, use_container_width=True, height=400)

    # ── Tab 3: Optimal Tariff ──
    with optimal_tab:
        st.subheader("Optimal Unilateral Tariff")
        st.markdown("""
        Computes the **welfare-maximizing tariff** for a single country, holding all other
        countries' tariffs at their factual levels. This is a **2N+1 system**.

        *Ahmad Lashkaripour's code only computes this for the US. Ours works for ANY country.*
        """)

        opt_country = st.selectbox("Country to Optimize",
            ge_data["countries"],
            index=ge_data["countries"].index("USA") if "USA" in ge_data["countries"] else 0,
            key="opt_country")

        if st.button(f"Compute Optimal Tariff for {opt_country}", type="primary", key="solve_opt"):
            with st.spinner(f"Solving optimal tariff for {opt_country}..."):
                opt = solve_optimal_tariff(ge_data, ge_elasticity, country=opt_country)

            if not opt["converged"]:
                st.warning(f"Solver residual: {opt['max_residual']:.2e}")

            countries = opt["countries"]
            welfare = opt["welfare"]
            opt_idx = countries.index(opt_country)

            m1, m2, m3 = st.columns(3)
            m1.markdown(metric_card(f"Optimal {opt_country} Tariff",
                        f"{opt['optimal_tariff_pct']:.1f}%", color="blue"), unsafe_allow_html=True)
            m2.markdown(metric_card(f"{opt_country} Welfare",
                        f"{welfare[opt_idx]:+.3f}%", color="green"), unsafe_allow_html=True)
            m3.markdown(metric_card("World Average",
                        f"{np.mean(welfare):+.3f}%",
                        color="red" if np.mean(welfare) < 0 else "cyan"), unsafe_allow_html=True)

            # Compare: who gets hurt?
            order = np.argsort(welfare)
            bar_df = pd.DataFrame({
                "Country": [countries[i] for i in order[:20]],
                "Welfare (%)": [welfare[i] for i in order[:20]],
            })
            fig_losers = px.bar(bar_df, x="Country", y="Welfare (%)",
                                color="Welfare (%)",
                                color_continuous_scale=["#991b1b", "#ef4444", "#fafafa"],
                                title=f"Countries Most Hurt by {opt_country}'s Optimal Tariff ({opt['optimal_tariff_pct']:.1f}%)")
            fig_losers.update_layout(
                height=400,
            )
            st.plotly_chart(fig_losers, use_container_width=True)

            # Compare multiple countries' optimal tariffs
            st.divider()
            st.markdown("**Compare Optimal Tariffs Across Countries**")
            compare_countries = st.multiselect("Select countries to compare",
                ge_data["countries"],
                default=["USA", "CHN", "DEU", "JPN", "IND"]
                    if all(c in ge_data["countries"] for c in ["USA", "CHN", "DEU", "JPN", "IND"])
                    else ge_data["countries"][:5],
                key="compare_opt")

            if compare_countries and st.button("Compare Optimal Tariffs", key="compare_btn"):
                compare_results = []
                progress = st.progress(0)
                for idx, cc_code in enumerate(compare_countries):
                    with st.spinner(f"Solving for {cc_code}..."):
                        r = solve_optimal_tariff(ge_data, ge_elasticity, country=cc_code)
                        c_idx = r["countries"].index(cc_code)
                        compare_results.append({
                            "Country": cc_code,
                            "Optimal Tariff (%)": r["optimal_tariff_pct"],
                            "Own Welfare (%)": r["welfare"][c_idx],
                            "World Avg Welfare (%)": np.mean(r["welfare"]),
                            "Converged": r["converged"],
                        })
                    progress.progress((idx + 1) / len(compare_countries))

                comp_df = pd.DataFrame(compare_results)
                st.dataframe(comp_df, use_container_width=True)

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name="Optimal Tariff (%)",
                    x=comp_df["Country"],
                    y=comp_df["Optimal Tariff (%)"],
                    marker_color="#ef4444",
                ))
                fig_comp.add_trace(go.Bar(
                    name="Own Welfare (%)",
                    x=comp_df["Country"],
                    y=comp_df["Own Welfare (%)"],
                    marker_color="#22c55e",
                ))
                fig_comp.update_layout(
                    barmode="group",
                    title="Optimal Tariffs & Welfare Gains by Country",
                    height=400,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

    # ── Methodology ──
    with st.expander("Methodology & References"):
        st.markdown("""
        **Model:** Multi-country, multi-sector CES trade model with Cobb-Douglas upper-tier
        preferences across sectors and CES within-sector competition. Hat algebra
        (exact changes) avoids level estimation of productivities or tastes.

        **Equations:**
        - **Eq 6 (Wage balance):** Each country's wage income equals its total export revenue
          net of tariffs, computed via updated bilateral trade shares.
        - **Eq 7 (Income balance):** National income = wage income + tariff revenue.
        - **Eq 14 (Nash FOC):** Optimal tariff = 1 + 1/(weighted avg inverse export supply elasticity).

        **Welfare:** W_hat = E_hat / P_hat where P_hat uses CES aggregation across exporters
        and Cobb-Douglas across sectors. More general than ACR (2012) — handles multi-sector,
        tariff revenue, and endogenous terms-of-trade effects.

        **Data:** OECD ICIO (81 countries, 28 sectors, 2011-2022), WIOD (44 countries, 16 sectors, 2000-2014).
        Tariffs from Teti Global Tariff Database. GDP from World Bank WDI.

        **Elasticities:** 8 specifications from Caliendo-Parro (2015), Simonovska-Waugh (2014),
        Bagwell-Staiger-Yurukoglu (2021), Shapiro (2016), Fontagné et al. (2022),
        Lashkaripour-Lugovskyy (2023), and in-sample estimates.

        **Reference:** Lashkaripour (2021) "The Cost of a Global Tariff War: A Sufficient-Statistics Approach" JIE.
        Python implementation with extensions (any-country optimal tariff, generalized Nash) by Ian Helfrich.
        """)
    st.divider()
    st.caption("**Next →** Run batch analyses across all countries on the **Research Lab** page. See how tariffs reshape network topology on the **Topology-Counterfactual** page.")


elif page == "Research Lab":
    st.markdown(section_header(
        "Research Lab",
        "Systematic analysis across countries, elasticities, and tariff rates"
    ), unsafe_allow_html=True)
    st.info("**Why this matters:** Individual scenarios tell part of the story. This page runs the GE model systematically — sweeping across all countries, all elasticities, and a range of tariff rates — to identify patterns that no single scenario reveals.")

    # ── Dataset & elasticity selection (shared with GE Lab) ──
    available_rl = get_available_datasets()
    col_ds_rl, col_yr_rl = st.columns(2)
    with col_ds_rl:
        rl_dataset = st.selectbox("Dataset", ["icio", "wiod"], key="rl_ds",
            format_func=lambda x: {"icio": "OECD ICIO (81×28)", "wiod": "WIOD (44×16)"}.get(x, x))
    with col_yr_rl:
        rl_years = sorted(yr for ds, yr in available_rl if ds == rl_dataset)
        rl_year = st.selectbox("Year", rl_years, index=len(rl_years)-1 if rl_years else 0, key="rl_yr")

    try:
        rl_data = load_trade_data(rl_dataset, rl_year)
    except FileNotFoundError:
        st.error(f"Data not found for {rl_dataset.upper()} {rl_year}")
        st.stop()

    from utils.research_pipeline import _data_fingerprint
    rl_hash = _data_fingerprint(rl_data)

    rl_tab1, rl_tab2, rl_tab3, rl_tab4 = st.tabs([
        "Optimal Tariff Survey", "Elasticity Sensitivity", "Laffer Curve", "Retaliation Comparison"
    ])

    # ── Tab 1: Optimal Tariff Survey ──
    with rl_tab1:
        st.markdown("Compute each country's **optimal unilateral tariff** under every elasticity specification.")
        if st.button("Run Survey", key="rl_ot_run"):
            prog = st.progress(0, text="Computing optimal tariffs...")
            df_ot = run_optimal_tariff_survey(
                rl_data, _data_hash=rl_hash,
                _progress_callback=lambda c, t: prog.progress(c / t, text=f"Country {c}/{t}")
            )
            prog.empty()
            st.session_state["rl_ot_df"] = df_ot

        if "rl_ot_df" in st.session_state:
            df_ot = st.session_state["rl_ot_df"]
            conv_mask = df_ot["Converged"]
            n_conv = conv_mask.sum()
            st.markdown(f"**{n_conv}** / {len(df_ot)} scenarios converged")

            # Summary stats by elasticity
            summary = df_ot[conv_mask].groupby("Elasticity").agg(
                Mean_Optimal_Tariff=("Optimal_Tariff_Pct", "mean"),
                Median_Optimal_Tariff=("Optimal_Tariff_Pct", "median"),
                Max_Optimal_Tariff=("Optimal_Tariff_Pct", "max"),
                Mean_Own_Welfare=("Own_Welfare_Pct", "mean"),
            ).round(2)
            st.dataframe(summary, use_container_width=True)

            # Box plot of optimal tariffs by elasticity
            fig_ot = px.box(df_ot[conv_mask], x="Elasticity", y="Optimal_Tariff_Pct",
                            color="Elasticity", color_discrete_sequence=CATEGORY_COLORS,
                            title="Optimal Unilateral Tariff Distribution by Elasticity")
            apply_theme(fig_ot)
            st.plotly_chart(fig_ot, use_container_width=True)

            # Scatter: own welfare vs world welfare
            fig_scat = px.scatter(df_ot[conv_mask], x="Own_Welfare_Pct", y="World_Avg_Welfare_Pct",
                                  color="Elasticity", hover_data=["Country"],
                                  color_discrete_sequence=CATEGORY_COLORS,
                                  title="Optimal Tariff: Own vs World Welfare")
            fig_scat.add_hline(y=0, line_dash="dot", line_color="#ddd9d3")
            fig_scat.add_vline(x=0, line_dash="dot", line_color="#ddd9d3")
            apply_theme(fig_scat)
            st.plotly_chart(fig_scat, use_container_width=True)

    # ── Tab 2: Elasticity Sensitivity ──
    with rl_tab2:
        st.markdown("Run the **same tariff scenario** under every elasticity specification.")
        es_col1, es_col2, es_col3 = st.columns(3)
        with es_col1:
            es_country = st.selectbox("Imposing Country", rl_data["countries"], key="rl_es_c",
                                      index=rl_data["countries"].index("USA") if "USA" in rl_data["countries"] else 0)
        with es_col2:
            es_rate = st.slider("Tariff Rate (%)", 0, 100, 10, key="rl_es_r") / 100
        with es_col3:
            es_retal = st.selectbox("Retaliation", ["none", "reciprocal"], key="rl_es_ret")

        if st.button("Run Sensitivity", key="rl_es_run"):
            scenario = {"type": "uniform", "country": es_country, "rate": es_rate}
            prog = st.progress(0, text="Comparing elasticities...")
            df_es = run_elasticity_sensitivity(
                rl_data, scenario, retaliation=es_retal, _data_hash=rl_hash,
                _progress_callback=lambda c, t: prog.progress(c / t, text=f"Elasticity {c}/{t}")
            )
            prog.empty()
            st.session_state["rl_es_df"] = df_es

        if "rl_es_df" in st.session_state:
            df_es = st.session_state["rl_es_df"]
            # Bar chart: world avg welfare by elasticity
            world_avg = df_es.groupby("Elasticity_Name")["Welfare_Pct"].mean().reset_index()
            fig_es = px.bar(world_avg, x="Elasticity_Name", y="Welfare_Pct",
                           color="Welfare_Pct", color_continuous_scale=WELFARE_COLORSCALE,
                           color_continuous_midpoint=0,
                           title=f"World Avg Welfare: {es_country} {int(es_rate*100)}% Tariff by Elasticity")
            apply_theme(fig_es)
            st.plotly_chart(fig_es, use_container_width=True)

            # Heatmap: country × elasticity welfare
            pivot = df_es.pivot_table(index="Country", columns="Elasticity_Name",
                                      values="Welfare_Pct").round(3)
            top_affected = pivot.mean(axis=1).abs().nlargest(20).index
            fig_hm = px.imshow(pivot.loc[top_affected], color_continuous_scale=WELFARE_COLORSCALE,
                               color_continuous_midpoint=0, aspect="auto",
                               title="Welfare Impact (%) — Top 20 Most Affected Countries")
            apply_theme(fig_hm)
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Tab 3: Laffer Curve ──
    with rl_tab3:
        st.markdown("Sweep tariff rates to find the welfare-maximizing tariff (**Laffer curve**).")
        laf_col1, laf_col2 = st.columns(2)
        with laf_col1:
            laf_country = st.selectbox("Country", rl_data["countries"], key="rl_laf_c",
                                       index=rl_data["countries"].index("USA") if "USA" in rl_data["countries"] else 0)
        with laf_col2:
            laf_elas = st.selectbox("Elasticity", list(ELASTICITY_REGISTRY.keys()), key="rl_laf_e",
                                    format_func=lambda x: f"{x} — {ELASTICITY_REGISTRY[x]['name']}")

        if st.button("Run Sweep", key="rl_laf_run"):
            prog = st.progress(0, text="Sweeping tariff rates...")
            df_laf = run_tariff_rate_sweep(
                rl_data, laf_elas, laf_country, _data_hash=rl_hash,
                _progress_callback=lambda c, t: prog.progress(c / t, text=f"Rate {c}/{t}")
            )
            prog.empty()
            st.session_state["rl_laf_df"] = df_laf

        if "rl_laf_df" in st.session_state:
            df_laf = st.session_state["rl_laf_df"]
            conv_laf = df_laf[df_laf["Converged"]]
            if len(conv_laf) > 0:
                peak = conv_laf.loc[conv_laf["Imposer_Welfare_Pct"].idxmax()]
                c1, c2, c3 = st.columns(3)
                c1.markdown(metric_card("Peak Welfare", f"{peak['Imposer_Welfare_Pct']:.3f}%", color="green"), unsafe_allow_html=True)
                c2.markdown(metric_card("Optimal Rate", f"{peak['Tariff_Rate_Pct']:.0f}%", color="navy"), unsafe_allow_html=True)
                c3.markdown(metric_card("World Avg at Peak", f"{peak['World_Avg_Welfare_Pct']:.3f}%",
                            f"{'+' if peak['World_Avg_Welfare_Pct'] >= 0 else ''}{peak['World_Avg_Welfare_Pct']:.3f}%",
                            "green" if peak["World_Avg_Welfare_Pct"] >= 0 else "red"), unsafe_allow_html=True)

            fig_laf = go.Figure()
            fig_laf.add_trace(go.Scatter(
                x=conv_laf["Tariff_Rate_Pct"], y=conv_laf["Imposer_Welfare_Pct"],
                mode="lines+markers", name=f"{laf_country} Welfare",
                line=dict(color="#b45309", width=2.5), marker=dict(size=6),
            ))
            fig_laf.add_trace(go.Scatter(
                x=conv_laf["Tariff_Rate_Pct"], y=conv_laf["World_Avg_Welfare_Pct"],
                mode="lines+markers", name="World Avg",
                line=dict(color="#ef4444", width=2, dash="dash"), marker=dict(size=5),
            ))
            fig_laf.add_hline(y=0, line_dash="dot", line_color="#ddd9d3")
            fig_laf.update_layout(
                title=f"Tariff Laffer Curve — {laf_country}",
                xaxis_title="Tariff Rate (%)", yaxis_title="Welfare Change (%)",
            )
            apply_theme(fig_laf)
            st.plotly_chart(fig_laf, use_container_width=True)

    # ── Tab 4: Retaliation Comparison ──
    with rl_tab4:
        st.markdown("Compare welfare **with and without retaliation** for the same scenario.")
        rt_col1, rt_col2, rt_col3 = st.columns(3)
        with rt_col1:
            rt_country = st.selectbox("Imposing Country", rl_data["countries"], key="rl_rt_c",
                                      index=rl_data["countries"].index("USA") if "USA" in rl_data["countries"] else 0)
        with rt_col2:
            rt_rate = st.slider("Tariff Rate (%)", 0, 100, 10, key="rl_rt_r") / 100
        with rt_col3:
            rt_elas = st.selectbox("Elasticity", list(ELASTICITY_REGISTRY.keys()), key="rl_rt_e",
                                   format_func=lambda x: f"{x} — {ELASTICITY_REGISTRY[x]['name']}")

        if st.button("Compare", key="rl_rt_run"):
            scenario = {"type": "uniform", "country": rt_country, "rate": rt_rate}
            prog = st.progress(0, text="Solving with and without retaliation...")
            df_rt = run_retaliation_comparison(
                rl_data, rt_elas, scenario, _data_hash=rl_hash,
                _progress_callback=lambda c, t: prog.progress(c / t, text=f"Regime {c}/2")
            )
            prog.empty()
            st.session_state["rl_rt_df"] = df_rt

        if "rl_rt_df" in st.session_state:
            df_rt = st.session_state["rl_rt_df"]

            # Scatter: no retaliation vs reciprocal
            fig_rt = px.scatter(df_rt, x="Welfare_NoRetaliation_Pct", y="Welfare_Reciprocal_Pct",
                                text="Country", color="Welfare_Diff_Pct",
                                color_continuous_scale=WELFARE_COLORSCALE, color_continuous_midpoint=0,
                                title="Welfare: No Retaliation vs Reciprocal")
            fig_rt.add_trace(go.Scatter(
                x=[-5, 5], y=[-5, 5], mode="lines", line=dict(dash="dot", color="#ddd9d3"),
                showlegend=False,
            ))
            fig_rt.update_traces(textposition="top center", textfont_size=9, selector=dict(mode="markers+text"))
            apply_theme(fig_rt)
            st.plotly_chart(fig_rt, use_container_width=True)

            # Bar of top winners/losers from retaliation
            df_rt_sorted = df_rt.sort_values("Welfare_Diff_Pct")
            top_losers = df_rt_sorted.head(10)
            top_winners = df_rt_sorted.tail(10)
            df_extremes = pd.concat([top_losers, top_winners])
            fig_diff = px.bar(df_extremes, x="Country", y="Welfare_Diff_Pct",
                             color="Welfare_Diff_Pct", color_continuous_scale=WELFARE_COLORSCALE,
                             color_continuous_midpoint=0,
                             title="Who Gains/Loses Most from Retaliation?")
            apply_theme(fig_diff)
            st.plotly_chart(fig_diff, use_container_width=True)

    with st.expander("Methodology"):
        st.markdown("""
        All results use the **sufficient-statistics GE trade model** (Lashkaripour 2021, JIE).
        Each scenario solves a non-linear system of 2N or 3N equations using `scipy.optimize.root`.
        Results cached for 2 hours via `@st.cache_data`.

        **Elasticity sensitivity** is critical: the 8 specifications span a wide range (mean σ from 2.5 to 12),
        producing qualitatively different welfare predictions. This is a key source of model uncertainty
        that the literature often ignores by fixing a single estimate.
        """)
    st.divider()
    st.caption("**Next →** Explore the topological implications of these tariff scenarios on the **Topology-Counterfactual** page.")


elif page == "Topology-Counterfactual":
    st.markdown(section_header(
        "Topological Policy Analysis",
        "How does the topology of the global trade network change under alternative tariff regimes?"
    ), unsafe_allow_html=True)

    st.markdown("""
    When a country imposes tariffs, trade flows do not simply shrink — they reorganize. This page
    applies persistent homology to the counterfactual trade networks predicted by the general
    equilibrium model, measuring how Betti numbers (β₀ = connected components, β₁ = independent cycles)
    and total H₁ persistence respond to policy changes.
    """)
    st.info(
        "**Why this matters:** Standard applications of topological data analysis to trade use observed data, "
        "so their results largely reflect GDP and geographic proximity. By applying persistent homology to "
        "GE-predicted counterfactual networks, this analysis isolates how *tariff policy itself* reshapes the "
        "trade network's topological structure — revealing structural phase transitions invisible to conventional methods."
    )

    available_tc = get_available_datasets()
    tc_col1, tc_col2, tc_col3 = st.columns(3)
    with tc_col1:
        tc_dataset = st.selectbox("Dataset", ["icio", "wiod"], key="tc_ds",
            format_func=lambda x: {"icio": "OECD ICIO", "wiod": "WIOD"}.get(x, x))
    with tc_col2:
        tc_years = sorted(yr for ds, yr in available_tc if ds == tc_dataset)
        tc_year = st.selectbox("Year", tc_years, index=len(tc_years)-1 if tc_years else 0, key="tc_yr")
    with tc_col3:
        tc_elas = st.selectbox("Elasticity", list(ELASTICITY_REGISTRY.keys()), key="tc_el",
                               format_func=lambda x: f"{x} — {ELASTICITY_REGISTRY[x]['name']}")

    tc_tab1, tc_tab2 = st.tabs(["Factual vs Counterfactual", "Topological Laffer Curve"])

    # ── Tab 1: Compare factual vs counterfactual topology ──
    with tc_tab1:
        tc_s1, tc_s2, tc_s3, tc_s4 = st.columns(4)
        with tc_s1:
            tc_country = st.selectbox("Imposing Country", ["USA", "CHN", "DEU", "JPN", "GBR", "IND"], key="tc_c")
        with tc_s2:
            tc_rate = st.slider("Tariff Rate (%)", 5, 100, 20, 5, key="tc_r") / 100
        with tc_s3:
            tc_retal = st.selectbox("Retaliation", ["none", "reciprocal"], key="tc_ret")
        with tc_s4:
            tc_topn = st.slider("Top-N countries", 20, 80, 50, 5, key="tc_n")

        if st.button("Compare Topology", key="tc_run"):
            try:
                tc_data = load_trade_data(tc_dataset, tc_year)
                tc_hash = hashlib.sha256(f"{tc_dataset}_{tc_year}".encode()).hexdigest()[:16]
                scenario = {"type": "uniform", "country": tc_country, "rate": tc_rate}
                with st.spinner("Solving GE counterfactual & computing persistent homology..."):
                    topo_result = compare_topology_factual_vs_counterfactual(
                        tc_hash, tc_dataset, tc_year, tc_elas, scenario,
                        retaliation=tc_retal, top_n=tc_topn,
                    )
                st.session_state["tc_result"] = topo_result
            except Exception as e:
                st.error(f"Error: {e}")

        if "tc_result" in st.session_state:
            tr = st.session_state["tc_result"]
            pc = tr["persistence_change"]

            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(metric_card("Δβ₀", f"{pc['delta_b0']:+d}",
                        f"{'Fewer' if pc['delta_b0'] < 0 else 'More'} components",
                        "green" if pc["delta_b0"] < 0 else "red"), unsafe_allow_html=True)
            m2.markdown(metric_card("Δβ₁", f"{pc['delta_b1']:+d}",
                        f"{'Fewer' if pc['delta_b1'] < 0 else 'More'} cycles",
                        "cyan"), unsafe_allow_html=True)
            m3.markdown(metric_card("ΔH₁ Persistence", f"{pc['pct_change_persistence']:+.1f}%",
                        color="purple"), unsafe_allow_html=True)
            m4.markdown(metric_card("Countries", str(tr["n_countries"]),
                        color="blue"), unsafe_allow_html=True)

            st.divider()

            # Side-by-side persistence diagrams
            diag_col1, diag_col2 = st.columns(2)

            for col, label, dgms_key in [(diag_col1, "Factual", "factual_diagrams"),
                                          (diag_col2, "Counterfactual", "cf_diagrams")]:
                with col:
                    st.markdown(f"**{label} Persistence Diagram**")
                    dgms = tr[dgms_key]
                    fig_pd = go.Figure()
                    colors_dim = {"0": "#1e3a5f", "1": "#dc2626"}
                    for dim_str, points in dgms.items():
                        births = [p["birth"] for p in points if p["death"] is not None]
                        deaths = [p["death"] for p in points if p["death"] is not None]
                        if births:
                            fig_pd.add_trace(go.Scatter(
                                x=births, y=deaths, mode="markers",
                                name=f"H{dim_str}", marker=dict(
                                    color=colors_dim.get(dim_str, "#1e3a5f"), size=6, opacity=0.7,
                                ),
                            ))
                    # Diagonal
                    all_vals = []
                    for dim_str, points in dgms.items():
                        for p in points:
                            all_vals.append(p["birth"])
                            if p["death"] is not None:
                                all_vals.append(p["death"])
                    if all_vals:
                        mn, mx = min(all_vals), max(all_vals)
                        fig_pd.add_trace(go.Scatter(
                            x=[mn, mx], y=[mn, mx], mode="lines",
                            line=dict(dash="dot", color="#ddd9d3"), showlegend=False,
                        ))
                    fig_pd.update_layout(
                        xaxis_title="Birth", yaxis_title="Death",
                        height=380,
                    )
                    apply_theme(fig_pd)
                    st.plotly_chart(fig_pd, use_container_width=True)

            # Interpretation
            st.markdown(f"""
            **Interpretation:** A {int(tc_rate*100)}% tariff by {tc_country}
            {"increases" if pc["pct_change_persistence"] > 0 else "decreases"} total H₁ persistence
            by **{abs(pc['pct_change_persistence']):.1f}%**, meaning the trade network's cycle structure
            becomes {"more complex" if pc["pct_change_persistence"] > 0 else "simpler"} under this tariff regime.

            β₀ change of {pc['delta_b0']:+d} indicates the network {"fragments" if pc["delta_b0"] > 0 else "consolidates"}.
            β₁ change of {pc['delta_b1']:+d} indicates {"new trade triangles form" if pc["delta_b1"] > 0 else "existing trade cycles collapse"}.
            """)

    # ── Tab 2: Topological Laffer Curve ──
    with tc_tab2:
        st.markdown("""
        How do Betti numbers and total persistence respond to tariff rate? The *topological Laffer curve*
        traces network complexity as a function of tariff policy — revealing rates at which the trade
        network's structure undergoes qualitative change.
        """)

        tl_col1, tl_col2 = st.columns(2)
        with tl_col1:
            tl_country = st.selectbox("Country", ["USA", "CHN", "DEU", "JPN", "GBR"], key="tl_c")
        with tl_col2:
            tl_topn = st.slider("Top-N countries", 20, 60, 40, 5, key="tl_n")

        if st.button("Compute Topological Laffer Curve", key="tl_run"):
            try:
                tl_data = load_trade_data(tc_dataset, tc_year)
                tl_hash = hashlib.sha256(f"{tc_dataset}_{tc_year}".encode()).hexdigest()[:16]
                with st.spinner("Computing PH at each tariff rate (may take 30-60s)..."):
                    tl_result = topological_laffer_curve(
                        tl_hash, tc_dataset, tc_year, tc_elas,
                        country=tl_country, top_n=tl_topn,
                    )
                st.session_state["tl_result"] = tl_result
                st.session_state["tl_country"] = tl_country
            except Exception as e:
                st.error(f"Error: {e}")

        if "tl_result" in st.session_state:
            tlr = st.session_state["tl_result"]
            tl_c = st.session_state.get("tl_country", "USA")
            rates_pct = [r * 100 for r in tlr["rates"]]

            # Dual-axis plot: β₁ + total persistence
            from plotly.subplots import make_subplots
            fig_tl = make_subplots(specs=[[{"secondary_y": True}]])
            fig_tl.add_trace(go.Scatter(
                x=rates_pct, y=tlr["b1"], mode="lines+markers", name="β₁ (cycles)",
                line=dict(color="#b45309", width=2.5), marker=dict(size=7),
            ), secondary_y=False)
            fig_tl.add_trace(go.Scatter(
                x=rates_pct, y=tlr["total_persistence_h1"], mode="lines+markers",
                name="Total H₁ Persistence",
                line=dict(color="#a855f7", width=2.5), marker=dict(size=7),
            ), secondary_y=True)
            fig_tl.update_layout(title=f"Topological Laffer Curve — {tl_c}")
            fig_tl.update_xaxes(title_text="Tariff Rate (%)")
            fig_tl.update_yaxes(title_text="β₁ (Betti-1)", secondary_y=False)
            fig_tl.update_yaxes(title_text="Total H₁ Persistence", secondary_y=True)
            apply_theme(fig_tl)
            st.plotly_chart(fig_tl, use_container_width=True)

            # Welfare overlay
            fig_tw = go.Figure()
            fig_tw.add_trace(go.Scatter(
                x=rates_pct, y=tlr["welfare_imposer"], mode="lines+markers",
                name=f"{tl_c} Welfare", line=dict(color="#22c55e", width=2.5),
            ))
            fig_tw.add_trace(go.Scatter(
                x=rates_pct, y=tlr["welfare_world"], mode="lines+markers",
                name="World Avg Welfare", line=dict(color="#ef4444", width=2, dash="dash"),
            ))
            fig_tw.add_hline(y=0, line_dash="dot", line_color="#ddd9d3")
            fig_tw.update_layout(
                title=f"Welfare at Each Tariff Rate — {tl_c}",
                xaxis_title="Tariff Rate (%)", yaxis_title="Welfare Change (%)",
            )
            apply_theme(fig_tw)
            st.plotly_chart(fig_tw, use_container_width=True)

            # Combined insight
            if len(tlr["rates"]) > 2:
                b1_trend = "increases" if tlr["b1"][-1] > tlr["b1"][0] else "decreases"
                st.info(f"""
                **Key finding:** As {tl_c}'s tariff rate rises from 0% to {rates_pct[-1]:.0f}%,
                β₁ {b1_trend} from {tlr['b1'][0]} to {tlr['b1'][-1]},
                indicating that the trade network's cycle structure {"becomes more complex" if b1_trend == "increases" else "simplifies"}.
                This traces how persistent homology responds to counterfactual tariff policy —
                identifying the structural tipping points where the network reorganizes.
                """)

    with st.expander("Methodology"):
        st.markdown("""
        **Pipeline:** GE counterfactual (Lashkaripour 2021) → reconstruct bilateral flows →
        negative-log distance matrix → Vietoris-Rips filtration → persistent homology (ripser) → Betti numbers.

        **Approach:** Standard TDA-trade papers (Topaz et al., 2015; Feng et al., 2019) compute persistent
        homology on *observed* trade data. This analysis instead computes PH on *counterfactual* networks —
        trade flows that the GE model predicts would exist under alternative tariff regimes but that have
        never actually occurred. This makes it possible to ask: "Does protectionism create or destroy trade cycles?"

        **Topological Laffer Curve:** Just as the conventional Laffer curve relates welfare to tariff rate,
        the topological Laffer curve relates β₁ and total H₁ persistence to tariff rate — tracing how
        network complexity responds to trade policy and identifying structural tipping points.
        """)

    st.divider()
    st.caption(
        "Understand the base topology before counterfactuals → **Persistent Homology**. "
        "Check statistical significance → **Statistical Significance**."
    )


# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("""
<div style="font-size: 0.72rem; color: #9b9ba7; line-height: 1.6;">
    <strong style="color: #6b6b7b;">Data:</strong> BACI, OECD ICIO, WIOD, Gravity, Tau<br>
    <strong style="color: #6b6b7b;">Methods:</strong> PPML, ACR, GE Counterfactual, Nash, PH, Mapper<br>
    <strong style="color: #6b6b7b;">Refs:</strong> Lashkaripour (2021), Carlsson (2009), Santos Silva & Tenreyro (2006)<br>
    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(221, 217, 211, 0.6);">
        <strong style="color: #1e3a5f;">Dr. Ian Helfrich</strong> · Georgia Institute of Technology
    </div>
</div>
""", unsafe_allow_html=True)
