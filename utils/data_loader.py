"""
Data loading and caching for Trade Network Visualizer.
Loads BACI bilateral trade, tariff, gravity, and country code data.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Data resolution: prefer local warehouse, fall back to bundled data/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WAREHOUSE = Path.home() / "trade_data_warehouse"
_BUNDLED = _PROJECT_ROOT / "data"

BACI_DIR = _WAREHOUSE / "baci" if (_WAREHOUSE / "baci").exists() else _BUNDLED / "baci"
TAU_DIR = _WAREHOUSE / "tau" if (_WAREHOUSE / "tau").exists() else _BUNDLED / "tau"
GRAVITY_DIR = _WAREHOUSE / "gravity" if (_WAREHOUSE / "gravity").exists() else _BUNDLED / "gravity"
# Keep WAREHOUSE reference for backward compat
WAREHOUSE = _WAREHOUSE

# ISO3 -> lat/lon for plotting (major trading nations + all others via fallback)
COUNTRY_COORDS = {
    "USA": (39.8, -98.5), "CHN": (35.0, 105.0), "DEU": (51.2, 10.4),
    "JPN": (36.2, 138.3), "GBR": (55.4, -3.4), "FRA": (46.6, 2.2),
    "KOR": (36.5, 128.0), "NLD": (52.1, 5.3), "ITA": (41.9, 12.5),
    "CAN": (56.1, -106.3), "MEX": (23.6, -102.6), "IND": (20.6, 78.9),
    "BRA": (-14.2, -51.9), "AUS": (-25.3, 133.8), "RUS": (61.5, 105.3),
    "ESP": (40.5, -3.7), "SGP": (1.35, 103.8), "CHE": (46.8, 8.2),
    "BEL": (50.5, 4.5), "TWN": (23.7, 121.0), "THA": (15.9, 100.5),
    "VNM": (14.1, 108.3), "MYS": (4.2, 101.9), "IDN": (-0.8, 113.9),
    "SAU": (23.9, 45.1), "ARE": (23.4, 53.8), "TUR": (38.9, 35.2),
    "POL": (51.9, 19.1), "SWE": (60.1, 18.6), "AUT": (47.5, 14.6),
    "NOR": (60.5, 8.5), "IRL": (53.1, -8.0), "DNK": (56.3, 9.5),
    "ZAF": (-30.6, 22.9), "ISR": (31.0, 34.9), "CZE": (49.8, 15.5),
    "CHL": (-35.7, -71.5), "ARG": (-38.4, -63.6), "COL": (4.6, -74.1),
    "PHL": (12.9, 121.8), "NGA": (9.1, 8.7), "EGY": (26.8, 30.8),
    "BGD": (23.7, 90.4), "PAK": (30.4, 69.3), "PER": (-9.2, -75.0),
    "KAZ": (48.0, 68.0), "QAT": (25.4, 51.2), "HUN": (47.2, 19.5),
    "ROU": (45.9, 25.0), "PRT": (39.4, -8.2), "NZL": (-40.9, 174.9),
    "FIN": (61.9, 25.7), "GRC": (39.1, 21.8), "UKR": (48.4, 31.2),
    "KWT": (29.3, 47.5), "MAR": (31.8, -7.1), "SVK": (48.7, 19.7),
    "ECU": (-1.8, -78.2), "LKA": (7.9, 80.8), "MMR": (21.9, 96.0),
    "KEN": (-0.0, 37.9), "ETH": (9.1, 40.5), "GHA": (7.9, -1.0),
    "TZA": (-6.4, 34.9), "AGO": (-11.2, 17.9), "IRQ": (33.2, 43.7),
    "LBY": (26.3, 17.2), "OMN": (21.5, 55.9), "JOR": (30.6, 36.2),
    "LBN": (33.9, 35.9), "TUN": (34.0, 9.5), "URY": (-32.5, -55.8),
    "PRY": (-23.4, -58.4), "BOL": (-16.3, -63.6), "VEN": (6.4, -66.6),
    "DOM": (18.7, -70.2), "GTM": (15.8, -90.2), "CRI": (10.0, -84.2),
    "PAN": (9.0, -79.5), "HND": (15.2, -86.2), "SLV": (13.8, -88.9),
    "NIC": (12.9, -85.2), "CUB": (21.5, -78.0), "JAM": (18.1, -77.3),
    "BHR": (26.0, 50.5), "CYP": (35.1, 33.4), "LUX": (49.8, 6.1),
    "SVN": (46.2, 15.0), "HRV": (45.1, 15.2), "BGR": (42.7, 25.5),
    "LTU": (55.2, 23.9), "LVA": (56.9, 24.1), "EST": (58.6, 25.0),
    "ISL": (65.0, -18.0), "MLT": (35.9, 14.4), "SRB": (44.0, 21.0),
    "BIH": (43.9, 17.7), "ALB": (41.2, 20.2), "MKD": (41.5, 21.7),
    "MNE": (42.7, 19.4), "GEO": (42.3, 43.4), "ARM": (40.1, 45.0),
    "AZE": (40.1, 47.6), "MDA": (47.4, 28.4), "BLR": (53.7, 27.9),
    "KHM": (12.6, 105.0), "LAO": (19.9, 102.5), "MNG": (46.9, 103.8),
    "BRN": (4.5, 114.7), "NPL": (28.4, 84.1), "AFG": (33.9, 67.7),
    "CMR": (7.4, 12.4), "CIV": (7.5, -5.5), "SEN": (14.5, -14.5),
    "UGA": (1.4, 32.3), "MOZ": (18.7, 35.5), "ZMB": (-13.1, 27.8),
    "ZWE": (-19.0, 29.2), "COD": (-4.0, 21.8), "COG": (-0.2, 15.8),
    "GAB": (-0.8, 11.6), "GNQ": (1.6, 10.3), "TCD": (15.5, 18.7),
    "MLI": (17.6, -4.0), "BFA": (12.2, -1.6), "NER": (17.6, 8.1),
    "BEN": (9.3, 2.3), "TGO": (8.6, 1.2), "SLE": (8.5, -11.8),
    "LBR": (6.4, -9.4), "GIN": (9.9, -9.7), "MWI": (-13.3, 34.3),
    "RWA": (-1.9, 29.9), "BDI": (-3.4, 29.9), "SOM": (5.2, 46.2),
    "ERI": (15.2, 39.8), "DJI": (11.6, 43.1), "MRT": (21.0, -10.9),
    "SSD": (6.9, 31.3), "SDN": (12.9, 30.2), "NAM": (-22.6, 17.1),
    "BWA": (-22.3, 24.7), "SWZ": (-26.5, 31.5), "LSO": (-29.6, 28.2),
    "MDG": (-18.8, 46.9), "MUS": (-20.3, 57.6), "SYC": (-4.7, 55.5),
    "CPV": (16.0, -24.0), "STP": (0.2, 6.6), "COM": (-11.6, 43.3),
    "FJI": (-17.7, 178.1), "PNG": (-6.3, 143.9), "WSM": (-13.8, -172.0),
    "TON": (-21.2, -175.2), "VUT": (-15.4, 166.9), "SLB": (-9.6, 160.2),
    "KIR": (1.9, -157.5), "MHL": (7.1, 171.2), "FSM": (7.4, 150.6),
    "PLW": (7.5, 134.6), "TLS": (-8.9, 125.7),
}


@st.cache_data(ttl=3600)
def load_country_codes() -> pd.DataFrame:
    df = pd.read_parquet(BACI_DIR / "country_codes.parquet")
    return df


@st.cache_data(ttl=3600)
def load_bilateral_trade() -> pd.DataFrame:
    df = pd.read_parquet(BACI_DIR / "baci_bilateral_totals.parquet")
    # BACI values are in thousands of USD despite column name; convert to true millions
    df["trade_value_usd_millions"] = df["trade_value_usd_millions"] / 1000
    cc = load_country_codes()
    iso_map = cc.set_index("country_iso3")["country_name"].to_dict()
    df["exporter_name"] = df["iso_o"].map(iso_map)
    df["importer_name"] = df["iso_d"].map(iso_map)
    return df


@st.cache_data(ttl=3600)
def load_sector_flows() -> pd.DataFrame:
    return pd.read_parquet(BACI_DIR / "bilateral_sector_flows.parquet")


@st.cache_data(ttl=3600)
def load_tariffs() -> pd.DataFrame:
    return pd.read_parquet(TAU_DIR / "tau_bilateral.parquet")


@st.cache_data(ttl=3600)
def load_gravity() -> pd.DataFrame:
    cols = [
        "year", "iso3_o", "iso3_d", "distw_harmonic",
        "gdp_o", "gdp_d", "pop_o", "pop_d",
        "contig", "comlang_off", "col_dep_ever", "rta",
    ]
    gv = pd.read_parquet(GRAVITY_DIR / "gravity_v202211.parquet", columns=cols)
    return gv


SECTOR_LABELS = {
    "C10T12": "Food, beverages & tobacco",
    "C13T15": "Textiles, apparel & leather",
    "C16": "Wood & cork products",
    "C17_18": "Paper & printing",
    "C19": "Petroleum & coal products",
    "C20": "Chemicals",
    "C21": "Pharmaceuticals",
    "C22": "Rubber & plastics",
    "C23": "Non-metallic minerals",
    "C24A": "Basic iron & steel",
    "C24B": "Non-ferrous metals",
    "C25": "Fabricated metal products",
    "C26": "Electronics & computers",
    "C27": "Electrical equipment",
    "C28": "Machinery & equipment",
    "C29": "Motor vehicles",
    "C301": "Ships & boats",
    "C302T309": "Other transport equipment",
    "C31T33": "Furniture & other manufacturing",
    "H51": "Air transport equipment",
    "J58T60": "Publishing & broadcasting",
    "J62_63": "IT & information services",
    "M": "Professional & scientific services",
    "N": "Administrative services",
}


def get_coords(iso3: str) -> tuple:
    return COUNTRY_COORDS.get(iso3, (0.0, 0.0))


@st.cache_data(ttl=3600)
def get_year_network_data(year: int) -> pd.DataFrame:
    trade = load_bilateral_trade()
    df = trade[trade["year"] == year].copy()
    return df


@st.cache_data(ttl=3600)
def get_top_n_flows(year: int, n: int = 200) -> pd.DataFrame:
    df = get_year_network_data(year)
    return df.nlargest(n, "trade_value_usd_millions")


@st.cache_data(ttl=3600)
def get_country_summary(year: int) -> pd.DataFrame:
    df = get_year_network_data(year)
    exports = df.groupby("iso_o")["trade_value_usd_millions"].sum().rename("total_exports")
    imports = df.groupby("iso_d")["trade_value_usd_millions"].sum().rename("total_imports")
    n_partners_exp = df.groupby("iso_o")["iso_d"].nunique().rename("export_partners")
    n_partners_imp = df.groupby("iso_d")["iso_o"].nunique().rename("import_partners")
    summary = pd.concat([exports, imports, n_partners_exp, n_partners_imp], axis=1).fillna(0)
    summary["total_trade"] = summary["total_exports"] + summary["total_imports"]
    summary["trade_balance"] = summary["total_exports"] - summary["total_imports"]
    cc = load_country_codes()
    iso_map = cc.set_index("country_iso3")["country_name"].to_dict()
    summary["country_name"] = summary.index.map(iso_map)
    return summary.sort_values("total_trade", ascending=False)
