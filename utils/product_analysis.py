"""
Product-level (HS02) trade analysis.

Provides deep dives into specific product categories, including
supply chain analysis for EV batteries, semiconductors, and other
strategic sectors.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WAREHOUSE_BACI = Path.home() / "trade_data_warehouse" / "baci"
_BUNDLED_BACI = _PROJECT_ROOT / "data" / "baci"

HS_BY_DYAD_DIR = _WAREHOUSE_BACI / "hs_by_dyad"  # Large data, not bundled
PRODUCT_CODES_PATH = (
    _WAREHOUSE_BACI / "product_codes_hs02.parquet"
    if (_WAREHOUSE_BACI / "product_codes_hs02.parquet").exists()
    else _BUNDLED_BACI / "product_codes_hs02.parquet"
)


# Strategic product groups by HS02 code prefixes
STRATEGIC_PRODUCTS = {
    "EV Battery Supply Chain": {
        "description": "Lithium, cobalt, nickel, battery cells, and EV components",
        "codes": {
            "2825": "Lithium oxide & hydroxide",
            "2836": "Lithium carbonate",
            "8507": "Electric accumulators (batteries)",
            "8501": "Electric motors & generators",
            "8503": "Parts for electric motors",
            "8504": "Transformers & inductors",
            "7502": "Unwrought nickel",
            "8105": "Cobalt mattes & intermediates",
            "2602": "Manganese ores",
            "2844": "Rare earths & compounds",
            "8541": "Semiconductor devices",
            "8542": "Electronic integrated circuits",
            "7601": "Unwrought aluminum",
            "7403": "Refined copper",
            "3901": "Polymers of ethylene (separators)",
        },
    },
    "Semiconductors & Electronics": {
        "description": "Chips, wafers, circuit boards, and electronic components",
        "codes": {
            "8541": "Semiconductor devices (diodes, transistors)",
            "8542": "Electronic integrated circuits",
            "8471": "Computers & processing units",
            "8473": "Computer parts & accessories",
            "8517": "Telecom equipment (phones)",
            "8534": "Printed circuit boards",
            "8540": "Thermionic valves & tubes",
            "8532": "Electrical capacitors",
            "8533": "Electrical resistors",
            "2804": "Silicon (semiconductor grade)",
            "3818": "Chemical elements (doped for electronics)",
            "9013": "Liquid crystal displays",
            "8486": "Semiconductor manufacturing equipment",
        },
    },
    "Critical Minerals & Rare Earths": {
        "description": "Strategic minerals essential for defense and technology",
        "codes": {
            "2844": "Radioactive elements & isotopes",
            "2846": "Rare earth compounds",
            "2611": "Tungsten ores",
            "2602": "Manganese ores",
            "2603": "Copper ores",
            "2604": "Nickel ores",
            "2605": "Cobalt ores",
            "2606": "Aluminum ores (bauxite)",
            "2607": "Lead ores",
            "2608": "Zinc ores",
            "2609": "Tin ores",
            "2610": "Chromium ores",
            "2612": "Uranium & thorium ores",
            "2615": "Niobium, tantalum, vanadium ores",
            "7110": "Platinum group metals",
        },
    },
    "Pharmaceuticals & Medical": {
        "description": "Active pharmaceutical ingredients and medical devices",
        "codes": {
            "3003": "Medicaments (not dosed)",
            "3004": "Medicaments (dosed for retail)",
            "2941": "Antibiotics",
            "2942": "Other organic compounds",
            "2933": "Heterocyclic compounds",
            "2934": "Nucleic acids & compounds",
            "9018": "Medical/surgical instruments",
            "9021": "Orthopedic appliances",
            "9022": "X-ray/radiation apparatus",
            "3002": "Vaccines & blood products",
        },
    },
    "Energy & Petroleum": {
        "description": "Crude oil, natural gas, refined products",
        "codes": {
            "2709": "Crude petroleum",
            "2710": "Petroleum oils (refined)",
            "2711": "Natural gas (LNG/pipeline)",
            "2701": "Coal",
            "2716": "Electrical energy",
            "8502": "Electric generating sets",
            "8411": "Turbojets & gas turbines",
        },
    },
    "Agriculture & Food Security": {
        "description": "Staple crops, fertilizers, and food chain",
        "codes": {
            "1001": "Wheat",
            "1005": "Corn (maize)",
            "1201": "Soybeans",
            "1006": "Rice",
            "3102": "Nitrogen fertilizers",
            "3103": "Phosphate fertilizers",
            "3104": "Potassium fertilizers",
            "3105": "Mixed fertilizers",
            "0201": "Beef (fresh/chilled)",
            "0207": "Poultry meat",
        },
    },
    "Motor Vehicles & Parts": {
        "description": "Automobiles, trucks, and automotive supply chain",
        "codes": {
            "8703": "Passenger vehicles",
            "8704": "Trucks & commercial vehicles",
            "8708": "Motor vehicle parts & accessories",
            "8706": "Chassis fitted with engines",
            "8707": "Vehicle bodies",
            "4011": "Rubber tires (pneumatic)",
            "7009": "Glass mirrors (rear-view)",
            "8409": "Engine parts",
            "8511": "Ignition/starting equipment",
            "8512": "Vehicle lighting",
        },
    },
}


@st.cache_data(ttl=3600)
def load_product_codes() -> pd.DataFrame:
    return pd.read_parquet(PRODUCT_CODES_PATH)


@st.cache_data(ttl=3600)
def load_product_trade(year: int) -> pd.DataFrame:
    """Load HS-level bilateral trade for a given year."""
    filepath = HS_BY_DYAD_DIR / f"baci_hs_by_dyad_Y{year}.parquet"
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_parquet(filepath)
    # Values are in thousands of USD; convert to millions
    df["value"] = df["value"] / 1000
    return df


@st.cache_data(ttl=3600)
def get_strategic_product_flows(year: int, product_group: str, focus_country: str = "USA") -> dict:
    """Analyze trade flows for a strategic product group."""
    if product_group not in STRATEGIC_PRODUCTS:
        return {}

    group = STRATEGIC_PRODUCTS[product_group]
    hs_codes = list(group["codes"].keys())

    df = load_product_trade(year)
    if len(df) == 0:
        return {}

    # HS codes are 6-digit; our codes might be 4-digit prefixes
    # Match by prefix
    df["hs4"] = df["hs02"].astype(str).str[:4]
    filtered = df[df["hs4"].isin(hs_codes)].copy()
    filtered["product_name"] = filtered["hs4"].map(group["codes"])

    # Imports to focus country
    imports = filtered[filtered["iso_d"] == focus_country].copy()
    imports_by_source = imports.groupby("iso_o")["value"].sum().sort_values(ascending=False)
    imports_by_product = imports.groupby(["hs4", "product_name"])["value"].sum().sort_values(ascending=False)

    # Exports from focus country
    exports = filtered[filtered["iso_o"] == focus_country].copy()
    exports_by_dest = exports.groupby("iso_d")["value"].sum().sort_values(ascending=False)

    # Global flows for this product group
    global_by_exporter = filtered.groupby("iso_o")["value"].sum().sort_values(ascending=False)
    global_total = filtered["value"].sum()

    # Concentration metrics
    if len(imports_by_source) > 0:
        total_imports = imports_by_source.sum()
        import_shares = imports_by_source / total_imports
        import_hhi = (import_shares ** 2).sum()
        top_source = imports_by_source.index[0]
        top_source_share = import_shares.iloc[0]
    else:
        import_hhi = 0
        top_source = None
        top_source_share = 0
        total_imports = 0

    return {
        "group_name": product_group,
        "description": group["description"],
        "imports_by_source": imports_by_source.reset_index(),
        "imports_by_product": imports_by_product.reset_index(),
        "exports_by_dest": exports_by_dest.reset_index(),
        "global_by_exporter": global_by_exporter.reset_index(),
        "global_total": global_total,
        "total_imports": total_imports,
        "import_hhi": import_hhi,
        "top_source": top_source,
        "top_source_share": top_source_share,
        "n_products": len(imports_by_product),
        "n_sources": len(imports_by_source),
    }


@st.cache_data(ttl=3600)
def get_product_evolution(product_group: str, focus_country: str = "USA") -> pd.DataFrame:
    """Track a product group's trade over time."""
    if product_group not in STRATEGIC_PRODUCTS:
        return pd.DataFrame()

    group = STRATEGIC_PRODUCTS[product_group]
    hs_codes = list(group["codes"].keys())

    records = []
    available_years = sorted([
        int(f.stem.split("Y")[1])
        for f in HS_BY_DYAD_DIR.iterdir()
        if f.suffix == ".parquet"
    ])

    for year in available_years:
        df = load_product_trade(year)
        if len(df) == 0:
            continue
        df["hs4"] = df["hs02"].astype(str).str[:4]
        filtered = df[df["hs4"].isin(hs_codes)]

        imports = filtered[filtered["iso_d"] == focus_country]
        total_imports = imports["value"].sum()

        # Top 5 sources
        top_sources = imports.groupby("iso_o")["value"].sum().nlargest(5)

        record = {"year": year, "total_imports": total_imports}
        for iso, val in top_sources.items():
            record[iso] = val
        records.append(record)

    return pd.DataFrame(records).fillna(0)
