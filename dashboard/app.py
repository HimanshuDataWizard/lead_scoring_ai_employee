from pathlib import Path
import json
import math
import sys
from typing import Dict

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lead_scoring.config import DATA_PATH
from lead_scoring.inference import load_artifacts, predict_one

st.set_page_config(page_title="Lead Scoring Dashboard", page_icon="LS", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

    :root {
      --bg: #eef3f8;
      --panel: #ffffff;
      --text: #0f1f33;
      --muted: #4f6074;
      --line: #d8e1eb;
      --primary: #0a4fbf;
      --good: #0b7f5d;
      --warn: #9a6600;
      --bad: #b8344b;
    }

    html, body, [class*="css"] {
      font-family: 'Source Sans 3', sans-serif;
      color: var(--text);
    }

    .stApp {
      background: linear-gradient(180deg, #f6f9fc 0%, var(--bg) 100%);
    }

    .header {
      background: linear-gradient(120deg, #0a2f6b 0%, #0a4fbf 100%);
      color: #ffffff;
      border-radius: 16px;
      padding: 18px 22px;
      border: 1px solid rgba(255,255,255,0.2);
      box-shadow: 0 10px 28px rgba(10, 44, 102, 0.25);
      margin-bottom: 14px;
    }

    .header h1 {
      margin: 0;
      font-size: 1.45rem;
      font-weight: 700;
      letter-spacing: 0.2px;
    }

    .header p {
      margin: 6px 0 0;
      color: #dbe8ff;
      font-size: 0.96rem;
    }

    .chip-row {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .chip {
      background: rgba(255, 255, 255, 0.18);
      border: 1px solid rgba(255, 255, 255, 0.26);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #f3f8ff;
    }

    .section-title {
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      font-weight: 700;
      margin: 4px 0 8px;
    }

    .decision-box {
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 5px solid var(--primary);
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: 0 8px 24px rgba(15, 31, 51, 0.06);
    }

    .decision-box h3 {
      margin: 0;
      font-size: 1.02rem;
      color: var(--text);
    }

    .decision-box p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.35;
    }

    .band {
      margin-top: 10px;
      border-radius: 10px;
      padding: 9px 11px;
      border: 1px solid;
      font-weight: 700;
      font-size: 0.9rem;
    }

    .band-high { background: #e6f7f1; color: var(--good); border-color: #bfe8d9; }
    .band-medium { background: #fff6e6; color: var(--warn); border-color: #efd8a8; }
    .band-low { background: #fff0f3; color: var(--bad); border-color: #efc0cb; }

    div[data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 10px;
    }

    div[data-testid="stMetricLabel"] {
      color: var(--muted);
      font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
      color: var(--text);
      font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _load_model_and_metrics():
    return load_artifacts()


@st.cache_data
def _dataset_stats() -> Dict[str, float]:
    df = pd.read_csv(DATA_PATH)
    return {
        "company_size_median": float(df["company_size"].median()),
        "prev_purchases_median": float(df["prev_purchases"].median()),
        "response_time_median": float(df["response_time"].median()),
        "last_contact_median": float(df["last_contact"].median()),
        "deal_value_median": float(df["deal_value"].median()),
    }


PRESETS = {
    "Executive Demo": {
        "company_size": 280,
        "industry": "Pharma",
        "region": "Mumbai",
        "prev_purchases": 5,
        "response_time": 8,
        "last_contact": 6,
        "source": "Referral",
        "deal_value": 29000.0,
    },
    "Growth Inbound": {
        "company_size": 160,
        "industry": "E-commerce",
        "region": "Delhi",
        "prev_purchases": 3,
        "response_time": 10,
        "last_contact": 9,
        "source": "Website",
        "deal_value": 17000.0,
    },
    "Cold Prospect": {
        "company_size": 50,
        "industry": "Retail",
        "region": "Chennai",
        "prev_purchases": 0,
        "response_time": 30,
        "last_contact": 40,
        "source": "Email",
        "deal_value": 7000.0,
    },
    "Large Slow Account": {
        "company_size": 880,
        "industry": "Manufacturing",
        "region": "Bangalore",
        "prev_purchases": 2,
        "response_time": 24,
        "last_contact": 34,
        "source": "Phone",
        "deal_value": 62000.0,
    },
}


def _apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def _read_query_scenario() -> str:
    try:
        scenario = st.query_params.get("scenario")
    except Exception:
        params = st.experimental_get_query_params()
        raw = params.get("scenario")
        scenario = raw[0] if isinstance(raw, list) and raw else None

    if scenario in PRESETS:
        return scenario
    return "Growth Inbound"


def _priority(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    if prob >= 0.45:
        return "Medium"
    return "Low"


def _action(priority: str) -> str:
    if priority == "High":
        return "Immediate sales action: senior rep assignment + same-day proposal track."
    if priority == "Medium":
        return "Targeted nurture action: send value proof sequence and schedule a follow-up call."
    return "Long-cycle nurture action: automation touchpoints and re-score after next engagement."


def _signal_breakdown(payload: Dict[str, float], stats: Dict[str, float]) -> pd.DataFrame:
    signals = {
        "Purchase History": (payload["prev_purchases"] - stats["prev_purchases_median"]) / (stats["prev_purchases_median"] + 1.0),
        "Deal Size": math.log1p(payload["deal_value"]) - math.log1p(stats["deal_value_median"]),
        "Response Speed": (stats["response_time_median"] - payload["response_time"]) / (stats["response_time_median"] + 1.0),
        "Contact Freshness": (stats["last_contact_median"] - payload["last_contact"]) / (stats["last_contact_median"] + 1.0),
        "Company Scale": math.log1p(payload["company_size"]) - math.log1p(stats["company_size_median"]),
    }

    rows = []
    for name, value in signals.items():
        rows.append({"signal": name, "impact": round(max(min(value * 100, 100), -100), 1)})

    return pd.DataFrame(rows).sort_values("impact", ascending=False)


try:
    model, metrics = _load_model_and_metrics()
    stats = _dataset_stats()
except Exception:
    st.error("Model artifacts missing. Run `python3 scripts/train.py` first.")
    st.stop()

threshold = float(metrics.get("decision_threshold", 0.5))
model_name = metrics.get("model", {}).get("name", "n/a")
calibration = metrics.get("model", {}).get("calibration", "none")

if "scenario" not in st.session_state:
    st.session_state["scenario"] = _read_query_scenario()
    _apply_preset(st.session_state["scenario"])

st.markdown(
    f"""
    <div class="header">
      <h1>Lead Scoring Dashboard</h1>
      <p>Professional scoring workspace for sales prioritization and decision reporting.</p>
      <div class="chip-row">
        <span class="chip">Model: {model_name}</span>
        <span class="chip">Calibration: {calibration}</span>
        <span class="chip">Decision Threshold: {threshold:.2f}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Scenario Presets")
    scenario = st.selectbox("Select Scenario", list(PRESETS.keys()), key="scenario")
    if st.button("Load Scenario", use_container_width=True):
        _apply_preset(scenario)

    st.markdown("---")
    st.caption("Load a scenario and click Run Scoring Decision.")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("ROC-AUC", metrics.get("roc_auc", "n/a"))
kpi2.metric("F1", metrics.get("f1", "n/a"))
kpi3.metric("Precision", metrics.get("precision", "n/a"))
kpi4.metric("Recall", metrics.get("recall", "n/a"))
kpi5.metric("Rows", metrics.get("dataset", {}).get("rows", "n/a"))

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("<div class='section-title'>Lead Inputs</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        company_size = st.number_input("Company Size", min_value=1, step=1, key="company_size")
        industry = st.selectbox("Industry", ["Pharma", "Retail", "E-commerce", "Manufacturing"], key="industry")
        prev_purchases = st.number_input("Previous Purchases", min_value=0, step=1, key="prev_purchases")
        response_time = st.number_input("Response Time (days)", min_value=0, step=1, key="response_time")
    with b:
        region = st.selectbox("Region", ["Mumbai", "Delhi", "Chennai", "Bangalore"], key="region")
        source = st.selectbox("Lead Source", ["Website", "Email", "Phone", "Referral"], key="source")
        last_contact = st.number_input("Days Since Last Contact", min_value=0, step=1, key="last_contact")
        deal_value = st.number_input("Deal Value ($)", min_value=0.0, step=500.0, key="deal_value")

    score = st.button("Run Scoring Decision", type="primary", use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Model Governance</div>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.metric("Threshold", metrics.get("decision_threshold", "n/a"))
    g2.metric("Calibration", calibration)
    g3.metric("Model", model_name)

    compare = pd.DataFrame(metrics.get("comparison", []))
    if not compare.empty:
        st.dataframe(compare.sort_values("val_roc_auc", ascending=False), hide_index=True, use_container_width=True)

if score:
    payload = {
        "company_size": int(company_size),
        "industry": industry,
        "region": region,
        "prev_purchases": int(prev_purchases),
        "response_time": int(response_time),
        "last_contact": int(last_contact),
        "source": source,
        "deal_value": float(deal_value),
    }
    result = predict_one(model, payload, threshold=threshold)
    prob = float(result["conversion_probability"])
    band = _priority(prob)

    st.markdown("<div class='section-title'>Decision Output</div>", unsafe_allow_html=True)
    dleft, dright = st.columns([0.95, 1.05], gap="large")

    with dleft:
        st.markdown(
            f"""
            <div class="decision-box">
              <h3>{'Pursue Now' if result['prediction'] == 1 else 'Nurture First'}</h3>
              <p>{_action(band)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Probability", f"{prob * 100:.1f}%")
        m2.metric("Class", "Convert" if result["prediction"] == 1 else "No Convert")
        m3.metric("Band", band)

        css = {"High": "band-high", "Medium": "band-medium", "Low": "band-low"}[band]
        st.markdown(
            f"<div class='band {css}'>Priority: {band} | Threshold: {threshold:.2f}</div>",
            unsafe_allow_html=True,
        )

        st.progress(prob, text=f"Confidence: {prob * 100:.1f}%")

        card = {
            "scenario": st.session_state.get("scenario"),
            "input": payload,
            "output": result,
            "recommended_action": _action(band),
            "model": {
                "name": model_name,
                "calibration": calibration,
                "decision_threshold": threshold,
            },
        }
        st.download_button(
            "Download Decision Report",
            data=json.dumps(card, indent=2),
            file_name="lead_scoring_decision.json",
            mime="application/json",
            use_container_width=True,
        )

    with dright:
        st.markdown("<div class='section-title'>Signal Breakdown</div>", unsafe_allow_html=True)
        signal_df = _signal_breakdown(payload, stats)
        st.bar_chart(signal_df.set_index("signal"), color="#0a4fbf")
        st.dataframe(signal_df, hide_index=True, use_container_width=True)

st.markdown("---")
ux1, ux2, ux3 = st.columns(3)
ux1.info("Inbound Prioritization: route highest probability leads to fastest response.")
ux2.info("Pipeline Review: rescore weekly and focus rep effort where conversion odds are strongest.")
ux3.info("Campaign Segmentation: sync priority bands to CRM lifecycle automations.")
