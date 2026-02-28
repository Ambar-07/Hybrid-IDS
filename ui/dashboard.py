"""
Hybrid IDS — Streamlit Dashboard
Run: streamlit run ui/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.feature_extractor import FeatureExtractor
from engine.rule_engine import RuleEngine
from engine.ml_detector import MLDetector
from engine.fusion import FusionLayer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace; }

[data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #0d1225 !important;
    border-right: 1px solid #1e2a45;
}
.metric-card {
    background: #0d1225;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.block-badge {
    background: #ff0033;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
}
.alert-badge {
    background: #ff8c00;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
}
.allow-badge {
    background: #00c853;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
}
.risk-bar-bg {
    background: #1e2a45;
    border-radius: 6px;
    height: 10px;
    width: 100%;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.sub-title {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0;
}
div[data-testid="stMetric"] {
    background: #0d1225;
    border: 1px solid #1e2a45;
    border-radius: 10px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)


# ── Cache models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    extractor    = FeatureExtractor()
    rule_engine  = RuleEngine(rules_path="config/rules.yaml")
    ml_detector  = MLDetector()
    fusion       = FusionLayer()

    model_path = "models/isolation_forest.pkl"
    if os.path.exists(model_path):
        ml_detector.load(model_path)

    return extractor, rule_engine, ml_detector, fusion


extractor, rule_engine, ml_detector, fusion = load_components()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Hybrid IDS")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📂 Analyze Traffic", "⚙️ Train Model", "📋 Rules Viewer"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Status**")
    if ml_detector.trained:
        st.success("✅ ML Model Loaded")
    else:
        st.warning("⚠️ No model — train first")

    st.markdown(f"**Rules Loaded:** `{len(rule_engine.rules)}`")
    st.markdown("---")

    # Fusion weights
    st.markdown("**Fusion Weights**")
    rule_w = st.slider("Rule Weight", 0.0, 1.0, 0.55, 0.05)
    ml_w   = 1.0 - rule_w
    st.caption(f"ML Weight: {ml_w:.2f}")
    fusion.rule_weight = rule_w
    fusion.ml_weight   = ml_w

    st.markdown("**Decision Thresholds**")
    fusion.allow_threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.35, 0.05)
    fusion.block_threshold = st.slider("Block Threshold", 0.1, 1.0, 0.70, 0.05)


# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(action):
    cls = {"BLOCK": "block-badge", "ALERT": "alert-badge", "ALLOW": "allow-badge"}.get(action, "")
    return f'<span class="{cls}">{action}</span>'


def analyze_df(df: pd.DataFrame, max_rows=300):
    df = df.head(max_rows)
    extractor.fit(df)
    flows = extractor.transform_df(df)

    rows = []
    for i, ff in enumerate(flows):
        rule_out = rule_engine.evaluate(ff)
        if not ml_detector.trained:
            from engine.ml_detector import MLResult
            ml_out = MLResult(anomaly_score=0.0, is_anomaly=False, raw_score=0.0, confidence=0.0)
        else:
            ml_out = ml_detector.predict(ff)
        decision = fusion.decide(rule_out, ml_out)

        rows.append({
            "Action":        decision.action,
            "Risk Score":    decision.risk_score,
            "Rule Hit":      "✅" if decision.rule_matched else "—",
            "Matched Rules": ", ".join(decision.matched_rule_names) or "—",
            "Rule Severity": decision.rule_severity,
            "ML Score":      round(decision.ml_anomaly_score, 3),
            "Dst Port":      ff.dst_port,
            "Label":         ff.label,
            "Reasoning":     decision.reasoning,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dashboard ─────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<p class="header-title">Hybrid Intrusion Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Rule-Based + ML Anomaly Detection + Risk Fusion</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### System Architecture")
    cols = st.columns(5)
    steps = [
        ("📡", "Traffic Input", "Raw network flows or PCAP/CSV"),
        ("🔧", "Feature Extraction", "35 statistical features per flow"),
        ("📏", "Rule Engine", "12 signature rules across 5 categories"),
        ("🧠", "ML Detection", "Isolation Forest anomaly scoring"),
        ("⚖️", "Risk Fusion", "Weighted decision → ALLOW/ALERT/BLOCK"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:700;margin:8px 0;color:#00d4ff">{title}</div>
                <div style="font-size:0.8rem;color:#64748b">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Demo — Sample Flow Analysis")

    # Generate a demo flow
    if st.button("🔄 Simulate Random Traffic Flow"):
        scenario = np.random.choice(["Normal", "Port Scan", "SYN Flood", "Brute Force SSH", "Unknown Anomaly"])

        demo_flows = {
            "Normal": {
                "Flow Duration": 50000, "Protocol": 6, "Source Port": 55234,
                "Destination Port": 443, "Total Fwd Packets": 12,
                "Total Length of Fwd Packets": 2400, "Flow Bytes/s": 480,
                "Flow Packets/s": 8, "Fwd Packet Length Mean": 200,
                "Bwd Packet Length Mean": 180, "SYN Flag Count": 1,
                "ACK Flag Count": 10, "FIN Flag Count": 1, "RST Flag Count": 0,
                "PSH Flag Count": 3, "URG Flag Count": 0,
                "Average Packet Size": 200, "Active Mean": 1000, "Idle Mean": 500,
            },
            "Port Scan": {
                "Flow Duration": 100, "Protocol": 6, "Source Port": 45678,
                "Destination Port": 22, "Total Fwd Packets": 50,
                "Total Length of Fwd Packets": 300, "Flow Bytes/s": 3000,
                "Flow Packets/s": 500, "Fwd Packet Length Mean": 6,
                "Bwd Packet Length Mean": 0, "SYN Flag Count": 48,
                "ACK Flag Count": 0, "FIN Flag Count": 2, "RST Flag Count": 1,
                "PSH Flag Count": 0, "URG Flag Count": 0,
                "Average Packet Size": 6, "Active Mean": 50, "Idle Mean": 10,
            },
            "SYN Flood": {
                "Flow Duration": 500, "Protocol": 6, "Source Port": 12345,
                "Destination Port": 80, "Total Fwd Packets": 5000,
                "Total Length of Fwd Packets": 30000, "Flow Bytes/s": 60000,
                "Flow Packets/s": 10000, "Fwd Packet Length Mean": 6,
                "Bwd Packet Length Mean": 0, "SYN Flag Count": 4990,
                "ACK Flag Count": 2, "FIN Flag Count": 0, "RST Flag Count": 0,
                "PSH Flag Count": 0, "URG Flag Count": 0,
                "Average Packet Size": 6, "Active Mean": 100, "Idle Mean": 0,
            },
            "Brute Force SSH": {
                "Flow Duration": 20000, "Protocol": 6, "Source Port": 34567,
                "Destination Port": 22, "Total Fwd Packets": 200,
                "Total Length of Fwd Packets": 8000, "Flow Bytes/s": 400,
                "Flow Packets/s": 10, "Fwd Packet Length Mean": 40,
                "Bwd Packet Length Mean": 60, "SYN Flag Count": 5,
                "ACK Flag Count": 190, "FIN Flag Count": 2, "RST Flag Count": 40,
                "PSH Flag Count": 50, "URG Flag Count": 0,
                "Average Packet Size": 40, "Active Mean": 500, "Idle Mean": 200,
            },
            "Unknown Anomaly": {
                "Flow Duration": 99999, "Protocol": 17, "Source Port": 44444,
                "Destination Port": 9999, "Total Fwd Packets": 1500,
                "Total Length of Fwd Packets": 999999, "Flow Bytes/s": 9999,
                "Flow Packets/s": 15, "Fwd Packet Length Mean": 666,
                "Bwd Packet Length Mean": 999, "SYN Flag Count": 3,
                "ACK Flag Count": 1400, "FIN Flag Count": 0, "RST Flag Count": 0,
                "PSH Flag Count": 200, "URG Flag Count": 15,
                "Average Packet Size": 666, "Active Mean": 99999, "Idle Mean": 0,
            },
        }

        raw = demo_flows[scenario]

        # Process
        extractor.fit(pd.DataFrame([raw]))
        ff = extractor.transform(raw)
        rule_out = rule_engine.evaluate(ff)
        if ml_detector.trained:
            ml_out = ml_detector.predict(ff)
        else:
            from engine.ml_detector import MLResult
            ml_out = MLResult(
                anomaly_score=np.random.uniform(0.6, 0.95) if scenario != "Normal" else np.random.uniform(0.05, 0.25),
                is_anomaly=scenario != "Normal",
                raw_score=-0.3,
                confidence=0.8,
            )
        decision = fusion.decide(rule_out, ml_out)

        # Display
        st.markdown(f"**Simulated Scenario:** `{scenario}`")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Action",     f"{decision.icon} {decision.action}")
        c2.metric("Risk Score", f"{decision.risk_score:.2f}")
        c3.metric("Rule Score", f"{decision.rule_score:.2f}")
        c4.metric("ML Score",   f"{decision.ml_score:.2f}")

        # Risk gauge bar
        pct = int(decision.risk_score * 100)
        color = {"ALLOW": "#00c853", "ALERT": "#ff8c00", "BLOCK": "#ff0033"}.get(decision.action, "#ccc")
        st.markdown(f"""
        <div style="margin:15px 0">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-weight:700">Risk Score</span>
                <span style="color:{color};font-weight:700">{pct}%</span>
            </div>
            <div style="background:#1e2a45;border-radius:6px;height:14px">
                <div style="background:{color};width:{pct}%;height:14px;border-radius:6px;transition:width 0.5s"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Reasoning:** {decision.reasoning}")

        if rule_out.any_match:
            st.markdown("**Matched Signature Rules:**")
            for r in rule_out.matched_rules:
                st.code(f"[{r.rule_id}] {r.rule_name} | Severity: {r.severity} | Confidence: {r.confidence}")


# ── Analyze Traffic ───────────────────────────────────────────────────────────
elif page == "📂 Analyze Traffic":
    st.markdown("## 📂 Analyze Traffic")
    st.markdown("Upload a CSV file (CICIDS2017 format or any flow CSV)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        max_rows = st.slider("Max rows to analyze", 50, 2000, 300, 50)

        if st.button("🚀 Run Analysis"):
            with st.spinner("Analyzing traffic..."):
                df_raw = pd.read_csv(uploaded, low_memory=False)
                st.info(f"Loaded {len(df_raw)} rows. Analyzing first {max_rows}...")

                results_df = analyze_df(df_raw, max_rows=max_rows)

            # Summary
            st.markdown("---")
            st.markdown("### Results Summary")
            c1, c2, c3, c4 = st.columns(4)
            total = len(results_df)
            blocks = (results_df["Action"] == "BLOCK").sum()
            alerts = (results_df["Action"] == "ALERT").sum()
            allows = (results_df["Action"] == "ALLOW").sum()

            c1.metric("Total Flows",  total)
            c2.metric("🚫 Blocked",  blocks, delta=f"{blocks/total*100:.1f}%")
            c3.metric("⚠️ Alerts",   alerts, delta=f"{alerts/total*100:.1f}%")
            c4.metric("✅ Allowed",  allows, delta=f"{allows/total*100:.1f}%")

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                action_counts = results_df["Action"].value_counts()
                st.bar_chart(action_counts)

            with col2:
                st.markdown("**Risk Score Distribution**")
                hist_data = pd.DataFrame({"Risk Score": results_df["Risk Score"]})
                st.bar_chart(hist_data["Risk Score"].value_counts(bins=10, sort=False))

            # Table
            st.markdown("### Flow-by-Flow Results")
            st.dataframe(
                results_df.drop(columns=["Reasoning"]),
                use_container_width=True,
                height=400,
            )

            # Download
            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results CSV",
                csv_out,
                "ids_results.csv",
                "text/csv",
            )


# ── Train Model ───────────────────────────────────────────────────────────────
elif page == "⚙️ Train Model":
    st.markdown("## ⚙️ Train ML Model")
    st.markdown("Upload a CSV with **normal (BENIGN) traffic** to train the Isolation Forest.")

    train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
    contamination = st.slider("Contamination (expected anomaly % in training data)", 0.01, 0.20, 0.05, 0.01)
    n_estimators  = st.slider("Number of Trees", 50, 300, 100, 50)

    if train_file and st.button("🧠 Train Model"):
        with st.spinner("Training..."):
            df_train = pd.read_csv(train_file, low_memory=False)

            # Filter to benign
            label_col = None
            for col in df_train.columns:
                if col.strip().lower() == "label":
                    label_col = col
                    break

            if label_col:
                normal_df = df_train[df_train[label_col].str.upper().str.contains("BENIGN")]
                st.info(f"Found {len(normal_df)} BENIGN flows out of {len(df_train)} total.")
            else:
                normal_df = df_train
                st.info(f"No 'Label' column found. Using all {len(df_train)} rows.")

            extractor.fit(df_train)
            normal_flows = extractor.transform_df(normal_df)

            ml_detector.contamination = contamination
            ml_detector.n_estimators  = n_estimators
            ml_detector.train(normal_flows)
            ml_detector.save("models/isolation_forest.pkl")

        st.success(f"✅ Model trained on {len(normal_flows)} flows and saved!")
        st.balloons()


# ── Rules Viewer ──────────────────────────────────────────────────────────────
elif page == "📋 Rules Viewer":
    st.markdown("## 📋 Signature Rules")
    st.markdown(f"**{len(rule_engine.rules)} rules loaded** from `config/rules.yaml`")

    # Filter
    categories = list(set(r.get("category", "Misc") for r in rule_engine.rules))
    selected_cats = st.multiselect("Filter by Category", categories, default=categories)

    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    selected_sev = st.multiselect("Filter by Severity", severities, default=severities)

    filtered = [
        r for r in rule_engine.rules
        if r.get("category", "Misc") in selected_cats
        and r.get("severity", "LOW") in selected_sev
    ]

    SEV_COLORS = {
        "CRITICAL": "#ff0033",
        "HIGH":     "#ff8c00",
        "MEDIUM":   "#ffd700",
        "LOW":      "#00c853",
    }

    for rule in filtered:
        sev   = rule.get("severity", "LOW")
        color = SEV_COLORS.get(sev, "#888")
        with st.expander(f"[{rule['id']}] {rule['name']}  |  {sev}  |  {rule.get('category')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**ID:** `{rule['id']}`")
                st.markdown(f"**Category:** `{rule.get('category')}`")
                st.markdown(f"**Severity:** <span style='color:{color};font-weight:700'>{sev}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{rule.get('confidence', 0.8)}`")
            with col2:
                st.markdown("**Conditions (ALL must be true):**")
                for cond in rule.get("conditions", []):
                    st.code(f"{cond['field']}  {cond['op']}  {cond['value']}")
