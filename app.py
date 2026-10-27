
import streamlit as st
import pandas as pd
from pathlib import Path

from risk_review import (
    build_review,
    inspect_application_model,
    load_applicant_snapshot,
    load_graph_snapshot,
)

st.set_page_config(
    page_title="Synthetic Identity Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "graph_risk_results.csv"
THRESHOLD = 0.5416

st.title("🛡️ Synthetic Identity Fraud Detection")
st.caption("Digital Lending | Graph-based Risk Analysis")

if not DATA_PATH.exists():
    st.error("Dataset file missing. Keep graph_risk_results.csv beside app.py.")
    st.stop()

df = pd.read_csv(DATA_PATH)

required = {
    "node_id", "out_degree", "in_degree", "total_degree",
    "ics", "ics_risk", "fraud_probability", "predicted_fraud"
}
if not required.issubset(df.columns):
    st.error("Dataset columns do not match the expected format.")
    st.stop()

df["risk_level"] = pd.cut(
    df["fraud_probability"],
    bins=[-0.001, 0.30, THRESHOLD, 1.001],
    labels=["Low", "Medium", "High"],
    right=False
)

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Risk Explorer", "Risk Review", "Model Information"]
)

st.sidebar.info(
    "Research prototype using a synthetic graph dataset. "
    "Not validated for real lending decisions."
)

if page == "Dashboard":
    st.subheader("Graph Risk Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graph Nodes", f"{len(df):,}")
    c2.metric("High Risk Nodes", int((df["predicted_fraud"] == 1).sum()))
    c3.metric("Average ICS", f'{df["ics"].mean():.3f}')
    c4.metric("Average Risk Score", f'{df["fraud_probability"].mean():.3f}')

    st.subheader("Predicted Risk Distribution")
    risk_counts = (
        df["risk_level"].value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )
    st.bar_chart(risk_counts)

    st.subheader("Highest-Risk Graph Nodes")
    display = df.sort_values(
        "fraud_probability", ascending=False
    ).head(15)

    # Do not display actual labels in the user-facing dashboard.
    st.dataframe(
        display[[
            "node_id", "fraud_probability", "risk_level",
            "ics", "ics_risk", "in_degree", "out_degree"
        ]],
        width="stretch",
        hide_index=True
    )

elif page == "Risk Explorer":
    st.subheader("Explore a Synthetic Graph Node")

    search = st.text_input(
        "Search node ID", placeholder="e.g. NODE_00003"
    ).strip().upper()

    filtered = df
    level = st.selectbox("Filter risk level", ["All", "Low", "Medium", "High"])

    if level != "All":
        filtered = filtered[filtered["risk_level"].astype(str) == level]

    if search:
        filtered = filtered[filtered["node_id"].str.upper().str.contains(search, regex=False)]

    st.write(f"Matching nodes: {len(filtered)}")

    if len(filtered):
        chosen = st.selectbox("Select node", filtered["node_id"].tolist())
        row = df.loc[df["node_id"] == chosen].iloc[0]

        a, b, c = st.columns(3)
        a.metric("Fraud Probability", f'{row["fraud_probability"]:.1%}')
        b.metric("Predicted Risk", str(row["risk_level"]))
        c.metric("Identity Coherence Score", f'{row["ics"]:.3f}')

        d, e, f = st.columns(3)
        d.metric("Incoming Connections", int(row["in_degree"]))
        e.metric("Outgoing Connections", int(row["out_degree"]))
        f.metric("Total Degree", int(row["total_degree"]))

        st.progress(float(row["fraud_probability"]))
        st.caption(
            "Probability is the model's score, not a verified probability "
            "of real-world fraud. Node-level ICS is inherited from its applicant."
        )
    else:
        st.warning("No matching nodes found.")

elif page == "Risk Review":
    st.subheader("Saved Evidence Review")
    st.caption(
        "Look up existing records from the two synthetic modules. "
        "These IDs are not linked, and this page does not score new applicants."
    )

    applicant_path = Path(__file__).parent / "application_stage_final_20k.csv"
    model_path = Path(__file__).parent / "xgb_identity_model.json"
    model_status = inspect_application_model(model_path, applicant_path)

    if model_status["loadable"]:
        st.success("Application model is available for inference.")
    else:
        st.info(f"Live application scoring unavailable: {model_status['reason']}")

    applicant_input, graph_input = st.columns(2)
    applicant_id = applicant_input.text_input(
        "Saved applicant ID", placeholder="e.g. APP100000"
    ).strip()
    graph_node_id = graph_input.text_input(
        "Saved graph node ID", placeholder="e.g. NODE_00003"
    ).strip()

    applicant_record = None
    graph_record = None
    if applicant_id:
        try:
            applicant_record = load_applicant_snapshot(applicant_path, applicant_id)
        except (FileNotFoundError, KeyError, ValueError) as error:
            st.warning(f"Application record unavailable: {error}")

    if graph_node_id:
        try:
            graph_record = load_graph_snapshot(DATA_PATH, graph_node_id)
        except (FileNotFoundError, KeyError, ValueError) as error:
            st.warning(f"Graph record unavailable: {error}")

    review = build_review(applicant_record, graph_record)
    application_column, graph_column = st.columns(2)

    with application_column:
        st.markdown("**Module A: Application-stage saved output**")
        if applicant_record:
            st.metric("Historical XGBoost score", f"{applicant_record['xgb_proba']:.1%}")
            st.metric("Historical blended risk", f"{applicant_record['final_identity_risk_score']:.1%}")
            st.metric("Identity coherence", f"{applicant_record['identity_coherence_score']:.3f}")
            st.write("Saved risk category:", applicant_record["risk_category"])
            st.caption(applicant_record["score_origin"])
        else:
            st.write("No applicant record selected.")

    with graph_column:
        st.markdown("**Module B: Precomputed graph-node output**")
        if graph_record:
            st.metric("Saved graph risk score", f"{graph_record['fraud_probability']:.1%}")
            st.metric("Graph ICS", f"{graph_record['ics']:.3f}")
            st.metric("Predicted fraud flag", str(graph_record["predicted_fraud"]))
            st.write(
                "Connections (in / out / total):",
                f"{graph_record['in_degree']:.0f} / "
                f"{graph_record['out_degree']:.0f} / "
                f"{graph_record['total_degree']:.0f}",
            )
            st.caption(graph_record["score_origin"])
        else:
            st.write("No graph node selected.")

    st.warning(review["decision_status"])

elif page == "Model Information":
    st.subheader("Methodology")

    st.markdown("""
    This prototype explores precomputed outputs from a synthetic,
    transaction-linked graph.

    **Selected model:** Coherence-Aware Graph Attention Network (GAT),
    with lambda = 2.0.

    **Node features:** outgoing degree, incoming degree, total degree,
    Identity Coherence Score (ICS), and ICS-derived risk.

    **Prediction threshold:** 0.5416, selected using the validation set
    in the existing experiment.

    **Important limitation:** This app explores existing graph nodes.
    It does not create a new applicant graph or score arbitrary new
    applicants live. The dataset is synthetic and includes intentionally
    encoded fraud-related patterns.
    """)

    st.subheader("Existing Test Results")
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        "Score": [0.9078, 0.7684, 0.8395, 0.8024, 0.9552, 0.8350]
    })
    st.dataframe(metrics, hide_index=True, width="stretch")

    st.warning(
        "These are the previously recorded test results for lambda = 2.0. "
        "They are reported for the existing experiment, not recomputed by this app."
    )
