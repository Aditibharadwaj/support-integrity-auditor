"""
app.py — Support Integrity Auditor (SIA) — Streamlit Web App

Features:
  • Single-ticket form input → binary judgment + Evidence Dossier
  • Batch CSV upload → predictions for all tickets + downloadable dossiers
  • Priority Mismatch Dashboard:
      - Distribution of flagged vs consistent tickets
      - Mismatch type breakdown (Hidden Crisis / False Alarm)
      - Top contributing keyword signals
      - Severity delta heatmap across categories and channels

Run:
  streamlit run app.py
"""

import json
import os
import pickle
import re
import io
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Constants (must match notebook / train_pipeline.py)
# ──────────────────────────────────────────────────────────────
DISTILBERT_MODEL = "distilbert-base-uncased"
MAX_LEN          = 128
LORA_R           = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.1
DROPOUT          = 0.3
N_METADATA       = 3
N_CLASSES        = 2
MODEL_DIR        = "saved_model"

PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
INV_PRIORITY = {v: k for k, v in PRIORITY_MAP.items()}
SEVERITY_LABEL = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

CRITICAL_WORDS = [
    "outage", "security", "fraud", "breach", "data loss",
    "stolen card", "unauthorized", "cannot login", "system down",
    "payment failed", "account hacked", "service unavailable", "data corruption",
]
HIGH_WORDS = [
    "crash", "error", "failed", "sync", "invoice discrepancy",
    "login issue", "payment issue", "screen freezes", "api error",
    "application crash", "data not syncing",
]

SEVERITY_COLORS = {
    "Low":      "#2ecc71",
    "Medium":   "#f39c12",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}

MISMATCH_COLORS = {
    "Consistent":    "#2ecc71",
    "Hidden Crisis": "#e74c3c",
    "False Alarm":   "#3498db",
}

# ──────────────────────────────────────────────────────────────
# Text & signal helpers  (identical to notebook)
# ──────────────────────────────────────────────────────────────
def clean_ticket_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"^Hi Support,\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"[?.!]", text)[0]
    return text.strip()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_rule_score(text):
    score, evidence = 0, []
    for word in CRITICAL_WORDS:
        if word in text:
            score += 3
            evidence.append(word)
    for word in HIGH_WORDS:
        if word in text:
            score += 2
            evidence.append(word)
    if score >= 6:   sev = 3
    elif score >= 4: sev = 2
    elif score >= 2: sev = 1
    else:            sev = 0
    return sev, ",".join(evidence)


# ──────────────────────────────────────────────────────────────
# Model definition  (identical to notebook)
# ──────────────────────────────────────────────────────────────
class DistilBERTLoRAWithMetadata(nn.Module):
    def __init__(self, model_name, lora_config,
                 n_metadata=N_METADATA, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        base_model   = AutoModel.from_pretrained(model_name)
        self.encoder = get_peft_model(base_model, lora_config)
        hidden_size  = base_model.config.hidden_size
        self.meta_proj = nn.Sequential(
            nn.Linear(n_metadata, 32), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask, channel, category, res_time):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        meta       = torch.stack([channel, category, res_time], dim=1)
        meta_out   = self.meta_proj(meta)
        combined   = torch.cat([cls_output, meta_out], dim=1)
        return self.classifier(combined)


# ──────────────────────────────────────────────────────────────
# Cached artifact loading
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SIA model …")
def load_artifacts():
    config_path = os.path.join(MODEL_DIR, "training_config.json")
    if not os.path.exists(config_path):
        return None, None, None, None, None, None, "Model not found. Run train_pipeline.py first."

    with open(config_path) as f:
        config = json.load(f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"),      "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "le_channel.pkl"),  "rb") as f:
        le_channel = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "le_category.pkl"), "rb") as f:
        le_category = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=["q_lin", "v_lin"], inference_mode=True,
    )
    model = DistilBERTLoRAWithMetadata(DISTILBERT_MODEL, lora_config).to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model_state_dict.pt"), map_location=device)
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    return model, tokenizer, scaler, le_channel, le_category, device, None


# ──────────────────────────────────────────────────────────────
# Preprocessing + inference
# ──────────────────────────────────────────────────────────────
def safe_encode(le, value):
    classes = list(le.classes_)
    if str(value) in classes:
        return int(le.transform([str(value)])[0])
    return 0


def preprocess_single(subject, description, channel, category, res_time,
                       priority, scaler, le_channel, le_category):
    clean_desc  = clean_ticket_text(description)
    ct          = clean_text(f"{subject} {clean_desc}")
    rule_sev, rule_ev = compute_rule_score(ct)

    res_norm      = float(scaler.transform([[res_time]])[0][0])
    ch_enc        = safe_encode(le_channel, channel)
    cat_enc       = safe_encode(le_category, category)
    priority_num  = PRIORITY_MAP.get(priority, 1)
    inferred      = rule_sev
    delta         = inferred - priority_num

    return {
        "clean_text":         ct,
        "rule_score":         rule_sev,
        "rule_evidence":      rule_ev,
        "resolution_score":   rule_sev,
        "resolution_time_norm": res_norm,
        "channel_encoded":    ch_enc,
        "category_encoded":   cat_enc,
        "inferred_severity":  inferred,
        "assigned_priority_num": priority_num,
        "severity_delta":     delta,
        "Priority_Level":     priority,
        "Ticket_Subject":     subject,
        "Ticket_Description": description,
        "Ticket_Channel":     channel,
        "Issue_Category":     category,
        "Resolution_Time_Hours": res_time,
    }


def predict_single(features, model, tokenizer, device):
    enc = tokenizer(
        features["clean_text"],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    channel_t  = torch.tensor([features["channel_encoded"]],   dtype=torch.float).to(device)
    category_t = torch.tensor([features["category_encoded"]],  dtype=torch.float).to(device)
    res_time_t = torch.tensor([features["resolution_score"]],  dtype=torch.float).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, channel_t, category_t, res_time_t)
        probs  = torch.softmax(logits, dim=1)

    pred = int(torch.argmax(probs, dim=1).item())
    prob = float(probs[0, 1].item())
    return pred, prob


def predict_batch(df_proc, model, tokenizer, device, batch_size=64):
    all_preds, all_probs = [], []
    texts      = df_proc["clean_text"].tolist()
    channels   = df_proc["channel_encoded"].tolist()
    categories = df_proc["category_encoded"].tolist()
    res_times  = df_proc["resolution_score"].tolist()

    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i: i + batch_size],
            max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                torch.tensor(channels[i: i + batch_size],   dtype=torch.float).to(device),
                torch.tensor(categories[i: i + batch_size], dtype=torch.float).to(device),
                torch.tensor(res_times[i: i + batch_size],  dtype=torch.float).to(device),
            )
            probs = torch.softmax(logits, dim=1)
        all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    return all_preds, all_probs


def preprocess_batch_df(df, scaler, le_channel, le_category):
    df = df.copy()
    df["clean_description"] = df["Ticket_Description"].apply(clean_ticket_text)
    df["clean_text"] = (
        df["Ticket_Subject"].fillna("") + " " + df["clean_description"].fillna("")
    ).apply(clean_text)
    df[["rule_score", "rule_evidence"]] = df["clean_text"].apply(
        lambda x: pd.Series(compute_rule_score(x))
    )
    df["resolution_score"] = df["rule_score"]

    if "Resolution_Time_Hours" not in df.columns:
        df["Resolution_Time_Hours"] = 0.0
    df["resolution_time_norm"] = scaler.transform(df[["Resolution_Time_Hours"]])

    df["channel_encoded"]  = df["Ticket_Channel"].astype(str).apply(
        lambda x: safe_encode(le_channel, x)
    )
    df["category_encoded"] = df["Issue_Category"].astype(str).apply(
        lambda x: safe_encode(le_category, x)
    )
    df["inferred_severity"]     = df["rule_score"].clip(0, 3)
    df["assigned_priority_num"] = df.get(
        "Priority_Level", pd.Series(["Low"] * len(df))
    ).apply(lambda p: PRIORITY_MAP.get(str(p), 1))
    df["severity_delta"] = df["inferred_severity"] - df["assigned_priority_num"]
    return df


# ──────────────────────────────────────────────────────────────
# Dossier builder
# ──────────────────────────────────────────────────────────────
def build_dossier(row: dict, pred: int, prob: float) -> dict:
    inferred    = int(row.get("inferred_severity", 0))
    delta       = int(row.get("severity_delta", 0))
    evidence    = row.get("rule_evidence", "") or "No keyword evidence"
    confidence  = min(0.99, round(0.50 + abs(delta) * 0.15, 2))
    res_time    = row.get("Resolution_Time_Hours", 0)
    mtype       = row.get("mismatch_type", "Consistent")
    assigned    = row.get("Priority_Level", "Unknown")
    inferred_lbl = SEVERITY_LABEL.get(inferred, str(inferred))
    subject     = row.get("Ticket_Subject", "")

    if mtype == "Hidden Crisis":
        analysis = (
            f"The ticket '{subject}' was assigned priority '{assigned}', but keyword signals "
            f"({evidence}) and a resolution time of {res_time} hours indicate a higher actual "
            f"severity of '{inferred_lbl}' (Δ={delta:+d}). "
            f"This is a Hidden Crisis — a critical issue is under-prioritised, "
            f"potentially violating SLA commitments."
        )
    elif mtype == "False Alarm":
        analysis = (
            f"The ticket '{subject}' carries priority '{assigned}', but content analysis "
            f"and resolution time of {res_time} hours suggest a lower severity of '{inferred_lbl}' (Δ={delta:+d}). "
            f"This is a False Alarm — resources are being over-allocated to a low-urgency issue."
        )
    else:
        analysis = (
            f"The assigned priority '{assigned}' aligns with the ticket content "
            f"and a resolution time of {res_time} hours. No significant mismatch detected."
        )

    return {
        "ticket_id":         str(row.get("Ticket_ID", "N/A")),
        "assigned_priority": assigned,
        "inferred_severity": inferred_lbl,
        "mismatch_type":     mtype,
        "severity_delta":    delta,
        "mismatch_probability": round(prob, 4),
        "feature_evidence": [
            {"signal": "subject",          "value": subject},
            {"signal": "keyword",          "value": evidence,
             "weight": "high" if any(w in str(row.get("clean_text", "")) for w in CRITICAL_WORDS) else "medium"},
            {"signal": "resolution_time",  "value": str(res_time),
             "interpretation": "High urgency" if float(res_time) < 12 else "Lower urgency"},
            {"signal": "ticket_channel",   "value": str(row.get("Ticket_Channel", "")), "weight": "low"},
            {"signal": "issue_category",   "value": str(row.get("Issue_Category", "")), "weight": "medium"},
        ],
        "constraint_analysis": analysis,
        "confidence": confidence,
    }


# ──────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────
def render_dossier_card(dossier: dict):
    mtype = dossier["mismatch_type"]
    color = MISMATCH_COLORS.get(mtype, "#95a5a6")

    badge_html = (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:20px;font-weight:bold;font-size:0.85rem;">{mtype}</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assigned Priority",  dossier["assigned_priority"])
    col2.metric("Inferred Severity",  dossier["inferred_severity"])
    col3.metric("Severity Delta",     f"{dossier['severity_delta']:+d}")
    col4.metric("Confidence",         f"{dossier['confidence']:.0%}")

    st.markdown("**Constraint Analysis**")
    st.info(dossier["constraint_analysis"])

    with st.expander("📋 Feature Evidence"):
        for ev in dossier["feature_evidence"]:
            cols = st.columns([1, 3])
            cols[0].markdown(f"**{ev['signal']}**")
            detail = ev.get("value", "")
            if "weight" in ev:
                detail += f"  *(weight: {ev['weight']})*"
            if "interpretation" in ev:
                detail += f"  → {ev['interpretation']}"
            cols[1].markdown(detail)

    with st.expander("🔧 Raw Dossier JSON"):
        st.json(dossier)


def render_dashboard(results_df: pd.DataFrame):
    st.markdown("---")
    st.header("📊 Priority Mismatch Dashboard")

    # ── KPI row ──────────────────────────────────────────────
    total      = len(results_df)
    n_mismatch = (results_df["predicted_mismatch"] == 1).sum()
    n_hidden   = (results_df["mismatch_type"] == "Hidden Crisis").sum()
    n_false    = (results_df["mismatch_type"] == "False Alarm").sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Tickets",     total)
    k2.metric("Mismatches Flagged", n_mismatch,
              delta=f"{100*n_mismatch/total:.1f}%", delta_color="inverse")
    k3.metric("Hidden Crises 🚨",  n_hidden)
    k4.metric("False Alarms 🔵",   n_false)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # ── Mismatch distribution pie ─────────────────────────────
    with col_a:
        st.subheader("Mismatch Distribution")
        dist_df = results_df["mismatch_type"].value_counts().reset_index()
        dist_df.columns = ["Type", "Count"]
        fig_pie = px.pie(
            dist_df, names="Type", values="Count",
            color="Type",
            color_discrete_map=MISMATCH_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Mismatch by priority bar ──────────────────────────────
    with col_b:
        st.subheader("Mismatch Count by Assigned Priority")
        if "Priority_Level" in results_df.columns:
            cross = (
                results_df.groupby(["Priority_Level", "mismatch_type"])
                .size().reset_index(name="Count")
            )
            fig_bar = px.bar(
                cross, x="Priority_Level", y="Count",
                color="mismatch_type",
                barmode="group",
                color_discrete_map=MISMATCH_COLORS,
                category_orders={"Priority_Level": ["Low", "Medium", "High", "Critical"]},
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Priority_Level column not found in input CSV.")

    # ── Severity delta heatmap ────────────────────────────────
    st.subheader("Severity Delta Heatmap (Category × Channel)")
    if "Issue_Category" in results_df.columns and "Ticket_Channel" in results_df.columns:
        heat_data = (
            results_df.groupby(["Issue_Category", "Ticket_Channel"])["severity_delta"]
            .mean()
            .reset_index()
        )
        heat_pivot = heat_data.pivot(
            index="Issue_Category", columns="Ticket_Channel", values="severity_delta"
        ).fillna(0)

        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.tolist(),
            y=heat_pivot.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Avg Δ Severity"),
            text=np.round(heat_pivot.values, 2),
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(
            xaxis_title="Ticket Channel",
            yaxis_title="Issue Category",
            height=420,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Issue_Category / Ticket_Channel columns not found.")

    # ── Top keyword signals ───────────────────────────────────
    st.subheader("Top Contributing Keyword Signals")
    if "rule_evidence" in results_df.columns:
        flagged_ev = results_df[results_df["predicted_mismatch"] == 1]["rule_evidence"].dropna()
        all_words  = [w for ev in flagged_ev for w in str(ev).split(",") if w.strip()]
        if all_words:
            freq_df = (
                pd.Series(all_words).value_counts().reset_index()
            )
            freq_df.columns = ["Signal", "Count"]
            fig_sig = px.bar(
                freq_df.head(15), x="Count", y="Signal",
                orientation="h",
                color="Count",
                color_continuous_scale="Reds",
            )
            fig_sig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig_sig, use_container_width=True)
        else:
            st.info("No keyword evidence found in flagged tickets.")

    # ── Mismatch probability distribution ─────────────────────
    if "mismatch_probability" in results_df.columns:
        st.subheader("Mismatch Probability Distribution")
        fig_hist = px.histogram(
            results_df, x="mismatch_probability",
            color="mismatch_type",
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map=MISMATCH_COLORS,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/inspection.png",
            width=72,
        )
        st.title("SIA")
        st.caption("Support Integrity Auditor")
        st.markdown("---")
        mode = st.radio(
            "Mode",
            ["🎫 Single Ticket", "📂 Batch CSV", "📊 Dashboard Only"],
            index=0,
        )
        st.markdown("---")
        st.markdown("**Model Info**")
        config_path = os.path.join(MODEL_DIR, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            st.markdown(f"- Backbone: `DistilBERT + LoRA`")
            st.markdown(f"- Test Accuracy: `{cfg.get('test_accuracy',0):.2%}`")
            st.markdown(f"- Test Macro F1: `{cfg.get('test_macro_f1',0):.4f}`")
            st.markdown(f"- Recall Class 0: `{cfg.get('test_recall_0',0):.4f}`")
            st.markdown(f"- Recall Class 1: `{cfg.get('test_recall_1',0):.4f}`")
        else:
            st.warning("Model not found.\nRun `train_pipeline.py` first.")
        st.markdown("---")
        st.markdown("**Threshold**")
        st.markdown("`|severity_delta| ≥ 2` → Mismatch")
    return mode


# ──────────────────────────────────────────────────────────────
# Single-ticket tab
# ──────────────────────────────────────────────────────────────
def single_ticket_tab(model, tokenizer, scaler, le_channel, le_category, device):
    st.header("🎫 Single Ticket Analysis")

    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticket_id   = st.text_input("Ticket ID",      value="TKT-001")
            subject     = st.text_input("Ticket Subject", value="Cannot login to account")
            priority    = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"], index=1)
            channel     = st.selectbox("Ticket Channel",
                                       ["Email", "Phone", "Chat", "Social media"], index=0)
        with col2:
            category    = st.selectbox("Issue Category",
                                       ["Technical support", "Billing inquiry",
                                        "Account access", "Product inquiry",
                                        "Cancellation request", "Refund request",
                                        "Shipping & delivery", "Other"], index=2)
            res_time    = st.number_input("Resolution Time (hours)", min_value=0.0,
                                          max_value=720.0, value=24.0, step=0.5)
            description = st.text_area("Ticket Description",
                                       value="I am unable to login to my account. "
                                             "I tried resetting my password but the link does not work. "
                                             "This is very urgent.",
                                       height=140)
        submitted = st.form_submit_button("🔍 Analyse Ticket", use_container_width=True)

    if submitted:
        if not subject.strip() or not description.strip():
            st.error("Please fill in at least the Subject and Description fields.")
            return

        with st.spinner("Analysing …"):
            features = preprocess_single(
                subject, description, channel, category, res_time,
                priority, scaler, le_channel, le_category,
            )
            pred, prob = predict_single(features, model, tokenizer, device)

        features["Ticket_ID"]   = ticket_id
        features["predicted_mismatch"]     = pred
        features["mismatch_probability"]   = prob

        delta = features["severity_delta"]
        if pred == 1 and delta >= 1:
            features["mismatch_type"] = "Hidden Crisis"
        elif pred == 1 and delta <= -1:
            features["mismatch_type"] = "False Alarm"
        else:
            features["mismatch_type"] = "Consistent"

        dossier = build_dossier(features, pred, prob)

        st.markdown("---")
        # Verdict banner
        mtype  = features["mismatch_type"]
        color  = MISMATCH_COLORS.get(mtype, "#95a5a6")
        verdict = "⚠️ MISMATCH DETECTED" if pred == 1 else "✅ CONSISTENT"

        st.markdown(
            f'<div style="background:{color};padding:16px 24px;border-radius:10px;'
            f'text-align:center;color:white;font-size:1.4rem;font-weight:bold;">'
            f'{verdict} — {mtype}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        render_dossier_card(dossier)

        # Download button
        st.download_button(
            "⬇️ Download Dossier (JSON)",
            data=json.dumps(dossier, indent=4),
            file_name=f"dossier_{ticket_id}.json",
            mime="application/json",
        )


# ──────────────────────────────────────────────────────────────
# Batch CSV tab
# ──────────────────────────────────────────────────────────────
def batch_tab(model, tokenizer, scaler, le_channel, le_category, device):
    st.header("📂 Batch CSV Analysis")

    st.markdown(
        "Upload a CSV with these columns: "
        "`Ticket_ID`, `Ticket_Subject`, `Ticket_Description`, "
        "`Priority_Level` *(optional)*, `Ticket_Channel`, `Issue_Category`, "
        "`Resolution_Time_Hours` *(optional)*"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Awaiting CSV upload …")
        return

    df_in = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df_in)} tickets")
    st.dataframe(df_in.head(), use_container_width=True)

    if st.button("🚀 Run Batch Inference", use_container_width=True):
        required = ["Ticket_ID", "Ticket_Subject", "Ticket_Description",
                    "Ticket_Channel", "Issue_Category"]
        missing  = [c for c in required if c not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        progress = st.progress(0, text="Pre-processing …")
        df_proc  = preprocess_batch_df(df_in, scaler, le_channel, le_category)
        progress.progress(30, text="Running model …")
        preds, probs = predict_batch(df_proc, model, tokenizer, device)
        progress.progress(70, text="Generating dossiers …")

        df_proc["predicted_mismatch"]   = preds
        df_proc["mismatch_probability"] = [round(p, 4) for p in probs]
        df_proc["mismatch_type"]        = "Consistent"
        df_proc.loc[
            (df_proc["predicted_mismatch"] == 1) & (df_proc["severity_delta"] >= 1),
            "mismatch_type"
        ] = "Hidden Crisis"
        df_proc.loc[
            (df_proc["predicted_mismatch"] == 1) & (df_proc["severity_delta"] <= -1),
            "mismatch_type"
        ] = "False Alarm"

        # Re-attach original columns
        for col in ["Priority_Level", "Ticket_Subject", "Ticket_Description",
                    "Ticket_Channel", "Issue_Category", "Resolution_Time_Hours"]:
            if col in df_in.columns:
                df_proc[col] = df_in[col].values

        dossiers = []
        flagged  = df_proc[df_proc["predicted_mismatch"] == 1]
        for idx, row in flagged.iterrows():
            d = build_dossier(row.to_dict(), int(row["predicted_mismatch"]),
                              float(row["mismatch_probability"]))
            dossiers.append(d)

        progress.progress(100, text="Done!")
        st.success(f"Analysis complete — {len(flagged)} mismatch(es) detected out of {len(df_proc)} tickets.")

        # Results table
        st.subheader("Results")
        display_cols = ["Ticket_ID", "Priority_Level", "predicted_mismatch",
                        "mismatch_probability", "mismatch_type", "severity_delta",
                        "rule_evidence"]
        display_cols = [c for c in display_cols if c in df_proc.columns]

        def color_row(row):
            c = MISMATCH_COLORS.get(row.get("mismatch_type", "Consistent"), "#ffffff")
            return [f"background-color:{c}20"] * len(row)

        st.dataframe(df_proc[display_cols].style.apply(color_row, axis=1),
                     use_container_width=True)

        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            csv_buf = df_proc[display_cols].to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv_buf,
                               "predictions.csv", "text/csv",
                               use_container_width=True)
        with col2:
            st.download_button("⬇️ Download Dossiers JSON",
                               json.dumps(dossiers, indent=4),
                               "dossiers.json", "application/json",
                               use_container_width=True)

        # Dashboard
        render_dashboard(df_proc)

        # Show individual dossiers
        if dossiers:
            st.markdown("---")
            st.subheader("📄 Evidence Dossiers (Flagged Tickets)")
            for dos in dossiers:
                with st.expander(
                    f"🎫 {dos['ticket_id']} — {dos['mismatch_type']}  "
                    f"[{dos['assigned_priority']} → {dos['inferred_severity']}]"
                ):
                    render_dossier_card(dos)


# ──────────────────────────────────────────────────────────────
# Dashboard-only tab (loads pseudo_labeled_dataset.csv)
# ──────────────────────────────────────────────────────────────
def dashboard_only_tab():
    st.header("📊 Dashboard (Existing Results)")
    uploaded = st.file_uploader(
        "Upload predictions CSV (output of predict.py or batch run above)",
        type=["csv"], key="dash_upload",
    )
    if uploaded:
        df_dash = pd.read_csv(uploaded)
        render_dashboard(df_dash)
    else:
        # Try to load pseudo_labeled_dataset.csv as fallback
        if os.path.exists("pseudo_labeled_dataset.csv"):
            df_dash = pd.read_csv("pseudo_labeled_dataset.csv")
            df_dash = df_dash.rename(columns={"mismatch": "predicted_mismatch"})
            if "mismatch_probability" not in df_dash.columns:
                df_dash["mismatch_probability"] = df_dash["predicted_mismatch"].astype(float)
            st.info("Showing dashboard from `pseudo_labeled_dataset.csv`")
            render_dashboard(df_dash)
        else:
            st.info("Upload a predictions CSV or run batch inference first.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    mode = sidebar()

    st.title("🔍 Support Integrity Auditor (SIA)")
    st.caption(
        "Semantics-driven, evidence-grounded automated auditor — "
        "detects Priority Mismatches in CRM support tickets."
    )

    model, tokenizer, scaler, le_channel, le_category, device, error = load_artifacts()

    if error:
        st.error(error)
        st.markdown(
            "**To get started:**\n"
            "```bash\n"
            "python train_pipeline.py --pseudo-data pseudo_labeled_dataset.csv\n"
            "```"
        )
        # Still allow dashboard-only mode
        if mode == "📊 Dashboard Only":
            dashboard_only_tab()
        return

    if mode == "🎫 Single Ticket":
        single_ticket_tab(model, tokenizer, scaler, le_channel, le_category, device)
    elif mode == "📂 Batch CSV":
        batch_tab(model, tokenizer, scaler, le_channel, le_category, device)
    elif mode == "📊 Dashboard Only":
        dashboard_only_tab()


if __name__ == "__main__":
    main()
