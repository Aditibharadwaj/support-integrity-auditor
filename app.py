"""Ticket Priority Auditor - Streamlit front end.

Three ways to use it:
  1. Single ticket  - fill a short form, get one verdict
  2. Batch CSV      - upload a tickets CSV, score every row
  3. Summary        - counts and a chart over everything scored this session

Every analysis produces five fields:
  Ticket_ID, prediction, confidence, mismatch_type, dossier_json

This file only READS from predict.py (TicketPredictor + PRIORITY_MAP).
It does not import or run train_pipeline.py. Point it at a trained model
folder via the MODEL_DIR environment variable (default: ./saved_model).
"""

import os
import json
import urllib.request
import urllib.error

import pandas as pd
import streamlit as st

from predict import TicketPredictor, PRIORITY_MAP, PRIORITY_INVERSE

MODEL_DIR = os.environ.get("MODEL_DIR", "saved_model")
OUTPUT_COLS = ["Ticket_ID", "prediction", "confidence", "mismatch_type", "dossier_json"]

# Generative dossier configuration (all overridable by environment variable).
#   DOSSIER_BACKEND: mistral_api | local_mistral | template
#   mistral_api    -> fast, needs MISTRAL_API_KEY, recommended for CPU deploys
#   local_mistral  -> runs Mistral-7B locally (faithful to the notebook, needs a GPU)
#   template       -> the deterministic predict.py dossier, no LLM
DEFAULT_BACKEND = os.environ.get("DOSSIER_BACKEND", "mistral_api")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_API_MODEL = os.environ.get("MISTRAL_API_MODEL", "mistral-small-latest")
LOCAL_MISTRAL_MODEL = os.environ.get("LOCAL_MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
BACKEND_LABELS = {
    "mistral_api": "Mistral API (generative)",
    "local_mistral": "Local Mistral 7B (generative, slow)",
    "template": "Template (instant, no LLM)",
}

st.set_page_config(page_title="Ticket Priority Auditor", layout="wide")

# Fall back to .streamlit/secrets.toml for any config not set in the environment,
# so the API key can be set once in a file instead of exported every session.
def _from_secrets(name):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001  (no secrets.toml present, etc.)
        pass
    return ""


if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = _from_secrets("MISTRAL_API_KEY")
if os.environ.get("DOSSIER_BACKEND") is None and _from_secrets("DOSSIER_BACKEND"):
    DEFAULT_BACKEND = _from_secrets("DOSSIER_BACKEND")
if os.environ.get("MISTRAL_API_MODEL") is None and _from_secrets("MISTRAL_API_MODEL"):
    MISTRAL_API_MODEL = _from_secrets("MISTRAL_API_MODEL")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.5rem; max-width: 1100px;}
      h1 {font-weight: 700; letter-spacing: -0.5px;}
      .stTabs [data-baseweb="tab"] {font-size: 0.95rem;}
      .legend {color: #8b949e; font-size: 0.85rem; line-height: 1.6;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model loading (cached so it loads once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model and encoders...")
def load_predictor(model_dir):
    return TicketPredictor(model_dir=model_dir)


try:
    predictor = load_predictor(MODEL_DIR)
except Exception as exc:  # noqa: BLE001
    st.title("Ticket Priority Auditor")
    st.error(
        f"Could not load a model from `{MODEL_DIR}`.\n\n"
        f"```\n{exc}\n```\n\n"
        "Make sure that folder has model.pt, tokenizer/, and the three "
        ".pkl encoders, or set the MODEL_DIR environment variable to the "
        "folder that does."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Dropdown options come from the model's own encoders, so they always match
# what it was trained on. Fall back to sensible defaults if unavailable.
# --------------------------------------------------------------------------- #
def encoder_options(encoder, fallback):
    try:
        return [str(c) for c in encoder.classes_]
    except Exception:  # noqa: BLE001
        return fallback


CATEGORY_OPTS = encoder_options(predictor.category_encoder, ["Billing", "Technical issue", "Account access"])
CHANNEL_OPTS = encoder_options(predictor.channel_encoder, ["Email", "Chat", "Phone", "Social media"])
PRIORITY_OPTS = list(PRIORITY_MAP.keys())


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def reconcile_verdict(result):
    """Re-derive verdict + mismatch so a direction is only reported when the
    keyword-vs-priority gap actually supports it.

    predict.py marks a mismatch on the model flag alone and, when the keyword
    delta is 0, guesses a direction from resolution time. That can produce
    impossible labels (e.g. "False Alarm" on a Low-priority ticket, which has
    nothing lower to be over-prioritized against). Here a mismatch needs the
    rule delta to point a direction; the model flag refines within it, and a
    large gap (|delta| >= 2) still triggers on its own."""
    rule = result["rule"]
    delta = rule["severity_delta"]
    model_flag = result["model"]["label"] == "Mismatch"

    if delta > 0 and (model_flag or delta >= 2):
        verdict, is_mismatch = "Hidden Crisis", True
    elif delta < 0 and (model_flag or delta <= -2):
        verdict, is_mismatch = "False Alarm", True
    else:
        verdict, is_mismatch = "Consistent", False

    # Keep the reported confidence aligned with the reconciled decision.
    model = result["model"]
    if is_mismatch:
        model["confidence"] = round(float(model.get("mismatch_probability", model["confidence"])), 4)
    else:
        model["confidence"] = round(float(model.get("consistent_probability", model["confidence"])), 4)

    result["verdict"] = verdict
    result["is_mismatch"] = is_mismatch
    return result


def build_dossier_schema(ticket_id, result):
    """Assemble the structured dossier. All data fields come from the model
    result (always correct); constraint_analysis is the one generative field
    (LLM-written when a backend is active, template text otherwise)."""
    rule = result["rule"]
    inp = result["input"]
    evidence = rule["evidence"]
    res_norm = float(rule.get("resolution_time_norm", 0.0))
    res_hours = inp.get("resolution_hours", 0.0)

    if res_norm >= 0.5:
        res_interp = "above-median handling time, which leans under-prioritized"
    else:
        res_interp = "below-median handling time, consistent with lower urgency"

    return {
        "ticket_id": ticket_id,
        "assigned_priority": rule["assigned_priority"],
        "inferred_severity": PRIORITY_INVERSE.get(
            rule["inferred_severity"], str(rule["inferred_severity"])
        ),
        "mismatch_type": result["verdict"],
        "severity_delta": rule["severity_delta"],
        "feature_evidence": [
            {
                "signal": "keyword",
                "value": ", ".join(evidence) if evidence else "none",
                "weight": rule["inferred_severity"],
            },
            {
                "signal": "resolution_time",
                "value": res_hours,
                "interpretation": res_interp,
            },
        ],
        "constraint_analysis": result.get("analysis", result.get("dossier", "")),
        "confidence": round(float(result["model"]["confidence"]), 4),
    }


def result_to_row(ticket_id, result):
    """Collapse a predictor result into the five output fields."""
    schema = build_dossier_schema(ticket_id, result)
    return {
        "Ticket_ID": ticket_id,
        "prediction": "Mismatch" if result["is_mismatch"] else "Consistent",
        "confidence": round(float(result["model"]["confidence"]), 4),
        "mismatch_type": result["verdict"],
        "dossier_json": json.dumps(schema, ensure_ascii=False),
    }


def store_rows(rows):
    """Append rows to the session-wide results table (drives the Summary tab)."""
    new = pd.DataFrame(rows, columns=OUTPUT_COLS)
    if "results" not in st.session_state:
        st.session_state["results"] = new
    else:
        st.session_state["results"] = pd.concat(
            [st.session_state["results"], new], ignore_index=True
        )


def verdict_banner(verdict):
    if verdict == "Hidden Crisis":
        st.error("Hidden Crisis - ticket looks under-prioritized.")
    elif verdict == "False Alarm":
        st.warning("False Alarm - ticket looks over-prioritized.")
    else:
        st.success("Consistent - assigned priority matches the content.")


def find_col(df, candidates):
    """Return the first column in df that matches any candidate (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# --------------------------------------------------------------------------- #
# Generative dossier (added in app.py, predict.py is untouched)
# --------------------------------------------------------------------------- #
def _dossier_prompt(result):
    inp, rule = result["input"], result["rule"]
    evidence = ", ".join(rule["evidence"]) if rule["evidence"] else "none"
    return (
        f"Ticket subject: {inp['subject']}\n"
        f"Description: {inp['description']}\n"
        f"Assigned priority: {rule['assigned_priority']}\n"
        f"Model verdict: {result['verdict']} (mismatch={result['is_mismatch']})\n"
        f"Inferred severity 0-3: {rule['inferred_severity']}; "
        f"assigned 0-3: {rule['assigned_priority_num']}; gap: {rule['severity_delta']}\n"
        f"Severity keywords found: {evidence}\n"
        f"Resolution time (hours): {inp['resolution_hours']}\n\n"
        "Write a 2-3 sentence analyst assessment: state whether the ticket is "
        "correctly prioritized, why (cite the content signals above), and the "
        "recommended action. Plain prose, no greeting, no lists."
    )


def _mistral_api_dossier(result):
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    payload = json.dumps({
        "model": MISTRAL_API_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise support operations analyst."},
            {"role": "user", "content": _dossier_prompt(result)},
        ],
        "temperature": 0.3,
        "max_tokens": 220,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


@st.cache_resource(show_spinner="Loading local Mistral 7B (first load is slow)...")
def _load_local_mistral(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tok, model


def _local_mistral_dossier(result):
    import torch

    tok, model = _load_local_mistral(LOCAL_MISTRAL_MODEL)
    messages = [{"role": "user", "content": _dossier_prompt(result)}]
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=220, temperature=0.3,
            do_sample=True, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def generate_dossier(result, backend):
    """Return (analysis_text, source). Falls back to the template on any failure
    so the app never breaks. 'source' records what actually produced the text."""
    if backend == "template":
        return result["dossier"], "template"
    try:
        if backend == "mistral_api":
            return _mistral_api_dossier(result), "mistral-api"
        if backend == "local_mistral":
            return _local_mistral_dossier(result), "local-mistral"
    except Exception as exc:  # noqa: BLE001
        return (
            result["dossier"] + f"\n\n(Generative analysis unavailable: {exc}. Showing template.)",
            "template-fallback",
        )
    return result["dossier"], "template"


def attach_analysis(result, backend):
    """Add the generative analysis to the result so it flows into dossier_json."""
    text, source = generate_dossier(result, backend)
    result["analysis"] = text
    result["analysis_source"] = source
    return result


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Ticket Priority Auditor")
st.markdown(
    "<div class='legend'>Checks whether a ticket's assigned priority matches "
    "what its content suggests. <b>Hidden Crisis</b> = under-prioritized, "
    "<b>False Alarm</b> = over-prioritized, <b>Consistent</b> = aligned.</div>",
    unsafe_allow_html=True,
)
st.write("")

# Sidebar: choose how the dossier analysis is produced.
with st.sidebar:
    st.subheader("Analysis mode")
    options = list(BACKEND_LABELS.keys())
    default_idx = options.index(DEFAULT_BACKEND) if DEFAULT_BACKEND in options else 0
    backend = st.radio(
        "Dossier generation",
        options,
        index=default_idx,
        format_func=lambda x: BACKEND_LABELS[x],
    )
    if backend == "mistral_api":
        if MISTRAL_API_KEY:
            st.caption(f"Using Mistral API ({MISTRAL_API_MODEL}).")
        else:
            st.caption("MISTRAL_API_KEY not set, so analysis falls back to the template.")
    elif backend == "local_mistral":
        st.caption("Loads Mistral 7B locally. Practical only on a GPU.")
    else:
        st.caption("Deterministic template. No LLM, instant.")

tab_single, tab_batch, tab_summary = st.tabs(["Single ticket", "Batch CSV", "Summary"])


# --------------------------------------------------------------------------- #
# Tab 1: single ticket
# --------------------------------------------------------------------------- #
with tab_single:
    with st.form("single_ticket"):
        c1, c2 = st.columns(2)
        with c1:
            ticket_id = st.text_input("Ticket ID (optional)", value="")
            category = st.selectbox("Issue category", CATEGORY_OPTS)
            priority = st.selectbox("Assigned priority", PRIORITY_OPTS)
        with c2:
            channel = st.selectbox("Ticket channel", CHANNEL_OPTS)
            resolution_hours = st.number_input(
                "Resolution time (hours)", min_value=0.0, value=24.0, step=1.0
            )

        subject = st.text_input("Ticket subject", value="")
        description = st.text_area("Ticket description", value="", height=140)
        submitted = st.form_submit_button("Analyze ticket", type="primary")

    if submitted:
        if not subject.strip() and not description.strip():
            st.info("Enter a subject or description first.")
        else:
            ticket = {
                "subject": subject,
                "description": description,
                "channel": channel,
                "category": category,
                "resolution_hours": resolution_hours,
                "priority": priority,
            }
            with st.spinner("Analyzing..."):
                result = predictor.predict(ticket)
                result = reconcile_verdict(result)
                result = attach_analysis(result, backend)
            tid = ticket_id.strip() or "MANUAL"
            row = result_to_row(tid, result)
            store_rows([row])

            verdict_banner(result["verdict"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ticket ID", tid)
            m2.metric("Prediction", row["prediction"])
            m3.metric("Confidence", f"{row['confidence'] * 100:.0f}%")
            m4.metric("Mismatch type", row["mismatch_type"])

            st.markdown("**Analysis**")
            st.write(result["analysis"])
            st.caption(f"Source: {result['analysis_source']}")

            with st.expander("dossier_json"):
                st.json(build_dossier_schema(tid, result))


# --------------------------------------------------------------------------- #
# Tab 2: batch CSV
# --------------------------------------------------------------------------- #
with tab_batch:
    st.write(
        "Upload a CSV of tickets. Recognized columns: Ticket_ID, "
        "Ticket_Subject, Ticket_Description, Issue_Category, Ticket_Channel, "
        "Priority_Level, Resolution_Time_Hours."
    )
    uploaded = st.file_uploader("Tickets CSV", type=["csv"])
    max_rows = st.number_input(
        "Rows to score (from the top)", min_value=1, value=200, step=50,
        help="Scoring runs on CPU, so a cap keeps it responsive. Raise it for full runs.",
    )
    gen_batch = st.checkbox(
        "Generate AI analysis for every row",
        value=False,
        help="Off uses the instant template. On calls the selected model per row, which is much slower.",
    )

    if uploaded is not None and st.button("Score CSV", type="primary"):
        df_in = pd.read_csv(uploaded)

        col_id = find_col(df_in, ["Ticket_ID", "id"])
        col_subject = find_col(df_in, ["Ticket_Subject", "subject"])
        col_desc = find_col(df_in, ["Ticket_Description", "description"])
        col_channel = find_col(df_in, ["Ticket_Channel", "channel"])
        col_category = find_col(df_in, ["Issue_Category", "category", "Ticket_Type"])
        col_priority = find_col(df_in, ["Priority_Level", "Ticket_Priority", "priority", "Assigned_Priority"])
        col_res = find_col(df_in, ["Resolution_Time_Hours", "Resolution_Hours", "resolution_hours"])

        if col_subject is None and col_desc is None:
            st.error("No subject or description column found. Check the file headers.")
        else:
            df_use = df_in.head(int(max_rows))
            rows, errors = [], 0
            progress = st.progress(0.0, text="Scoring tickets...")
            total = len(df_use)

            for i, (_, r) in enumerate(df_use.iterrows()):
                try:
                    ticket = {
                        "subject": str(r[col_subject]) if col_subject else "",
                        "description": str(r[col_desc]) if col_desc else "",
                        "channel": str(r[col_channel]) if col_channel else "",
                        "category": str(r[col_category]) if col_category else "",
                        "resolution_hours": r[col_res] if col_res else 0.0,
                        "priority": str(r[col_priority]) if col_priority else "Low",
                    }
                    result = predictor.predict(ticket)
                    result = reconcile_verdict(result)
                    row_backend = backend if gen_batch else "template"
                    result = attach_analysis(result, row_backend)
                    tid = str(r[col_id]) if col_id else f"ROW-{i + 1}"
                    rows.append(result_to_row(tid, result))
                except Exception:  # noqa: BLE001
                    errors += 1
                progress.progress((i + 1) / total, text=f"Scored {i + 1} of {total}")

            progress.empty()

            if rows:
                out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
                store_rows(rows)
                st.success(f"Scored {len(rows)} tickets" + (f", skipped {errors}." if errors else "."))
                st.dataframe(out_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results CSV",
                    out_df.to_csv(index=False).encode("utf-8"),
                    file_name="ticket_audit_results.csv",
                    mime="text/csv",
                )
            else:
                st.error("No rows could be scored.")


# --------------------------------------------------------------------------- #
# Tab 3: summary over everything scored this session
# --------------------------------------------------------------------------- #
with tab_summary:
    results = st.session_state.get("results")
    if results is None or results.empty:
        st.info("Score a ticket or a CSV first, then this tab fills in.")
    else:
        total = len(results)
        n_mismatch = int((results["prediction"] == "Mismatch").sum())
        n_hidden = int((results["mismatch_type"] == "Hidden Crisis").sum())
        n_false = int((results["mismatch_type"] == "False Alarm").sum())
        n_consistent = int((results["mismatch_type"] == "Consistent").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scored", total)
        m2.metric("Mismatches", n_mismatch)
        m3.metric("Hidden Crisis", n_hidden)
        m4.metric("False Alarm", n_false)

        st.write("")
        counts = (
            results["mismatch_type"]
            .value_counts()
            .reindex(["Hidden Crisis", "False Alarm", "Consistent"])
            .fillna(0)
        )
        st.bar_chart(counts)

        only_mismatch = st.checkbox("Show mismatches only", value=True)
        view = results[results["prediction"] == "Mismatch"] if only_mismatch else results
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.download_button(
            "Download all results CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="ticket_audit_session.csv",
            mime="text/csv",
        )
        if st.button("Clear results"):
            st.session_state.pop("results", None)
            st.rerun()

