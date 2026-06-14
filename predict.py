"""
predict.py — Support Integrity Auditor (SIA)
Inference script: loads trained model, runs predictions, generates Evidence Dossiers.

Usage:
  # Batch CSV inference:
  python predict.py --input tickets.csv --output predictions.csv

  # With Mistral-generated constraint analysis (requires GPU + mistral_scores for context):
  python predict.py --input tickets.csv --output predictions.csv --use-llm

  # Point to a different saved_model directory:
  python predict.py --input tickets.csv --model-dir saved_model --output predictions.csv
"""

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig

# ──────────────────────────────────────────────────────────────
# Constants (must match train_pipeline.py / notebook exactly)
# ──────────────────────────────────────────────────────────────
DISTILBERT_MODEL = "distilbert-base-uncased"
MAX_LEN          = 128
BATCH_SIZE       = 64
LORA_R           = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.1
DROPOUT          = 0.3
N_METADATA       = 3
N_CLASSES        = 2

PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
INV_PRIORITY = {v: k for k, v in PRIORITY_MAP.items()}

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

SEVERITY_LABEL = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


# ──────────────────────────────────────────────────────────────
# Text helpers  (identical to notebook)
# ──────────────────────────────────────────────────────────────
def clean_ticket_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"^Hi Support,\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"[?.!]", text)[0]
    return text.strip()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# Rule-based scoring  (identical to notebook)
# ──────────────────────────────────────────────────────────────
def compute_rule_score(text: str):
    score, evidence = 0, []
    for word in CRITICAL_WORDS:
        if word in text:
            score += 3
            evidence.append(word)
    for word in HIGH_WORDS:
        if word in text:
            score += 2
            evidence.append(word)
    if score >= 6:
        sev = 3
    elif score >= 4:
        sev = 2
    elif score >= 2:
        sev = 1
    else:
        sev = 0
    return sev, ",".join(evidence)


# ──────────────────────────────────────────────────────────────
# Model  (identical to train_pipeline.py / notebook)
# ──────────────────────────────────────────────────────────────
class DistilBERTLoRAWithMetadata(nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_config: LoraConfig,
        n_metadata: int = N_METADATA,
        n_classes: int  = N_CLASSES,
        dropout: float  = DROPOUT,
    ):
        super().__init__()
        base_model   = AutoModel.from_pretrained(model_name)
        self.encoder = get_peft_model(base_model, lora_config)
        hidden_size  = base_model.config.hidden_size

        self.meta_proj = nn.Sequential(
            nn.Linear(n_metadata, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
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
# Artifact loader
# ──────────────────────────────────────────────────────────────
def load_artifacts(model_dir: str, device: torch.device):
    """Load model, scaler, label encoders from saved_model directory."""
    config_path = os.path.join(model_dir, "training_config.json")
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: {config_path} not found. Run train_pipeline.py first.")

    with open(config_path) as f:
        config = json.load(f)

    with open(os.path.join(model_dir, "scaler.pkl"),     "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "le_channel.pkl"), "rb") as f:
        le_channel = pickle.load(f)
    with open(os.path.join(model_dir, "le_category.pkl"), "rb") as f:
        le_category = pickle.load(f)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_lin", "v_lin"],
        inference_mode=True,
    )
    model = DistilBERTLoRAWithMetadata(DISTILBERT_MODEL, lora_config).to(device)
    state_dict_path = os.path.join(model_dir, "best_model_state_dict.pt")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)

    return model, tokenizer, scaler, le_channel, le_category, config


# ──────────────────────────────────────────────────────────────
# Preprocessing for inference
# ──────────────────────────────────────────────────────────────
def preprocess_input(df: pd.DataFrame, scaler, le_channel, le_category) -> pd.DataFrame:
    df = df.copy()

    # Text
    df["clean_description"] = df["Ticket_Description"].apply(clean_ticket_text)
    df["clean_text"] = (
        df["Ticket_Subject"].fillna("") + " " + df["clean_description"].fillna("")
    ).apply(clean_text)

    # Rule score + evidence
    df[["rule_score", "rule_evidence"]] = df["clean_text"].apply(
        lambda x: pd.Series(compute_rule_score(x))
    )

    # Resolution score (use rule score as proxy for inference if no RF model)
    df["resolution_score"] = df["rule_score"]

    # Resolution time normalisation
    if "Resolution_Time_Hours" not in df.columns:
        df["Resolution_Time_Hours"] = 0.0
    df["resolution_time_norm"] = scaler.transform(df[["Resolution_Time_Hours"]])

    # Channel encoding — unseen labels fall back to 0
    channel_classes = list(le_channel.classes_)
    df["channel_encoded"] = df["Ticket_Channel"].astype(str).apply(
        lambda x: le_channel.transform([x])[0]
        if x in channel_classes else 0
    )

    # Category encoding — unseen labels fall back to 0
    category_classes = list(le_category.classes_)
    df["category_encoded"] = df["Issue_Category"].astype(str).apply(
        lambda x: le_category.transform([x])[0]
        if x in category_classes else 0
    )

    # Inferred severity from rule alone (for dossier delta before classifier)
    df["inferred_severity"] = df["rule_score"].clip(0, 3)

    # Priority numeric
    df["assigned_priority_num"] = df.get("Priority_Level", pd.Series(["Low"] * len(df))).map(
        lambda p: PRIORITY_MAP.get(str(p), 1)
    )
    df["severity_delta"] = df["inferred_severity"] - df["assigned_priority_num"]

    return df


# ──────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────
def run_inference(
    df: pd.DataFrame,
    model: nn.Module,
    tokenizer,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Returns (predictions list, confidence scores list)."""
    all_preds, all_probs = [], []
    texts      = df["clean_text"].tolist()
    channels   = df["channel_encoded"].tolist()
    categories = df["category_encoded"].tolist()
    res_times  = df["resolution_score"].tolist()

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i: i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        channel_t  = torch.tensor(channels[i: i + BATCH_SIZE],   dtype=torch.float).to(device)
        category_t = torch.tensor(categories[i: i + BATCH_SIZE], dtype=torch.float).to(device)
        res_time_t = torch.tensor(res_times[i: i + BATCH_SIZE],  dtype=torch.float).to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, channel_t, category_t, res_time_t)
            probs  = torch.softmax(logits, dim=1)

        all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())   # prob of mismatch

    return all_preds, all_probs


# ──────────────────────────────────────────────────────────────
# LLM constraint analysis (optional — requires Mistral)
# ──────────────────────────────────────────────────────────────
def load_mistral():
    """Load Mistral-7B-Instruct for dossier constraint_analysis. Returns (model, tokenizer, pipeline)."""
    from transformers import AutoTokenizer as AT, AutoModelForCausalLM as ACM, pipeline
    MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tok = AT.from_pretrained(MODEL)
    mdl = ACM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
    gen = pipeline("text-generation", model=mdl, tokenizer=tok)
    return gen


def mistral_constraint_analysis(row: dict, generator) -> str:
    prompt = f"""
You are a support operations analyst.

Ticket Subject: {row['Ticket_Subject']}
Ticket Description: {row['Ticket_Description']}
Assigned Priority: {row['Priority_Level']}
Inferred Severity: {row['inferred_severity']}
Mismatch Type: {row['mismatch_type']}
Keyword Evidence: {row['rule_evidence']}
Resolution Time: {row['Resolution_Time_Hours']} hours

Explain:
1. Why the assigned priority may not reflect the ticket urgency.
2. Which evidence supports the inferred severity.
3. Why the ticket is classified as {row['mismatch_type']}.

Use only the provided information. Do not mention scores. Do not invent information. Maximum 3 sentences.
"""
    response = generator(prompt, max_new_tokens=120, do_sample=False, temperature=0.0)
    generated = response[0]["generated_text"]
    return generated.replace(prompt, "").strip()


# ──────────────────────────────────────────────────────────────
# Template-based constraint analysis (no LLM required)
# ──────────────────────────────────────────────────────────────
def template_constraint_analysis(row: dict) -> str:
    mtype      = row.get("mismatch_type", "Consistent")
    assigned   = row.get("Priority_Level", "Unknown")
    inferred   = SEVERITY_LABEL.get(int(row.get("inferred_severity", 0)), "Unknown")
    delta      = int(row.get("severity_delta", 0))
    evidence   = row.get("rule_evidence", "") or "No keyword evidence"
    res_time   = row.get("Resolution_Time_Hours", "N/A")
    subject    = row.get("Ticket_Subject", "")

    if mtype == "Hidden Crisis":
        return (
            f"The ticket '{subject}' was assigned priority '{assigned}', but keyword signals "
            f"({evidence}) and a resolution time of {res_time} hours indicate a higher actual "
            f"severity of '{inferred}' (delta={delta}). "
            f"This constitutes a Hidden Crisis where a critical issue is under-prioritised, "
            f"potentially violating SLA commitments and risking customer churn."
        )
    elif mtype == "False Alarm":
        return (
            f"The ticket '{subject}' was assigned priority '{assigned}', however the text content "
            f"and a resolution time of {res_time} hours suggest a lower actual severity of '{inferred}' "
            f"(delta={delta}). "
            f"This is a False Alarm — resources are over-allocated to a low-urgency issue."
        )
    else:
        return (
            f"The assigned priority '{assigned}' is consistent with the ticket content. "
            f"No significant mismatch signal was detected."
        )


# ──────────────────────────────────────────────────────────────
# Dossier generation  (matches schema from problem statement)
# ──────────────────────────────────────────────────────────────
def build_dossier(row: dict, use_llm: bool = False, generator=None) -> dict:
    inferred   = int(row.get("inferred_severity", 0))
    delta      = int(row.get("severity_delta", 0))
    evidence   = row.get("rule_evidence", "") or "No keyword evidence"
    confidence = min(0.99, round(0.50 + abs(delta) * 0.15, 2))

    if use_llm and generator is not None:
        try:
            analysis = mistral_constraint_analysis(row, generator)
        except Exception:
            analysis = template_constraint_analysis(row)
    else:
        analysis = template_constraint_analysis(row)

    return {
        "ticket_id":         str(row.get("Ticket_ID", "")),
        "assigned_priority": str(row.get("Priority_Level", "")),
        "inferred_severity": SEVERITY_LABEL.get(inferred, str(inferred)),
        "mismatch_type":     str(row.get("mismatch_type", "Consistent")),
        "severity_delta":    delta,
        "feature_evidence":  [
            {
                "signal": "subject",
                "value":  str(row.get("Ticket_Subject", "")),
            },
            {
                "signal": "keyword",
                "value":  evidence,
                "weight": "high" if any(w in str(row.get("clean_text", "")) for w in CRITICAL_WORDS) else "medium",
            },
            {
                "signal":         "resolution_time",
                "value":          str(row.get("Resolution_Time_Hours", "N/A")),
                "interpretation": (
                    "High urgency (fast resolution expected)"
                    if float(row.get("Resolution_Time_Hours", 0)) < 12
                    else "Lower urgency (slow resolution acceptable)"
                ),
            },
            {
                "signal": "ticket_channel",
                "value":  str(row.get("Ticket_Channel", "")),
                "weight": "low",
            },
            {
                "signal": "issue_category",
                "value":  str(row.get("Issue_Category", "")),
                "weight": "medium",
            },
        ],
        "constraint_analysis": analysis,
        "confidence":          confidence,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    parser.add_argument("--input",     required=True,  help="Input CSV path")
    parser.add_argument("--output",    default="predictions.csv", help="Output CSV path")
    parser.add_argument("--dossier",   default="dossiers.json",   help="Output dossier JSON path")
    parser.add_argument("--model-dir", default="saved_model",     help="Directory with saved artifacts")
    parser.add_argument("--use-llm",   action="store_true",
                        help="Use Mistral-7B for constraint_analysis (requires GPU)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: Input file {args.input} not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load artifacts
    print(f"\nLoading model artifacts from {args.model_dir} …")
    model, tokenizer, scaler, le_channel, le_category, config = load_artifacts(
        args.model_dir, device
    )
    print("  ✅ Model loaded")

    # Load + preprocess input
    print(f"\nLoading input: {args.input} …")
    df_in = pd.read_csv(args.input)
    print(f"  {len(df_in)} tickets loaded")

    required = ["Ticket_ID", "Ticket_Subject", "Ticket_Description",
                "Ticket_Channel", "Issue_Category"]
    missing  = [c for c in required if c not in df_in.columns]
    if missing:
        sys.exit(f"ERROR: Input CSV missing columns: {missing}")

    df = preprocess_input(df_in, scaler, le_channel, le_category)

    # Inference
    print("\nRunning inference …")
    preds, probs = run_inference(df, model, tokenizer, device)

    df["predicted_mismatch"] = preds
    df["mismatch_probability"] = [round(p, 4) for p in probs]
    df["mismatch_type"] = "Consistent"
    df.loc[(df["predicted_mismatch"] == 1) & (df["severity_delta"] >= 1), "mismatch_type"] = "Hidden Crisis"
    df.loc[(df["predicted_mismatch"] == 1) & (df["severity_delta"] <= -1), "mismatch_type"] = "False Alarm"

    # Load Mistral if requested
    generator = None
    if args.use_llm:
        print("\nLoading Mistral-7B for constraint analysis (this may take a while) …")
        try:
            generator = load_mistral()
            print("  ✅ Mistral loaded")
        except Exception as e:
            print(f"  ❌ Mistral load failed ({e}). Falling back to template analysis.")

    # Generate dossiers for mismatched tickets
    flagged = df[df["predicted_mismatch"] == 1].copy()
    print(f"\nGenerating dossiers for {len(flagged)} flagged tickets …")

    dossiers = []
    for idx, row in flagged.iterrows():
        row_dict = row.to_dict()
        row_dict["Priority_Level"] = df_in.loc[idx, "Priority_Level"] if "Priority_Level" in df_in.columns else "Unknown"
        row_dict["Ticket_Subject"] = df_in.loc[idx, "Ticket_Subject"]
        row_dict["Ticket_Description"] = df_in.loc[idx, "Ticket_Description"]
        dossiers.append(build_dossier(row_dict, use_llm=args.use_llm, generator=generator))

    # Save predictions CSV
    output_cols = [
        "Ticket_ID", "predicted_mismatch", "mismatch_probability", "mismatch_type",
        "inferred_severity", "severity_delta", "rule_evidence",
    ]
    # Add Priority_Level if it was in the input
    if "Priority_Level" in df_in.columns:
        df["Priority_Level"] = df_in["Priority_Level"].values
        output_cols.insert(2, "Priority_Level")

    df[output_cols].to_csv(args.output, index=False)
    print(f"✅ Predictions saved → {args.output}")

    # Save dossiers JSON
    with open(args.dossier, "w") as f:
        json.dump(dossiers, f, indent=4)
    print(f"✅ Dossiers saved    → {args.dossier}")

    # Print summary
    n_mismatch   = (df["predicted_mismatch"] == 1).sum()
    n_hidden     = (df["mismatch_type"] == "Hidden Crisis").sum()
    n_false_alarm = (df["mismatch_type"] == "False Alarm").sum()
    print(f"\n{'='*45}")
    print(f"  Total tickets     : {len(df)}")
    print(f"  Mismatches flagged: {n_mismatch} ({100*n_mismatch/len(df):.1f}%)")
    print(f"    ↳ Hidden Crisis : {n_hidden}")
    print(f"    ↳ False Alarm   : {n_false_alarm}")
    print(f"{'='*45}")

    # Print sample dossier
    if dossiers:
        print("\n--- Sample Dossier (first flagged ticket) ---")
        print(json.dumps(dossiers[0], indent=4))


if __name__ == "__main__":
    main()
