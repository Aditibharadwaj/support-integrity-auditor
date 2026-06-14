# Ticket Severity Auditor — Hidden Crisis / False Alarm

Detects support tickets whose **assigned priority does not match their real
severity**:

- **Hidden Crisis** — logged too low (e.g. a "quick question" that is actually a
  payment outage with a suspected breach).
- **False Alarm** — logged too high (e.g. an "URGENT CRITICAL" ticket that is
  just a profile email change).

The model is a **DistilBERT + LoRA** classifier with a 3-feature metadata head
(channel, category, normalized resolution time), trained on **pseudo-labels**
fused from four weak signals (LLM severity, embedding clusters, a resolution-time
regressor, and a keyword lexicon).

## Files

| File | Purpose |
|---|---|
| `predict.py` | Shared library: preprocessing, model architecture, rule scorer, `TicketPredictor`, and a CLI. Single source of truth for the model graph. |
| `train_pipeline.py` | Full offline pipeline that reproduces the notebook end to end and writes `saved_model/`. |
| `app.py` | Streamlit app for live ticket auditing (loads `saved_model/`). |
| `requirements.txt` | Pinned dependencies. |

## Install

```bash
pip install -r requirements.txt
```

## 1. Train (produces `saved_model/`)

Put `customer_support_tickets.csv` in the working directory.

```bash
# Full fidelity (needs a GPU; uses Mistral-7B for the LLM severity signal)
python train_pipeline.py --data customer_support_tickets.csv --use-llm

# Runs anywhere (CPU ok): substitutes a rule-based proxy for the LLM signal
python train_pipeline.py --data customer_support_tickets.csv
```

Each stage caches to disk (`feature_engineered.csv`, `mistral_scores.csv`,
`pseudo_labeled_dataset.csv`, `train/val/test.csv`) and is skipped on re-run.
Use `--force` to recompute. The trained model and the app do **not** require
Mistral.

Artifacts written to `saved_model/`:
`model.pt`, `tokenizer/`, `encoder_config/`, `channel_encoder.pkl`,
`category_encoder.pkl`, `resolution_scaler.pkl`, `results.csv`,
`training_history.csv`.

> Note: the existing `saved_model/` from your notebook already works with
> `predict.py` and `app.py`. The only thing it lacks is `encoder_config/`
> (saved by this pipeline for fully offline loading); without it, the base
> DistilBERT config is fetched once on first load.

## 2. Predict (CLI)

```bash
python predict.py \
  --subject "Quick question about my dashboard" \
  --description "Hi Support, our payment system is down and we suspect an account breach." \
  --channel "Email" --category "Technical issue" \
  --resolution-hours 96 --priority "Low"
```

Outputs structured JSON with the verdict, mismatch probability, evidence, and a
dossier. You can also pass a ticket as a file: `--ticket ticket.json`.

## 3. Deploy (Streamlit)

```bash
streamlit run app.py
```

The app has one-click **Hidden Crisis**, **False Alarm**, and **Consistent**
demo tickets for the walkthrough, plus free-form input for live adversarial
tickets, and a methodology panel explaining the pseudo-label strategy. Point it
at a different artifact folder with `MODEL_DIR=/path streamlit run app.py`.

## Design notes

- Inference tokenizes `clean_text` (cleaned subject + first description sentence)
  and feeds channel/category/resolution as separate numeric metadata — exactly
  what the notebook's `TicketDataset` trained on (the unused `[CHANNEL]/[CATEGORY]`
  text variant is intentionally not used).
- The trained model decides **mismatch vs consistent**. A deterministic
  keyword-vs-priority guard then assigns the **direction** (Hidden Crisis vs
  False Alarm) and catches obvious adversarial tickets on borderline scores.
- Unseen channels/categories are mapped to a safe default instead of raising, so
  the live app never crashes on arbitrary input.
