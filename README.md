# Support Integrity Auditor (SIA)

> A semantics-driven, evidence-grounded auditing system that detects priority mismatches in enterprise CRM support tickets — without relying on manually assigned labels.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Live Application](#2-live-application)
3. [Dataset & Preprocessing](#3-dataset--preprocessing)
4. [Pseudo-Label Generation](#4-pseudo-label-generation)
5. [Classifier Training](#5-classifier-training)
6. [Evidence Dossier Generation](#6-evidence-dossier-generation)
7. [Ablation Study](#7-ablation-study)
8. [Results](#8-results)

---

## 1. Introduction

In enterprise-scale CRM ecosystems, manual ticket triage is riddled with **agent fatigue bias**, **customer favoritism**, and **keyword anchoring**. When critical issues are mislabeled as "Low" or trivial complaints are inflated to "Critical", Service Level Agreements (SLAs) are jeopardized and customer churn increases.

The **Support Integrity Auditor (SIA)** addresses this problem by building an intelligent auditing layer that:

- Infers the **true severity** of a ticket independently of the human-assigned priority
- Generates its own **binary mismatch supervision signal** (self-supervised, no ground-truth labels required)
- Trains a **fine-tuned classifier** on pseudo-labeled data to generalize to unseen tickets
- Produces a structured, **hallucination-free Evidence Dossier** for every flagged ticket

The system classifies each ticket into one of three categories:

| Category | Description |
|---|---|
| **Consistent** | Assigned priority aligns with inferred severity |
| **Hidden Crisis** | Ticket is genuinely severe but assigned a low priority |
| **False Alarm** | Ticket is not severe but assigned a high priority |

---

## 2. Live Application

The system is deployed as an interactive Streamlit web application supporting both single-ticket auditing and batch CSV upload.

**Live URL:** [https://support-integrity-auditor-j7xyhlerdxczw3aiyv8xui.streamlit.app/](https://support-integrity-auditor-j7xyhlerdxczw3aiyv8xui.streamlit.app/)

**Features:**
- Single ticket form input with real-time mismatch detection
- Batch CSV upload with downloadable results
- Priority Mismatch Dashboard showing distribution of flagged tickets
- Full Evidence Dossier rendered per flagged ticket
- Severity delta heatmap across ticket categories and channels

---

## 3. Dataset & Preprocessing

### Dataset

**Source:** [Customer Support Tickets — CRM Dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data)

| Column | Role |
|---|---|
| `Ticket_Subject` | Short summary of the issue |
| `Ticket_Description` | Full natural language problem statement |
| `Issue_Category` | Category of the issue (e.g. Technical, Billing) |
| `Priority_Level` | Human-assigned label (Low / Medium / High / Critical) |
| `Ticket_Channel` | Intake channel (email, chat, phone, social media) |
| `Resolution_Time_Hours` | Time to resolve — used as indirect severity proxy |
| `Satisfaction_Score` | Customer satisfaction rating |

### Preprocessing Steps

**Text Cleaning:**
Each ticket's `Ticket_Description` field was inspected and found to contain two distinct parts:
- The **first statement** — the actual customer complaint, which is semantically meaningful and directly relevant to severity inference
- The **second statement** — boilerplate or noise text appended uniformly across tickets with no discriminative value

Only the first statement was retained. The second statement was removed to prevent noise from diluting the severity signals.

The cleaned subject and description were combined into a single `clean_text` field:

Cleaning operations applied:
- Lowercasing
- URL and email removal
- Punctuation removal
- Whitespace normalization

**Feature Encoding:**
- `Ticket_Channel` → `channel_encoded` via `LabelEncoder` (fit on train only)
- `Issue_Category` → `category_encoded` via `LabelEncoder` (fit on train only)
- `Resolution_Time_Hours` → `resolution_time_norm` via `MinMaxScaler` (fit on train only)

**Train-Test Split:**
An 80/20 stratified split was performed before any encoding or normalization to prevent data leakage. All scalers and encoders were fit exclusively on the training set and applied to the test set.

```
Total tickets: 20,000
Train:         16,000
Test:           4,000
```

---

## 4. Pseudo-Label Generation

Since no ground-truth mismatch labels exist, the system bootstraps its own supervision signal by inferring ticket severity from four independent signals and comparing that inferred severity against the human-assigned priority.

**Binary priority mapping:**
```
Low / Medium  → 0  (low severity)
High / Critical → 1  (high severity)
```

### Signal 1 — Rule-Based NLP Score

A lightweight rule-based scorer scans ticket text for urgency and low-priority indicators.

**Urgency features captured:**
- High-urgency keywords: `urgent`, `critical`, `broken`, `outage`, `failure`, `data loss`, `security breach`, `escalate`
- Low-urgency keywords: `minor`, `suggestion`, `inquiry`, `no rush`, `whenever`
- Negation word density: `not`, `cannot`, `never`, `doesnt`
- Exclamation mark count
- Uppercase character ratio

**Score formula:**
```
rule_score = 0.4 × urgency_hits − 0.2 × low_hits + 0.15 × negation + 0.15 × exclamation + 0.1 × caps_ratio
```

Output: `rule_score` ∈ [0.0, 1.0]

### Signal 2 — Resolution Time Proxy (Random Forest Regression)

While semantic signals capture language-level urgency, operational severity is also reflected in how long a ticket takes to resolve. Tickets requiring greater effort or involving more complex business-critical issues tend to have longer resolution times.

Rather than using raw resolution time directly, a **Random Forest Regressor** is trained to predict resolution time from ticket text embeddings and metadata features. This allows the model to learn which combinations of textual content and operational context are associated with higher resolution effort.

**Model used:** `RandomForestRegressor` 

**Input features:**
- Sentence embeddings from `all-MiniLM-L6-v2` (384-dim)
- `channel_encoded`
- `category_encoded`
- `priority_encoded` (human label used as a feature, not a target)

**Target:** `resolution_time_norm` — MinMaxScaler-normalized resolution hours, fit on train only

The predicted resolution time is then binarized at the train-set median to produce a binary severity signal:

### Signal 3 — Embedding-Based Clustering

Ticket text is embedded using **`all-MiniLM-L6-v2`** from the Sentence Transformers library, producing dense 384-dimensional semantic vectors.

**K-Means clustering** is applied to group semantically similar tickets. The optimal number of clusters `k` is selected by maximizing the silhouette score across `k ∈ {2, ..., 8}`.

Cluster severity is mapped by computing average resolution time per cluster — clusters with higher average resolution time are assigned severity 1.

```
K-Means fit on train embeddings only
Cluster → severity map derived from train resolution time statistics
Applied to test via kmeans.predict()
```

Output: `cluster_score` ∈ {0, 1}

### Signal 4 — LLM-Based Zero-Shot Severity Scoring

A **Large Language Model** is used to score each ticket's urgency directly from its text content.

**Model used:** `Mistral-7B-Instruct`

Each ticket is scored against two hypotheses using zero-shot classification:

```
Hypothesis A: "this is an urgent critical issue requiring immediate attention"
Hypothesis B: "this is a low priority minor issue that can wait"
```

The probability assigned to Hypothesis A becomes the LLM severity score.

The LLM can identify urgency cues that keyword rules miss, including:
- Service outages described without explicit keywords
- Payment failures phrased as questions
- Account lockouts described politely
- Production incidents with indirect language

Output: `llm_score` ∈ [0.0, 1.0]

### Signal Fusion Strategy

The four signals are fused using a **weighted composite score**:

```
fused_score = 0.4 × llm_score_norm
            + 0.3 × cluster_score
            + 0.2 × resolution_time_norm
            + 0.1 × rule_score_norm
```

**Fusion rationale:**
- LLM score receives the highest weight because attention-based semantic reasoning captures urgency nuances that clustering and keywords miss
- Clustering receives secondary weight as it captures structural patterns across the ticket population
- Resolution time and rule scores act as supporting signals to stabilize boundary cases

**Threshold:** The fused score is binarized at the train-set median — tickets above the threshold are labeled severity 1, below as severity 0.

**Mismatch label generation:**
```
mismatch_label = 1  if inferred_severity ≠ priority_binary
mismatch_label = 0  if inferred_severity = priority_binary
```

**Mismatch type assignment:**
```
inferred=1, assigned=0  →  Hidden Crisis
inferred=0, assigned=1  →  False Alarm
inferred=assigned       →  Consistent
```

**Final pseudo-label distribution (train):**

| Label | Count | % |
|---|---|---|
| Consistent (0) | ~7,600 | ~47.5% |
| Mismatch (1) | ~8,400 | ~52.5% |

---

## 5. Classifier Training

### Model Architecture

A **DistilBERT** backbone is fine-tuned using **LoRA (Low-Rank Adaptation)** for parameter-efficient training. Structured metadata is fused with the text representation through a projection head.

**Models used:**
- Backbone: `distilbert-base-uncased`
- Adaptation: LoRA via `peft` library

**Architecture:**

```
Input Text (clean_text)
        ↓
DistilBERT (distilbert-base-uncased)
        ↓  LoRA adapters on q_lin + v_lin
CLS Token Output  (768-dim)
        ↓
        ├── Metadata Input [channel_encoded, category_encoded, resolution_time_norm]
        │         ↓
        │   Linear(3 → 32) + ReLU + Dropout
        │         ↓
        └──────── Concat (768 + 32 = 800-dim)
                  ↓
            Linear(800 → 256) + ReLU + Dropout
                  ↓
            Linear(256 → 2)
                  ↓
        Output: [Consistent, Mismatch]
```

---

## 6. Evidence Dossier Generation

For every ticket classified as a mismatch, the system generates a structured Evidence Dossier.

**LLM used for dossier generation:** `Mistral-7B-Instruct`

**Dossier schema:**

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "...",
  "feature_evidence": [
    {
      "signal": "keyword",
      "value": "...",
      "weight": "..."
    },
    {
      "signal": "resolution_time",
      "value": "...",
      "interpretation": "..."
    }
  ],
  "constraint_analysis": "<2-3 sentence grounded explanation>",
  "confidence": "..."
}
```

**Hard constraint:** Every `feature_evidence` item must be traceable to a specific field in the input ticket. The Mistral prompt explicitly prohibits fabricated or unverifiable claims. Any hallucination results in immediate disqualification of that test case.

**Confidence score** is derived from the severity delta:

---

## 7. Ablation Study

Each signal was evaluated independently and in combination to measure its individual contribution to mismatch detection.

| Configuration | Mismatch Count | Mismatch Rate |
|---|---|---|
| LLM Only | 11,718 | 58.59% |
| Rule Only | 12,237 | 61.18% |
| Resolution Only | 12,430 | 62.15% |
| Cluster Only | 17,777 | 88.88% |
| LLM + Resolution | 11,718 | 58.59% |
| LLM + Rule | 11,718 | 58.59% |
| LLM + Resolution + Rule | 12,042 | 60.21% |
| **All Signals (Final)** | **12,387** | **61.94%** |

**Key findings:**

- **Cluster Only** produces an extremely high mismatch rate (88.88%), indicating that KMeans alone over-detects mismatches and cannot be used as the sole signal
- **LLM Only** produces the most conservative and semantically grounded mismatch rate (58.59%)
- **All Signals fused** (61.94%) strikes a balance between LLM semantic precision and the structural diversity contributed by clustering and resolution time
- Resolution time and rule scores contribute modest but complementary signal — they improve boundary case stability without inflating the mismatch rate
- Signal diversity is confirmed by the disagreement between LLM and Cluster signals, validating the need for multi-signal fusion

**Why LLM receives the highest fusion weight (0.5):**
The LLM score consistently identifies semantically urgent tickets regardless of keyword presence. It captures implicit urgency (e.g. "my presentation is in 2 hours and the system is not responding") which both clustering and rule-based approaches miss.

---

## 8. Results

### Final Test Set Performance

| Metric | Value |
|---|---|
| Test Loss | 0.5297 |
| **Accuracy** | **0.7040** |
| **Macro F1** | **0.6830** |
| Recall — Consistent (0) | 0.6627 |
| Recall — Mismatch (1) | 0.8131 |

### Classification Report

```
                   precision    recall  f1-score   support

   Consistent (0)     0.9035    0.6627    0.7646      2176
     Mismatch (1)     0.4772    0.8131    0.6014       824

         accuracy                         0.7040      3000
        macro avg     0.6904    0.7379    0.6830      3000
     weighted avg     0.7864    0.7040    0.7198      3000
```

---

## Repository Structure

```
.
├── notebook.ipynb              # Full reproducible pipeline
├── train_pipeline.py           # Standalone training script
├── predict.py                  # Inference script (CSV → predictions + dossiers)
├── app.py                      # Streamlit web application
├── requirements.txt            # Pinned dependencies
└── README.md                   # This file
```


