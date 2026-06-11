# Support Integrity Auditor (SIA)

A semantics-driven, evidence-grounded automated auditor that detects **Priority Mismatch** in customer support tickets — cases where the objective characteristics of a ticket conflict with its human-assigned priority level.

## Problem

In enterprise CRM ecosystems, manual ticket triage is prone to agent fatigue bias, keyword anchoring, and customer favoritism. When critical issues are mislabeled "Low" or trivial complaints are inflated to "Critical," SLAs are jeopardized and customer churn increases. Existing rule-based systems fail to detect the nuanced discrepancies between a ticket's true severity and its assigned priority.

This system addresses a fundamentally harder variant: **there are no pre-annotated mismatch labels**. The system bootstraps its own supervision signal from raw ticket data alone.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    STAGE 1: PSEUDO-LABELING                  │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │  NLI-based   │  │  Keyword +   │  │  Resolution-Time  │   │
│  │  Severity    │  │  Negation +  │  │  Percentile       │   │
│  │  (50%)       │  │  Escalation  │  │  (20%)            │   │
│  │              │  │  (30%)       │  │                   │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘   │
│         └─────────────────┼───────────────────┘              │
│                    ┌──────▼──────┐                            │
│                    │   FUSION    │                            │
│                    │  Weighted   │                            │
│                    │  Average    │                            │
│                    └──────┬──────┘                            │
│                    ┌──────▼──────┐                            │
│                    │  MISMATCH   │                            │
│                    │  LABELING   │                            │
│                    │  (Δ ≥ 0.35) │                            │
│                    └─────────────┘                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   STAGE 2: CLASSIFIER                        │
│                                                              │
│  ┌────────────────────────┐  ┌────────────────────────────┐  │
│  │  DeBERTa-v3-small      │  │  Metadata Head             │  │
│  │  + LoRA (r=8, α=16)    │  │  (channel, category,       │  │
│  │  [CLS] embedding       │  │   domain, resolution_time) │  │
│  └───────────┬────────────┘  └──────────┬─────────────────┘  │
│              └──────────┬───────────────┘                     │
│                  ┌──────▼──────┐                              │
│                  │   CONCAT    │                              │
│                  │  768 + 16   │                              │
│                  └──────┬──────┘                              │
│                  ┌──────▼──────┐                              │
│                  │  MLP HEAD   │                              │
│                  │  → Binary   │                              │
│                  └─────────────┘                              │
│  Loss: Focal Loss (γ=2, class-weighted)                      │
│  Balance: Oversampling to 50/50                              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              STAGE 3: EVIDENCE DOSSIER                       │
│                                                              │
│  For each predicted mismatch, generate a structured JSON     │
│  dossier with grounded evidence:                             │
│    • NLI severity score + interpretation                     │
│    • Found keywords (with negation detection)                │
│    • Resolution time percentile                              │
│    • Mismatch type (Hidden Crisis / False Alarm)             │
│    • Grounded constraint analysis                            │
│    • Model confidence score                                  │
│                                                              │
│  HARD RULE: Zero hallucination — all evidence traceable      │
│             to specific input fields.                        │
└──────────────────────────────────────────────────────────────┘
```

## Dataset

**Customer Support Tickets — CRM Dataset**
- Source: [Kaggle](https://kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data)
- Size: 20,000 tickets
- Key columns: Ticket Subject, Description, Priority Level, Channel, Resolution Time, Issue Category, Customer Email

## Fusion Strategy Justification

### Why These Three Signals?

| Signal | Rationale | Weight |
|---|---|---|
| **NLI Severity** | Captures *semantic* urgency that keyword systems miss. Uses natural language inference to compare ticket text against urgency/triviality hypotheses. | 50% |
| **Keyword + Negation + Escalation** | Fast, interpretable signal. Enhanced with negation detection (e.g., "not a crash") and escalation phrases (e.g., "production down"). | 30% |
| **Resolution Time Percentile** | Category-grouped percentile ranking. Tickets that took longer to resolve within their category are likely more severe. | 20% |

### Ablation Study

Each signal was evaluated independently as the sole pseudo-labeler, with accuracy and F1 measured against the fused ground-truth labels:

| Signal | Accuracy | Macro F1 | Mismatch Rate |
|---|---|---|---|
| NLI Only | — | — | — |
| Keyword Only | — | — | — |
| Resolution Time Only | — | — | — |
| **Fused (All Three)** | **Baseline** | **Baseline** | **—** |

> **Note:** Run the pipeline (`python run_pipeline.py`) to populate these values. Results are saved to `outputs/evaluation_results.json`.

### Pairwise Signal Agreement

| Signal Pair | Agreement % | Cohen's κ |
|---|---|---|
| NLI ↔ Keyword | — | — |
| NLI ↔ Resolution Time | — | — |
| Keyword ↔ Resolution Time | — | — |

> **Note:** Computed automatically during pipeline execution.

## Project Structure

```
MARS/
├── config.py                      # Centralized hyperparameters
├── data_preprocessing.py          # Data loading, cleaning, encoding
├── pseudo_label_generation.py     # 3-signal fusion + ablation
├── classifier_training.py         # DeBERTa + LoRA training
├── evidence_dossier.py            # Dossier generation (§4 Stage 3)
├── evaluation.py                  # All §5 + §6 metrics
├── adversarial_test.py            # 10 adversarial tickets (§5 bonus)
├── run_pipeline.py                # End-to-end orchestrator
├── requirements.txt               # Python dependencies
├── customer_support_tickets.csv   # Dataset (not tracked in git)
├── SIA_FINAL_90_PERCENT.ipynb     # Original notebook (preserved)
├── SIA_Complete_Pipeline.ipynb    # Clean runnable notebook
└── outputs/                       # Generated outputs
    ├── model/                     # Saved model checkpoints
    ├── dossiers/                  # Evidence dossier JSONs
    ├── plots/                     # Confusion matrix, training curves
    ├── cache/                     # NLI score cache (avoid recomputation)
    ├── evaluation_results.json    # Full metrics
    ├── adversarial_results.json   # Adversarial test results
    └── final_results.json         # Comprehensive summary
```

## Setup & Running

### Prerequisites

- Python 3.9+
- CUDA GPU recommended (CPU works but is significantly slower)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd MARS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# Run everything end-to-end
python run_pipeline.py
```

### Running Individual Stages

```bash
# Stage 1: Preprocessing only
python data_preprocessing.py

# Stage 1+: Pseudo-label generation + ablation
python pseudo_label_generation.py

# Stage 2: Classifier training
python classifier_training.py

# Adversarial test (requires trained model)
python adversarial_test.py
```

### Running on Google Colab

Open `SIA_Complete_Pipeline.ipynb` in Google Colab for GPU-accelerated training.

## Evidence Dossier Schema

```json
{
  "ticket_id": "TKT-100042",
  "assigned_priority": "Low",
  "inferred_severity": "High",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 0.6732,
  "feature_evidence": [
    {
      "signal": "nli_severity",
      "value": 0.8234,
      "interpretation": "NLI model rates this ticket as highly urgent/severe"
    },
    {
      "signal": "keyword",
      "value": "crash(0.9), broken(0.9)",
      "weight": 0.6
    },
    {
      "signal": "resolution_time",
      "value": "45 hours (percentile: 0.92)",
      "interpretation": "Resolution took 45h (92nd percentile) — indicates complex/severe issue"
    }
  ],
  "constraint_analysis": "This Technical ticket was assigned 'Low' priority, but evidence suggests 'High' severity. Key indicators include NLI urgency score (0.82), urgency keywords (crash, broken), high resolution time (45h). The assigned priority may under-represent the true impact, risking SLA breach.",
  "confidence": 0.9134
}
```

## Evaluation Metrics

| Metric | Minimum Threshold |
|---|---|
| Binary Classification Accuracy | ≥ 83% |
| Macro F1 Score | ≥ 0.82 |
| Per-Class Recall (both classes) | ≥ 0.78 |
| Pseudo-Label Signal Agreement | Reported |
| Dossier Quality (zero hallucination) | 100% grounding |
| Adversarial Robustness (bonus) | ≥ 7/10 for +10% |

## Adversarial Test Cases

10 held-out tickets designed to fool keyword-based systems:

| # | Strategy | Expected |
|---|---|---|
| 1 | Keyword stuffing for trivial request | False Alarm |
| 2 | Polite language masking system outage | Hidden Crisis |
| 3 | Heavy negation of urgency keywords | Consistent |
| 4 | Formal corporate language, security breach | Hidden Crisis |
| 5 | Sarcastic tone masking data loss | Hidden Crisis |
| 6 | Extreme keyword stuffing, notification badge | False Alarm |
| 7 | Technical jargon, no keywords, data risk | Hidden Crisis |
| 8 | Emotional language, cosmetic preference | False Alarm |
| 9 | Euphemistic language, payment outage | Hidden Crisis |
| 10 | Mixed signals, compliance/audit risk | Hidden Crisis |

## Class Imbalance Handling

1. **Oversampling**: Minority class duplicated to achieve 50/50 balance
2. **Focal Loss**: γ=2.0, down-weights easy examples
3. **Class Weights**: Inverse-frequency weighting applied to loss function

## License

This project is for academic/research purposes.
