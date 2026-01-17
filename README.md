# News Sentiment Analysis for Predicting Next-Day Stock Movements

This project studies whether financial news can predict **next-day stock movement** (Up / Down / NoChange).
We compare **sentiment-based baselines** (VADER, FinBERT sentiment) vs **direct text modeling** with FinBERT embeddings and limited fine-tuning.

> Course project (CS 6120 – NLP).  
> Authors: Abhinav Anil Kumar, Shreepoorna Purohit

---

## What’s inside this repo

- **Report:** [`report/Final_Report_CS6120.pdf`](report/Final_Report_CS6120.pdf)  
- **Notebook:** [`notebook/NLP_Project_Colab.ipynb`](notebook/NLP_Project.ipynb)

---

## Problem Setup

**Goal:** Predict next-day market movement from news text.

**Dataset:** FNSPID (Financial News + Stock Prices). We filtered:
- Years: **2018–2024**
- Tickers: **AAPL, AMZN, MSFT, GOOGL, META, TSLA, NVDA, JPM, BAC, XOM**
- News subset size: ~**56,506** articles after filtering.

---

## Labeling (Noise-aware 3-class)

We compute next-day return using OHLCV close prices:

Return(t → t+1) = (Close(t+1) − Close(t)) / Close(t)

Labels:
- **Up**: return ≥ +0.2%
- **Down**: return ≤ −0.2%
- **NoChange**: otherwise (deadband to ignore tiny fluctuations)

We use **time-based splits** to avoid leakage:
- Train: **2018–2021**
- Val: **2022**
- Test: **2023–2024**

---

## Pipeline Overview

1. **Filter** the raw FNSPID news + prices to target tickers and date window  
2. **Merge** news with OHLCV prices per ticker/day  
3. **Create labels** using next-day return threshold (±0.2%)  
4. **Modeling**
   - **Experiment 1 (Sentiment features):**
     - VADER / FinBERT sentiment scores → Logistic Regression / Random Forest
   - **Experiment 2 (Direct text modeling):**
     - FinBERT mean-pooled embeddings (768-d) → classifiers
     - Full-article modeling with 512-token limit and **chunking** for longer docs
5. **(Optional) Full text retrieval**
   - FNSPID body is often truncated → we scrape full articles with `newspaper3k`

---

## Key Results (from report)

### Experiment 1 — Sentiment baselines (VADER/FinBERT + LR/RF)
Performance is modest; Random Forest usually beats Logistic Regression.
(See report Table 1 for full matrix of variants.)

### Experiment 2 — Direct text modeling with FinBERT
Full text helps macro-F1 more than headline-only settings.
(See report Table 2 for variants including 512-token and chunked full-text.)

---

## How to Run (Colab)

1. Open the notebook in `notebook/` using Google Colab.
2. Set runtime to **GPU**: `Runtime → Change runtime type → GPU`
3. The notebook expects the FNSPID dataset in Google Drive with a layout like:
   `/MyDrive/NLP_Project/FSNPID/...`
4. Run cells top-to-bottom:
   - Data verification + seeding
   - News subset parquet creation
   - Price normalization + merge
   - Label generation + split
   - Experiments (VADER/FinBERT sentiment, embeddings, fine-tuning)

> Note: Raw FNSPID data is not included in this repo due to size/licensing.
> The notebook is written to run against Drive-mounted data.

---

## Tech Stack

Python · pandas · scikit-learn · PyTorch · HuggingFace Transformers  
NLTK (VADER) · FinBERT (ProsusAI/finbert) · newspaper3k · pyarrow/parquet

---

## Notes / Limitations

- Short-horizon stock prediction from news alone is noisy.
- Class imbalance (especially **NoChange**) affects macro-F1 and per-class recall.
- Full-scale FinBERT fine-tuning was compute-limited in Colab; smaller-scale runs are included.

---

## Citation

If you build on this work, please cite the report in `report/Final_Report_CS6120.pdf`.

