# Replication Archive
## "Reviewing Truth: What Can Museum Visitor Reviews Teach Us About Transitional Justice?"
**Ahlaad Yalavarthi, Hannah Ray, Kelebogile Zvobgo**  
William & Mary / International Justice Lab

---

## Overview

This repository contains the complete replication archive for the paper. It includes all data, code, and output figures necessary to reproduce the findings reported in the manuscript.

The analysis examines 10,116 English-language TripAdvisor reviews of the United States Holocaust Memorial Museum (USHMM) using pattern-based semantic categorization. The pipeline proceeds from raw data cleaning through semantic analysis, producing the figures and statistics reported in the paper.

---

## Repository Structure

```
ushmm-replication/
├── README.md
├── requirements.txt
├── .gitignore
├── Data/
│   ├── Raw/
│   │   └── ushmm_tripadvisor_full.xlsx      ← original scraped data (11,367 reviews)
│   └── Processed/
│       ├── ushmm_tripadvisor_eng.csv         ← English-only cleaned data (10,116 reviews)
│       ├── ushmm_tripadvisor_full_clean.csv  ← full data with dates and metadata
│       ├── ushmm_tripadvisor_eng_full.csv    ← English-only with dates and metadata
│       ├── ushmm_semantic_tagged.csv         ← reviews with semantic category tags
│       └── ushmm_master.csv                  ← final merged dataset
├── Notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 01b_data_exploration_full.ipynb
│   ├── 01c_temporal_extended.ipynb
│   ├── 02_semantic_analysis.ipynb
│   ├── 03_vader.ipynb
│   ├── 04_bert.ipynb
│   ├── 04b_bert_full.ipynb
│   └── 05_integration.ipynb
├── Outputs/
│   └── Figures/
│       ├── fig1_category_counts.pdf
│       ├── fig2_cooccurrence_heatmap.pdf
│       └── [additional figures]
└── Paper/
    └── reviewing_truth.pdf
```

---

## Setup

### Requirements
- Python 3.10 (recommended)
- Conda or virtualenv

### Installation

```bash
# Create and activate environment
conda create -n ushmm-replication python=3.10
conda activate ushmm-replication

# Install dependencies
pip install -r requirements.txt
```

### GPU Note
Notebook `04b_bert_full.ipynb` runs BERT sentiment analysis on all 10,116 reviews. A CUDA-compatible GPU is strongly recommended (tested on RTX 4070 Laptop GPU, ~3 minutes). On CPU this will take approximately 1–2 hours. Notebook `04_bert.ipynb` runs on a stratified sample of 1,242 reviews and is feasible on CPU (~30 minutes).

---

## Canonical Pipeline

Run notebooks in the following order to reproduce all paper results:

| Step | Notebook | Input | Output | Purpose |
|---|---|---|---|---|
| 1 | `01_data_exploration.ipynb` | `Data/Raw/ushmm_tripadvisor_full.xlsx` | `ushmm_tripadvisor_eng.csv` | Load, clean, filter to English |
| 1b | `01b_data_exploration_full.ipynb` | `Data/Raw/ushmm_tripadvisor_full.xlsx` | `ushmm_tripadvisor_full_clean.csv` | Extract dates, platform, reviewer metadata |
| 1c | `01c_temporal_extended.ipynb` | `ushmm_tripadvisor_full_clean.csv` | Figures | Temporal, seasonal, group type analysis |
| 2 | `02_semantic_analysis.ipynb` | `ushmm_tripadvisor_eng.csv` | `ushmm_semantic_tagged.csv` | Pattern-based semantic categorization |
| 3 | `03_vader.ipynb` | `ushmm_tripadvisor_eng.csv` | `ushmm_tripadvisor_eng_vader.csv` | Two-channel VADER sentiment analysis |
| 4 | `04_bert.ipynb` | `ushmm_tripadvisor_eng_vader.csv` | `ushmm_bert_sample.csv` | BERT on stratified 1,242-review sample |
| 4b | `04b_bert_full.ipynb` | `ushmm_tripadvisor_eng_vader.csv` | `ushmm_full_with_bert.csv` | BERT on full 10,116-review corpus |
| 5 | `05_integration.ipynb` | `ushmm_full_with_bert.csv` + `ushmm_semantic_tagged.csv` | `ushmm_master.csv` | Merge semantic and sentiment outputs |

**Note:** Notebooks 01b/01c and 03/04/04b/05 can be run in parallel after Step 1, since they all read from `ushmm_tripadvisor_eng.csv`. Run 04b before 05 to ensure the full BERT scores are available.

---

## Data

### Raw Data
`Data/Raw/ushmm_tripadvisor_full.xlsx` contains 11,367 TripAdvisor reviews of the USHMM scraped using the Apify TripAdvisor scraper service as of September 22, 2024. The earliest review dates to November 30, 2002.

Variables used in the analysis:

| Variable | Description |
|---|---|
| `rating` | Star rating (1–5) |
| `text` | Full review text |
| `tripType` | Visitor group (FAMILY, COUPLES, FRIENDS, SOLO, BUSINESS) |
| `user/userLocation/name` | Self-reported reviewer location |
| `helpfulVotes` | Number of helpful votes received |
| `publishedDate` | Date review was published |
| `travelDate` | Month of visit |
| `publishedPlatform` | Device used (OTHER/MOBILE/TABLET) |
| `user/contributions/totalContributions` | Reviewer's total TripAdvisor reviews |
| `lang` | Detected review language |

### Non-English Reviews
Reviews in languages other than English are removed in Notebook 01, reducing the dataset from 11,367 to 10,116 reviews (an 11% reduction). This may limit the diversity of perspectives from visitors with different cultural backgrounds. The full multilingual dataset is available in `ushmm_tripadvisor_full_clean.csv`.

### Reproducibility Note
Language detection uses the `langdetect` library, which is non-deterministic by default. Results may vary by ±2 reviews across runs. The paper reports 10,116 English reviews.

---

## Semantic Analysis

The paper uses pattern-based semantic categorization with six content categories:

| Category | Description | Coverage |
|---|---|---|
| Learning Context (General) | References to historical content, exhibits, artifacts | 68.0% |
| Reflection / Suggestions | Normative statements, civic prescription, "never forget" | 35.7% |
| Learning / Understanding | Explicit educational gains or changed perspectives | 30.7% |
| Comparison / Other Museums | References to other Holocaust memorial sites | 6.3% |
| Motivation / Why Visit | Stated reasons for choosing to visit | 4.4% |
| Prior Knowledge | Expressions of prior ignorance or corrected misconceptions | 1.5% |

Overall coverage: 85.1% of English reviews match at least one category.

The regex pattern sets for all six categories are defined in `02_semantic_analysis.ipynb`. The methodology was validated against large language model classification (Llama 3.1, 78.9% agreement). Full method comparison details are in Appendix B of the paper.

---

## Sentiment Analysis (Supplementary)

The sentiment analysis pipeline (Notebooks 03–05) is provided as supplementary material. It includes:

- **VADER two-channel model** (`03_vader.ipynb`): separates emotional tone (`emo_vader`) from evaluative sentiment (`eval_vader`) to address the Memorial Paradox
- **BERT scoring** (`04_bert.ipynb`, `04b_bert_full.ipynb`): uses `distilbert-base-uncased-finetuned-sst-2-english`
- **Integration** (`05_integration.ipynb`): merges semantic and sentiment outputs

These notebooks produce `ushmm_master.csv`, which includes all variables from the full pipeline.

---

## Figures

All figures are saved as both PDF (for LaTeX) and PNG (300 DPI) in `Outputs/Figures/`. Figures are automatically regenerated when notebooks are rerun and will overwrite existing files.

| File | Notebook | Description |
|---|---|---|
| `fig1_category_counts.pdf` | 02 | Bar chart of semantic category distribution |
| `fig2_cooccurrence_heatmap.pdf` | 02 | Category co-occurrence heatmap |
| `temporal_annual_counts.png` | 01b | Annual review volume with COVID and Oct 7 markers |
| `temporal_monthly_trend.png` | 01b | Monthly review trend 2002–2024 |
| `seasonal_pattern.png` | 01c | Average monthly volume and rating by season |
| `rating_over_time.png` | 01c | Mean star rating by year |
| `vader_sentiment_distribution.png` | 03 | VADER score distribution |
| `vader_quadrant_distribution.png` | 03 | Emo/Eval quadrant distribution |
| `bert_full_score_by_rating.png` | 04b | BERT score by star rating (full corpus) |
| `all_channels_by_rating.png` | 04b | All three sentiment channels by rating |

---

## Citation

If you use this code or data, please cite:

```
Yalavarthi, Ahlaad, Hannah Ray, and Kelebogile Zvobgo. [Year].
"Reviewing Truth: What Can Museum Visitor Reviews Teach Us About
Transitional Justice?" [Journal].
```

---

## Contact

For questions about the code or data:
- Ahlaad Yalavarthi — ayalavarthi@wm.edu
- Kelebogile Zvobgo — kzvobgo@wm.edu
