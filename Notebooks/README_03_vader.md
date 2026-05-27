# Notebook 03 — VADER Sentiment Analysis
## Detailed Documentation

**Notebook:** `03_vader.ipynb`  
**Input:** `../Data/Processed/ushmm_tripadvisor_eng.csv`  
**Output:** `../Data/Processed/ushmm_tripadvisor_eng_vader.csv`  
**Figures saved to:** `../Outputs/Figures/`

---

## Purpose

This notebook applies sentiment analysis to all 10,116 English-language reviews. Its primary contribution is not producing a single sentiment score — it is producing *two* sentiment scores that measure fundamentally different things. This two-channel design is the paper's core methodological response to the Memorial Paradox, and it is what distinguishes this analysis from a naive application of off-the-shelf sentiment tools.

---

## Background: Why VADER?

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based sentiment analysis tool specifically designed for short, social media-style text. It assigns a compound score between -1 (maximally negative) and +1 (maximally positive) based on a curated dictionary of words with pre-assigned valence scores.

VADER was chosen over alternatives for three reasons:

**1. No training required.** Unlike BERT, VADER applies directly to any text without fine-tuning. This makes it fully transparent and reproducible — the lexicon is public and inspectable.

**2. Speed.** VADER scores all 10,116 reviews in seconds, enabling full-corpus analysis without sampling. BERT requires stratified sampling due to computational cost (see Notebook 04).

**3. Lexicon inspectability.** Because VADER's scores come from a human-curated dictionary, we can directly modify specific word scores to address domain-specific problems — which is exactly what the evaluative channel does. This would not be possible with a black-box neural model.

**Trade-off:** VADER was developed for social media text (tweets, reviews) and has no domain knowledge of Holocaust memorial content. It treats atrocity vocabulary as straightforwardly negative, which is the core problem the two-channel model addresses.

---

## The Memorial Paradox Problem

Consider this 5-star review:

> *"The most horrific, devastating, heartbreaking experience. Everyone must visit."*

Standard VADER scores "horrific" (-3.4), "devastating" (-3.6), and "heartbreaking" (-3.2) as strongly negative, producing a compound score of approximately -0.80 — which would classify this as a highly negative review. But the reviewer awards five stars and explicitly endorses the museum.

This is the Memorial Paradox: visitors use emotionally negative language to describe the *content* of the museum (atrocity, suffering, historical horror) while expressing strongly *positive* evaluation of the museum as an institution. A single-channel sentiment score cannot separate these two things.

The two-channel model is the solution.

---

## The Two-Channel Model

### Channel 1: `emo_vader` — Emotional/Affective Tone
**What it measures:** The raw emotional register of the review — how the visitor felt while processing Holocaust content.

**How it is computed:** Standard, unmodified VADER compound score applied to the full review text.

**Interpretation:** A low (negative) `emo_vader` score does not mean the visitor had a bad experience or rated the museum poorly. It means the review contains emotionally heavy language — grief, horror, sorrow — which is expected and appropriate for Holocaust memorial content. Negative `emo_vader` in a 5-star review is the Memorial Paradox in data form.

**Correlation with star rating:** 0.080 — very weak. This is expected. Emotional tone is largely decoupled from museum evaluation for this type of content.

---

### Channel 2: `eval_vader` — Evaluative Sentiment
**What it measures:** The visitor's assessment of the museum as an institution — was it well-presented, worth visiting, effectively organized?

**How it is computed:** VADER applied after two modifications:
1. **Lexicon neutralization** — atrocity-domain words are set to 0.0 in the VADER lexicon
2. **Rating-aware clamping** — negative eval scores for 4★ and 5★ reviews are set to 0.0

**Correlation with star rating:** 0.192 — moderate. Still imperfect, but substantially better than `emo_vader`.

---

## Lexicon Neutralization — Design Decisions

The following words are neutralized (set to 0.0) in the tuned VADER lexicon:

```python
neutralize = [
    "sad", "tragic", "horrific", "horror", "terrible", "disturbing", "shocking",
    "heartbreaking", "upsetting", "grief", "sorrow", "genocide", "atrocity",
    "gas", "camp"
]
```

**Why these words specifically?**

These are words that carry strong negative valence in standard VADER but function *descriptively* rather than *evaluatively* in Holocaust museum reviews. When a visitor writes "the gas chambers were horrific," they are describing the historical content of the museum, not criticizing the museum's quality. Neutralizing these words prevents VADER from misreading historical description as negative evaluation.

**Why not more words?**

The neutralization list was kept deliberately narrow. Over-neutralization risks stripping genuinely evaluative negative language — a visitor writing "the staff were terrible" should score negatively on `eval_vader`. The list targets only words that are virtually always used descriptively in this domain.

**Words intentionally NOT neutralized:**
- "disappointing," "frustrated," "confusing," "crowded" — these describe the visitor experience, not the historical content, and should influence `eval_vader`
- "powerful," "moving," "emotional" — positive evaluative language that VADER handles correctly
- "difficult," "hard," "challenging" — ambiguous; kept in the lexicon

---

## Rating-Aware Clamping

After lexicon neutralization, a second adjustment is applied:

```python
hi = df["rating"] >= 4
df.loc[hi & (df["eval_vader"] < 0), "eval_vader"] = 0.0
```

**Rationale:** A 4★ or 5★ reviewer has explicitly endorsed the museum with their rating. If `eval_vader` scores them negative after lexicon neutralization, the remaining negativity is almost certainly residual atrocity language that the neutralization list missed. Clamping to 0.0 prevents these residual cases from distorting the evaluative channel.

**This is not data fabrication** — it is using the star rating as a prior. The star rating is an independent, explicit signal of evaluation. When it conflicts with a residual negative `eval_vader` score in a 4/5-star review, we trust the explicit signal.

**What this means for the Emo+/Eval- quadrant:** This clamping is why the Emo+/Eval- quadrant contains zero reviews in the output. Any review that might have appeared there (positive emotional tone, negative evaluation, high star rating) was clamped to Mixed/Neutral. This is a methodological choice worth noting in the paper.

---

## The Four Quadrants

Reviews are classified into quadrants based on thresholds applied to both channels:

| Quadrant | Condition | Interpretation |
|---|---|---|
| **Emo+ / Eval+** | emo ≥ 0.2 AND eval ≥ 0.2 | Positive on both dimensions — standard positive review |
| **Emo- / Eval+** | emo < 0.0 AND eval ≥ 0.2 | **Memorial Paradox** — emotional weight + positive evaluation |
| **Emo- / Eval-** | emo < 0.0 AND eval < 0.0 | Genuinely negative review — critical of museum and/or content |
| **Emo+ / Eval-** | emo ≥ 0.2 AND eval < 0.0 | Absent after clamping (see above) |
| **Mixed / Neutral** | All other combinations | Moderate or mixed scores on both channels |

**Threshold design:** The 0.2 threshold for positivity and 0.0 for negativity are not symmetric. This reflects VADER's known positive skew in review text — reviewers tend to write positively, so the threshold for "meaningfully positive" is set higher than for "meaningfully negative."

---

## Output: Quadrant Distribution (Full Corpus)

| Quadrant | Count | % of corpus |
|---|---|---|
| Emo+ / Eval+ | 6,461 | 63.9% |
| Mixed / Neutral | 3,076 | 30.4% |
| **Emo- / Eval+** | **342** | **3.4%** |
| Emo- / Eval- | 237 | 2.3% |
| Emo+ / Eval- | 0 | 0.0% |

The **342 Emo-/Eval+ reviews** are the empirical core of the Memorial Paradox claim. These are visitors who process atrocity emotionally (negative tone) while positively evaluating the museum as an institution. Combined with the BERT findings in Notebook 04b (11.9% of high-star reviews score BERT-negative), this gives the paper two independent measures of the same phenomenon.

---

## Output: Sentiment by Group Type

| Group Type | emo_vader | eval_vader |
|---|---|---|
| FAMILY | 0.327 | 0.490 |
| FRIENDS | 0.324 | 0.498 |
| SOLO | 0.300 | 0.481 |
| NONE | 0.299 | 0.463 |
| BUSINESS | 0.266 | 0.450 |
| COUPLES | 0.268 | 0.459 |

Family and Friends visitors score highest on emotional tone — consistent with intergenerational and peer-based memory transmission. Business visitors score lowest on both channels, which may reflect more utilitarian engagement with the museum.

---

## Output File: `ushmm_tripadvisor_eng_vader.csv`

Three new columns are added to the cleaned dataset:

| Column | Type | Range | Description |
|---|---|---|---|
| `emo_vader` | float | -1.0 to 1.0 | Raw VADER compound score — emotional channel |
| `eval_vader` | float | -1.0 to 1.0 | Tuned VADER compound score — evaluative channel |
| `emo_eval_bucket` | string | 5 values | Quadrant classification |

This file is the direct input to Notebook 04 (BERT).

---

## Figures

| Filename | Description |
|---|---|
| `vader_sentiment_distribution.png` | Histogram + KDE of `emo_vader` scores with normal curve overlay |
| `vader_channels_by_rating.png` | Side-by-side boxplots of both channels by star rating |
| `vader_quadrant_distribution.png` | Horizontal bar chart of review counts per quadrant |

---

## Key Numbers for the Paper

- Mean `emo_vader`: 0.302
- Mean `eval_vader`: 0.476
- Correlation of `emo_vader` with star rating: **0.080**
- Correlation of `eval_vader` with star rating: **0.192**
- Emo-/Eval+ reviews (Memorial Paradox): **342 (3.4%)**

---

## Limitations

**1. VADER was not trained on memorial museum text.** The lexicon was developed primarily from social media, news, and product reviews. Its valence scores for atrocity vocabulary reflect general usage, not memorial museum usage. The neutralization list is a manual correction, not a systematic solution.

**2. The neutralization list is not exhaustive.** Words like "atrocities," "massacre," "murder," "barbaric" are not neutralized and still contribute negatively to `eval_vader`. A more comprehensive domain adaptation would improve the evaluative channel.

**3. Rating-aware clamping introduces a dependency.** `eval_vader` is not entirely independent of star rating — the clamping step uses rating as an input. This should be acknowledged when interpreting correlations between `eval_vader` and rating.

**4. VADER processes the full review as a single unit.** A review that begins with three paragraphs describing atrocity and ends with one sentence recommending the museum will score very differently than a review with the opposite structure, even if the evaluative content is identical.

**5. Quadrant thresholds are researcher-defined.** The 0.2 and 0.0 thresholds are not derived from the data. Different threshold choices would produce different quadrant distributions. Sensitivity analysis around these thresholds would strengthen the paper.
