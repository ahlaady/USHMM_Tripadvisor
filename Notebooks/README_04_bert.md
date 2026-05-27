# Notebooks 04 & 04b — BERT Sentiment Analysis
## Detailed Documentation

**Notebooks:** `04_bert.ipynb` (stratified sample) and `04b_bert_full.ipynb` (full corpus)  
**Input:** `../Data/Processed/ushmm_tripadvisor_eng_vader.csv`  
**Outputs:**
- `../Data/Processed/ushmm_bert_sample.csv` — 1,242 stratified reviews with bert_score (from 04)
- `../Data/Processed/ushmm_full_with_bert.csv` — all 10,116 reviews with bert_score (from 04b)

**Figures saved to:** `../Outputs/Figures/`

---

## Purpose

These notebooks apply transformer-based sentiment analysis to the USHMM reviews and compare BERT's performance against VADER's two-channel model. The central question is whether a more sophisticated neural model does a better job of capturing visitor sentiment — and in particular, whether it handles the Memorial Paradox better or worse than VADER.

The answer is nuanced: **BERT achieves a substantially higher correlation with star ratings**, but it does not solve the Memorial Paradox — it still scores roughly 1 in 9 high-star reviews as negative, which is precisely the empirical measure of the paradox the paper reports.

Notebooks 04 and 04b are run sequentially. **04 should be run first** for methodology documentation purposes (it produces the stratified sample results used for model comparison). **04b should always be run before 05** since the integration notebook reads from `ushmm_full_with_bert.csv`.

---

## Why BERT?

BERT (Bidirectional Encoder Representations from Transformers) is a deep neural language model that represents text as contextual embeddings rather than individual word scores. Unlike VADER, which scores each word independently, BERT reads the full sentence and considers how words modify each other. This makes it theoretically better at handling:

- Negation ("not terrible" vs "terrible")
- Context ("the exhibits were dark" — dark as in gloomy/serious, not negative)
- Sentence-level meaning rather than word-level valence

For a paper addressing the Memorial Paradox, BERT's contextual reading is directly relevant — it should, in theory, understand that "horrific exhibits" in a 5-star review functions differently than "horrific service" in a 1-star review.

**The practical result:** BERT does outperform VADER substantially on correlation with star ratings. However, it still misclassifies a meaningful portion of Memorial Paradox reviews — atrocity language is sufficiently extreme that even contextual reading cannot fully separate it from negative evaluation.

---

## The Model: `distilbert-base-uncased-finetuned-sst-2-english`

### Why DistilBERT rather than full BERT?
DistilBERT is a distilled (compressed) version of BERT that retains ~97% of BERT's performance at ~60% of the size and ~40% of the compute cost. For a corpus of 10,116 reviews, full BERT would take significantly longer to run with negligible quality improvement. DistilBERT is the standard choice for production sentiment analysis tasks.

### Why SST-2?
SST-2 (Stanford Sentiment Treebank) is the standard binary sentiment classification benchmark. The model fine-tuned on SST-2 classifies text as POSITIVE or NEGATIVE with a confidence score between 0.5 and 1.0.

### The critical limitation
SST-2 was constructed from **movie reviews**. The model has never encountered Holocaust memorial content, atrocity vocabulary, or the emotional register of grief-as-endorsement. This is a genuine limitation that the paper should acknowledge.

However, the fact that BERT still achieves a 0.603 correlation with star ratings on the stratified sample — despite being trained on an entirely different domain — is itself a finding. It suggests the evaluative signal in museum reviews is strong enough to survive domain mismatch.

---

## Score Construction

BERT produces two outputs per review: a label (`POSITIVE` or `NEGATIVE`) and a confidence score (0.5–1.0). These are converted to a single signed score:

```python
def signed_score(row):
    return row["score"] if row["label"] == "POSITIVE" else -row["score"]
```

This produces a score in the range (-1.0, -0.5] ∪ [0.5, 1.0) — note there are no scores between -0.5 and 0.5 because BERT always outputs a binary label with at least 50% confidence. This is fundamentally different from VADER's continuous -1 to +1 range and means BERT scores cluster near the extremes.

**Implication for interpretation:** A BERT score of -0.55 and -0.99 are both "NEGATIVE" — they differ only in confidence. A VADER score of -0.55 and -0.99 are meaningfully different in magnitude. Keep this in mind when comparing distributions across the two tools.

---

## Notebook 04 — Stratified Sample

### Why stratify?

The full corpus is heavily imbalanced — 80% of reviews are 5-star. If BERT were run and evaluated on the full corpus, the performance metrics would be dominated by 5-star reviews. The correlation with star rating would tell us mostly how well BERT identifies very positive reviews, not how well it discriminates across the full rating spectrum.

Stratified sampling solves this by taking equal representation across all star ratings:

| Star Rating | Full Corpus | Stratified Sample |
|---|---|---|
| 1★ | 75 | 75 (all) |
| 2★ | 117 | 117 (all) |
| 3★ | 419 | 350 |
| 4★ | 1,397 | 350 |
| 5★ | 8,108 | 350 |
| **Total** | **10,116** | **1,242** |

1★ and 2★ reviews are included in full because they are rare — there are not enough to sample 350 from. 3★, 4★, and 5★ are each capped at 350. The total of 1,242 reflects this ceiling, not an arbitrary design choice.

### What stratification is for
The stratified sample is **for model comparison and performance measurement only**. It answers: how well does each sentiment tool discriminate across the rating spectrum? It is not intended to be a representative sample of USHMM reviews — for that, the full corpus is used (Notebook 04b).

Results from the stratified sample should be reported alongside results from the full corpus, with clear explanation of why the numbers differ.

---

## Notebook 04b — Full Corpus

### Why run the full corpus separately?

The stratified sample intentionally overrepresents 1★ and 2★ reviews. Any aggregate statistics computed on it (mean BERT score, memorial paradox rate, etc.) will not reflect the actual distribution of USHMM visitor sentiment. The full corpus run produces the numbers that are externally valid — they reflect what the actual population of English-language USHMM reviewers wrote.

### Hardware requirement
Running BERT on 10,116 reviews requires a GPU for practical runtime. On an RTX 4070 Laptop GPU with batch size 64, this takes approximately 2–3 minutes. On CPU it would take 1–2 hours.

If no GPU is available, the `ushmm_full_with_bert.csv` from Notebook 04 (which has `bert_score` populated for only 1,242 rows, NaN elsewhere) can be used as a fallback for the integration notebook, with the caveat that Section 3 of Notebook 05 will produce results based only on the sample rows.

---

## Results Comparison: Stratified Sample vs. Full Corpus

### Correlations with star rating

| Model | Stratified Sample | Full Corpus |
|---|---|---|
| BERT | **0.603** | **0.414** |
| eval_vader | 0.319 | 0.192 |
| emo_vader | 0.163 | 0.080 |

**Why the full corpus correlations are lower:** The full corpus has 80% 5-star reviews, which compresses the variance in the rating variable. When most reviews have the same rating, correlations with any predictor are attenuated. This is a statistical artifact, not evidence that the models perform worse on the full corpus.

Both sets of numbers are correct and should be reported for different purposes:
- **Stratified sample correlations** → model comparison (how well each tool discriminates)
- **Full corpus correlations** → external validity (how the tools perform on real-world distribution)

### Memorial Paradox rate

| Corpus | High-star reviews | BERT-negative | Paradox rate |
|---|---|---|---|
| Stratified sample | 700 (350×2) | 71 | **10.1%** |
| Full corpus | 9,505 | 1,202 | **11.9%** |

The full corpus rate is slightly higher than the stratified sample — meaning the stratified sample *underestimates* the paradox rate. This makes sense: the stratified sample over-represents lower-star reviews, which pulls the model toward negative predictions and makes fewer high-star reviews appear paradoxical by comparison.

The full corpus rate of **11.9%** is the number to report in the paper — it reflects the actual proportion of USHMM reviewers who use emotionally negative language while rating the museum highly.

---

## Output Files

### `ushmm_bert_sample.csv`
The stratified 1,242-review sample with all original columns plus `bert_score`. Used for model comparison analysis. Not used directly by Notebook 05.

### `ushmm_full_with_bert.csv`
All 10,116 reviews with `bert_score` populated for every row. This is the file Notebook 05 reads from. **Always run 04b before 05** to ensure this file contains full-corpus scores.

---

## Figures

| Filename | Source | Description |
|---|---|---|
| `bert_confidence_distribution.png` | 04 | Histogram of BERT confidence scores by label (POSITIVE/NEGATIVE) |
| `bert_score_by_rating.png` | 04 | Boxplot of BERT scores vs star rating (stratified sample) |
| `bert_vader_comparison.png` | 04 | Side-by-side correlation bars: BERT vs VADER channels |
| `bert_full_score_by_rating.png` | 04b | Boxplot of BERT scores vs star rating (full corpus) |
| `all_channels_by_rating.png` | 04b | Three-panel boxplot: all channels side by side (full corpus) |
| `sentiment_model_comparison_full.png` | 04b | Correlation bar chart — all three models vs star rating (full corpus) |

---

## Key Numbers for the Paper

**Stratified sample (n=1,242):**
- BERT vs rating correlation: **0.603**
- eval_vader vs rating correlation: 0.319
- emo_vader vs rating correlation: 0.163
- Memorial Paradox rate (high-star, BERT-negative): **10.1%**

**Full corpus (n=10,116):**
- BERT vs rating correlation: **0.414**
- eval_vader vs rating correlation: 0.192
- emo_vader vs rating correlation: 0.080
- Memorial Paradox rate (high-star, BERT-negative): **11.9%** (1,202 reviews)
- Mean BERT score: 0.669

---

## Limitations

**1. SST-2 domain mismatch.** The model was fine-tuned on movie reviews. It has no exposure to Holocaust memorial content, survivor testimony language, or atrocity vocabulary in the context of memorialization. Despite this, the model performs well — which is a finding in itself — but its limitations in this specific domain should be acknowledged.

**2. Binary classification.** BERT produces only POSITIVE or NEGATIVE, with no neutral class. Reviews that express genuine ambivalence or mixed sentiment are forced into one category. VADER's continuous scale is more expressive for these cases.

**3. Truncation at 256 tokens.** Long reviews are truncated to 256 tokens (approximately 180–200 words). Reviews that open with extended atrocity description before concluding with positive evaluation may be classified as negative if the positive evaluation falls beyond the truncation point.

**4. Stratified sample is not representative.** Results from Notebook 04 describe model performance under controlled conditions, not the actual distribution of USHMM visitor sentiment. The two sets of results must not be conflated.

**5. No domain fine-tuning.** A BERT model fine-tuned on USHMM reviews specifically — or on Holocaust museum review text more generally — would almost certainly produce better results. This would require labeled training data that does not currently exist for this domain. The paper could flag this as a direction for future work.

**6. Reproducibility of BERT inference.** Although seeds are set for Python, NumPy, and PyTorch, GPU inference can be non-deterministic across hardware. Results may vary by ±0.001–0.005 on aggregate statistics across different machines.
