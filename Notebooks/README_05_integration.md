# Notebook 05 — Integration
## Detailed Documentation

**Notebook:** `05_integration.ipynb`  
**Inputs:**
- `../Data/Processed/ushmm_full_with_bert.csv` — VADER + BERT scores (all 10,116 reviews)
- `../Data/Processed/ushmm_semantic_tagged.csv` — semantic category tags (all 10,116 reviews)

**Output:** `../Data/Processed/ushmm_master.csv` — single merged master dataset  
**Figures saved to:** `../Outputs/Figures/`

---

## Purpose

This notebook is where the analysis becomes more than the sum of its parts. Notebooks 02–04b each answer a specific technical question in isolation: what are visitors saying (02), how emotionally do they write (03), and how positive is their overall sentiment (04). The integration notebook asks what happens at the *intersection* of these dimensions.

The core question is: **does semantic content predict the Memorial Paradox?** Are certain types of visitors — those who discuss learning, or reflection, or comparative context — more or less likely to show emotional-evaluative tension? And does the answer change when you use BERT vs VADER to measure that tension?

These cross-cutting findings are what connect the computational analysis back to transitional justice theory, and they are what the paper's new findings section is built on.

---

## Merge Strategy

The two input files both contain 10,116 rows derived from the same original dataset in the same row order. The merge assigns semantic columns by **row position** rather than by matching on `review_text`:

```python
df = sent.copy()
df["content_cats"] = sem["content_cats"].values
df["content_cats_str"] = sem["content_cats_str"].values
df["num_content_cats"] = sem["num_content_cats"].values
```

### Why position-based rather than text-based merge?

A text-based merge (`pd.merge(sent, sem, on="review_text")`) produces 10,126 rows instead of 10,116 because a small number of reviews share identical text — very short reviews like "Great museum!" that multiple visitors wrote independently. A text-based merge duplicates these rows, inflating the dataset by 10. A deduplication step then drops 15 rows (removing 5 genuine reviews with coincidentally identical text alongside the 10 duplicates).

Position-based assignment avoids this entirely. Since both files were produced from the same 10,116-row cleaned dataset processed in the same order, row N in `ushmm_full_with_bert.csv` corresponds to row N in `ushmm_semantic_tagged.csv`. The assignment is correct by construction.

**Prerequisite:** This approach assumes both input files were generated without row reordering. Both notebooks (02 and 03/04b) preserve the original row order from `ushmm_tripadvisor_eng.csv` and do not shuffle rows before saving. If this pipeline is modified in future, verify row order is preserved before using position-based merging.

---

## Section-by-Section Findings

### Section 2 — Category × Quadrant Cross-tabulation

This section explodes the `content_cats` list column so each row represents one (review, category) pair, then cross-tabulates categories against VADER quadrants.

**What "exploding" means:** A review tagged as both Learning/Understanding and Reflection/Suggestions becomes two rows in the exploded dataframe — one for each category. This means a single review contributes to the counts of multiple categories. Total row count in the exploded dataframe exceeds 10,116.

**The heatmap (row percentages)** shows what proportion of each category's reviews fall into each quadrant. This is more informative than raw counts because the categories have very different sizes (Learning Context has 6,875 reviews; Prior Knowledge has only 193).

**Key pattern:** Across all categories, the Emo+/Eval+ quadrant dominates at roughly 65–70%. The Emo-/Eval+ (Memorial Paradox) column is the analytically interesting one — it varies meaningfully across categories.

---

### Section 3 — Memorial Paradox Rate by Semantic Category

The single most important cross-cutting finding in the notebook.

| Category | Total | Paradox n | Paradox % |
|---|---|---|---|
| Learning / Understanding | 3,110 | 116 | **3.7%** |
| Learning Context (General) | 6,875 | 248 | 3.6% |
| Reflection / Suggestions | 3,616 | 120 | 3.3% |
| Motivation / Why Visit | 140 | 4 | 2.9% |
| Prior Knowledge | 193 | 5 | 2.6% |
| Comparison / Other Museums | 799 | 17 | **2.1%** |

**The TJ interpretation:**

The gradient from Learning/Understanding (3.7%) to Comparison/Other Museums (2.1%) is substantively meaningful. Visitors who explicitly describe what they learned — engaging cognitively and emotionally with the content — show the highest rate of emotional-evaluative tension. They feel the weight of what they encountered most acutely.

Visitors in comparative mode (Comparison/Other Museums) show the lowest paradox rate. These are the cosmopolitan travelers who have visited Auschwitz or Yad Vashem. They are more analytically detached — contextualizing their USHMM experience rather than being emotionally overwhelmed by it. Prior exposure to Holocaust memorial content may serve as an emotional buffer.

This finding connects to broader TJ theory about the role of prior engagement with difficult histories. First-time engagers (Learning/Understanding) show more emotional processing; repeat engagers (Comparison) show more analytical processing.

---

### Section 4 — BERT Score by Semantic Category

| Category | Mean BERT | Interpretation |
|---|---|---|
| Prior Knowledge | 0.565 | Most emotionally ambivalent |
| Comparison / Other Museums | 0.645 | Analytically engaged, moderate positivity |
| Learning Context (General) | 0.701 | Broad engagement, moderately positive |
| Learning / Understanding | 0.763 | Explicit learning linked to positive evaluation |
| Motivation / Why Visit | 0.771 | Goal-directed visitors evaluate positively |
| Reflection / Suggestions | 0.787 | Normative endorsement = highest BERT positivity |

**The Prior Knowledge finding is particularly interesting.** Visitors who admit ignorance (2% of corpus) write the most emotionally ambivalent reviews — lowest mean BERT score, indicating the most mixed sentiment signal. This may reflect genuine cognitive dissonance: the discomfort of realizing how little one knew about a genocide of this scale. This is a TJ finding about the role of prior knowledge in shaping museum engagement.

**Reflection/Suggestions reviews have the highest mean BERT score** despite also having the third-highest paradox rate. This apparent contradiction resolves when you understand that BERT and the VADER quadrants measure different things. The majority of Reflection/Suggestions reviews are confidently positive (high BERT score) — they recommend, endorse, prescribe. But the 3.3% who are in Emo-/Eval+ are doing something different: making normative civic prescriptions from a place of emotional heaviness. Both groups are captured in this category.

---

### Section 5 — Reflection/Suggestions + Emo-/Eval+

This section isolates the 120 reviews that simultaneously:
1. Make normative prescriptions about visiting, remembering, or acting ("everyone should," "never forget")
2. Show negative emotional tone with positive museum evaluation (Emo-/Eval+ quadrant)

These 120 reviews — 3.3% of all Reflection/Suggestions reviews — are the strongest transitional justice signal in the entire dataset. They represent visitors who have processed atrocity, felt its weight emotionally, and translated that experience into civic prescription. This is precisely the mechanism that TJ theory attributes to memorial museums: transforming individual encounter with difficult history into collective moral commitment.

**For the paper:** The Slidell, Louisiana family review (already Example 1 in the paper) appears in this subset, confirming that the categories are correctly capturing the phenomenon the paper describes in its introduction.

---

### Section 6 — Group Type Analysis

| Group Type | Paradox % |
|---|---|
| BUSINESS | 3.7% |
| COUPLES | 3.6% |
| FAMILY | 3.6% |
| NONE | 3.2% |
| SOLO | 3.2% |
| FRIENDS | 3.1% |

The remarkable finding here is the **near-total absence of group type effects**. The memorial paradox rate ranges only 0.6 percentage points across all visitor types. Whether visitors come with family, as a couple, with friends, or alone, they show essentially the same emotional-evaluative tension.

**TJ implication:** The Memorial Paradox is not a social phenomenon mediated by group dynamics. It is an individual cognitive and emotional response to the museum's content that persists regardless of social context. This suggests the museum's impact operates at the individual level, which has implications for how we think about the mechanisms through which memorialization affects publics.

**Business visitors** showing the highest paradox rate (3.7%) is counterintuitive and worth a footnote in the paper — business travelers may visit as part of organized educational programs, potentially producing more emotionally engaged encounters than leisure tourism.

---

### Section 7 — Domestic vs. International Analysis

| Origin | Mean BERT | Paradox % | Mean Rating |
|---|---|---|---|
| International | 0.715 | 3.5% | 4.680 |
| Domestic (US) | 0.667 | 3.6% | 4.726 |

**The null finding is itself significant.** International and domestic visitors show virtually identical memorial paradox rates (3.5% vs 3.6%) and essentially the same mean sentiment scores. Despite potentially different historical relationships to the Holocaust, different national memory cultures, and different distances traveled to reach the museum, international visitors engage with the USHMM in almost exactly the same way as Americans.

This finding speaks directly to Levy and Sznaider's (2006) concept of the globalization of Holocaust memory. The USHMM appears to successfully communicate its message across national and cultural contexts — visitors bring their own backgrounds but leave having processed the experience through a similar emotional-evaluative framework.

**Caveat:** The domestic/international classification uses US state names as the detection criterion. Reviewers who listed country-level locations ("United States") without a city are classified as unknown (2,336 reviews, 23% of corpus). If these unknowns are predominantly domestic, the true domestic sample is larger and the comparison may shift slightly. Future work could use a more comprehensive location parsing approach.

---

### Section 8 — Master Summary Table

The summary table consolidates all cross-cutting findings into a single paper-ready output:

| Column | Description |
|---|---|
| `n` | Number of reviews matching this category |
| `mean_rating` | Mean star rating of matching reviews |
| `mean_bert` | Mean BERT score of matching reviews |
| `mean_emo_vader` | Mean emotional VADER score |
| `mean_eval_vader` | Mean evaluative VADER score |
| `paradox_pct` | % of reviews in Emo-/Eval+ quadrant |

This table is the most compact summary of the paper's quantitative findings and should be considered for inclusion as a table in the paper itself.

---

## The Master Dataset: `ushmm_master.csv`

The final output contains all 10,116 reviews with every variable from the full pipeline:

| Column | Source | Description |
|---|---|---|
| `likes` | Raw | Helpful votes received |
| `rating` | Raw | Star rating (1–5) |
| `review_text` | Raw | Full review text |
| `group_type` | Raw | Visitor group type |
| `reviewer_location` | Raw | Self-reported city/country |
| `emo_vader` | Notebook 03 | Emotional VADER channel |
| `eval_vader` | Notebook 03 | Evaluative VADER channel |
| `emo_eval_bucket` | Notebook 03 | VADER quadrant label |
| `bert_score` | Notebook 04b | Signed BERT confidence score |
| `content_cats` | Notebook 02 | List of matched semantic categories |
| `content_cats_str` | Notebook 02 | Pipe-separated categories (CSV-safe) |
| `num_content_cats` | Notebook 02 | Count of matched categories |
| `is_domestic` | Notebook 05 | Boolean — US reviewer flag |

This file is the single source of truth for all downstream analysis and paper figures. Any future analysis should start from `ushmm_master.csv` rather than rebuilding the merge.

---

## Key Numbers for the Paper

**Memorial Paradox by category:**
- Highest: Learning/Understanding — **3.7%**
- Lowest: Comparison/Other Museums — **2.1%**

**Reflection/Suggestions + Emo-/Eval+ (strongest TJ signal):**
- **120 reviews** (3.3% of all Reflection/Suggestions reviews)

**Group type effects:** Range of 3.1–3.7% — effectively null finding

**Domestic vs. international:** 3.6% vs 3.5% — effectively null finding

**Prior Knowledge mean BERT score:** 0.565 — lowest of all categories, most emotionally ambivalent

---

## Limitations

**1. Position-based merge assumes row order stability.** If either input file is regenerated with a different row order (e.g., after shuffling), the merge will silently assign incorrect semantic tags to reviews. Always verify row counts match before running this notebook.

**2. Exploded dataframe inflates counts.** The exploded dataframe used in Sections 2–4 counts multi-category reviews multiple times — once per category. Raw counts in the cross-tabulation therefore cannot be summed to get total review counts. Percentages (row-normalized) are the appropriate unit for comparison.

**3. Domestic/international classification is incomplete.** 23% of reviews have no location or an unparseable location and are excluded from geographic analysis. If these missing reviews are systematically different from reviews with locations (e.g., if international visitors are more likely to omit location), the domestic/international comparison may be biased.

**4. VADER quadrant thresholds affect paradox rates.** The Emo-/Eval+ classification uses thresholds of emo < 0.0 and eval ≥ 0.2. The paradox rates reported in Section 3 are sensitive to these choices. Tightening the emo threshold (e.g., emo < -0.2) would reduce paradox rates; loosening it would increase them.

**5. Correlation ≠ causation in category effects.** The finding that Learning/Understanding reviews have a higher paradox rate than Comparison reviews does not mean that discussing learning *causes* emotional-evaluative tension. Both may be driven by a third variable — for example, first-time visitors to Holocaust museums may both discuss learning more and experience more emotional processing.
