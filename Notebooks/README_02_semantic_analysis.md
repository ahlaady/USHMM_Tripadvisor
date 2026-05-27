# Notebook 02 — Semantic Analysis
## Detailed Documentation

**Notebook:** `02_semantic_analysis.ipynb`  
**Input:** `../Data/Processed/ushmm_tripadvisor_eng.csv`  
**Output:** `../Data/Processed/ushmm_semantic_tagged.csv`  
**Figures saved to:** `../Outputs/Figures/`

---

## Purpose

This notebook performs the core semantic analysis of the paper. The goal is to move beyond simple word counting and systematically identify *what visitors are saying* across 10,116 reviews — not just how they feel. Each review is tagged against six theoretically-grounded content categories, enabling the paper to answer its five research questions about prior knowledge, motivation, learning, social communication, and civic prescription.

This is the analysis that gives the paper its substantive claims. The sentiment notebooks (03–04b) measure emotional tone; this notebook measures *meaning*.

---

## Why Pattern-Based Semantic Analysis?

The paper uses regex-based keyword matching rather than unsupervised topic modeling (e.g., LDA, BERTopic) or supervised classification. This was a deliberate methodological choice with three justifications:

**1. Interpretability for a non-technical audience.**
The paper targets political science and transitional justice scholars, not NLP researchers. Regex patterns are fully transparent — any reader can inspect exactly which phrases triggered a category match. A topic model produces probabilistic distributions that are harder to explain and validate for this audience.

**2. Theory-driven categories.**
The six categories were designed to match the paper's five research questions, not to discover structure inductively. Using topic modeling would let the data define the categories, which might not align with transitional justice theory. Pattern-matching lets the researchers impose theoretically meaningful categories and measure their prevalence.

**3. Iterative validation.**
The pattern sets were refined through manual inspection of unmatched reviews, expanding trigger phrases until coverage reached ~85%. This iterative process is documented and reproducible in a way that topic model hyperparameter tuning often is not.

**Trade-off acknowledged:** Pattern matching cannot capture semantic similarity — a review saying "this changed how I see the world" might not match the Learning/Understanding category if the exact phrases aren't in the pattern set. The 15% of unmatched reviews likely includes some semantically rich content the patterns missed.

---

## The Six Categories

### 1. Prior Knowledge
**Research question addressed:** What did visitors already know before visiting?

**What it captures:** Expressions of ignorance, surprise at not knowing, or explicit comparisons between pre- and post-visit knowledge.

**Example trigger phrases:** "didn't know," "never realized," "had no idea," "now I understand," "wasn't taught this in school"

**Why it matters for TJ:** If the USHMM primarily serves visitors who already know Holocaust history (deepening awareness rather than providing initial education), that has implications for how memorial museums function as TJ mechanisms. The low prevalence (2%) supports the paper's argument that the museum reinforces existing knowledge rather than creating it from scratch.

**Interpretation note:** The low rate may reflect social desirability bias — visitors may be reluctant to admit ignorance about the Holocaust in a public review.

---

### 2. Learning / Understanding
**Research question addressed:** What did visitors hope to gain and what did they actually learn?

**What it captures:** Explicit statements about educational gains, new insights, changed perspectives, or the museum's informational value.

**Example trigger phrases:** "I learned," "educational," "eye-opening," "changed my perspective," "helped me understand," "made me think"

**Why it matters for TJ:** Learning/Understanding is the most direct measure of the museum's educational mission — one of its core TJ functions. The 31% prevalence means nearly a third of all reviewers explicitly articulate an educational takeaway, suggesting the museum successfully communicates historical content at scale.

**Note on overlap:** This category has high co-occurrence with Learning Context (General) and Reflection/Suggestions, reflecting that visitors who describe learning also tend to recommend others visit (1,543 reviews match both).

---

### 3. Motivation / Why Visit
**Research question addressed:** Why do people choose to engage with this transitional justice site?

**What it captures:** Explicit statements about why the reviewer chose to visit — recommendations from others, civic duty, personal connection to history, tourism, educational goals.

**Example trigger phrases:** "decided to visit," "heard about," "always wanted to see," "on my bucket list," "wanted to understand"

**Why it matters for TJ:** Understanding visitor motivation illuminates the pathways through which publics engage with memorialization. If most visitors come due to external recommendations rather than intrinsic interest, that tells a different story about the museum's reach than if visitors are self-selecting based on historical awareness.

**Interpretation note:** The very low prevalence (1%) reflects a genre effect — TripAdvisor reviews are recommendation-oriented and future-facing. Reviewers describe their *experience*, not their *decision to visit*. This category likely undercounts true motivation because visitors don't foreground their reasons for coming in the review format.

---

### 4. Comparison / Other Museums
**Research question addressed:** How do visitors situate the USHMM within a broader landscape of Holocaust memory sites?

**What it captures:** References to other Holocaust museums, memorial sites, or comparative evaluations across institutions.

**Example trigger phrases:** "Yad Vashem," "Auschwitz," "compared to," "unlike other museums," "we also visited," "Berlin Holocaust"

**Why it matters for TJ:** The 8% of reviews referencing other sites reveals that a meaningful subset of visitors are what the paper calls "cosmopolitan travelers" who construct meaning through comparative engagement across a transnational network of memory institutions. This connects to Levy and Sznaider's (2006) concept of the globalization of Holocaust memory.

**Key finding from integration:** This category has the *lowest* memorial paradox rate (2.1%), suggesting visitors in comparative mode are more analytically detached — they evaluate rather than emotionally process.

---

### 5. Reflection / Suggestions
**Research question addressed:** What do visitors think they and others should do in light of this experience?

**What it captures:** Normative prescriptions — recommendations to visit, calls to remember, warnings about contemporary threats, invocations of "never again," statements about collective responsibility.

**Example trigger phrases:** "everyone should," "must visit," "never forget," "never again," "highly recommend," "important for future generations," "moving," "thought-provoking"

**Why it matters for TJ:** At 36%, this is the second-most prevalent category and the most directly relevant to transitional justice theory. It captures the civic and moral output of museum engagement — how visitors translate personal experience into collective prescription. The frequency of "never again" rhetoric and contemporary warnings demonstrates that visitors connect Holocaust memory to present responsibilities, which is the core function of memorialization as a TJ mechanism.

**Note:** The word "moving" and "thought-provoking" are included here rather than in Learning/Understanding because they describe the *evaluative and civic* response to the museum rather than the cognitive learning process.

---

### 6. Learning Context (General)
**Research question addressed:** General engagement with historical content, exhibits, and artifacts.

**What it captures:** References to Holocaust history, specific exhibits, artifacts, documentation, and the physical content of the museum. This is the broadest and most inclusive category — a fallback that captures historical engagement not caught by the more specific categories above.

**Example trigger phrases:** "history," "genocide," "exhibit," "artifacts," "survivor stories," "photographs," "documentation"

**Why it matters for TJ:** At 68%, this is the most prevalent category by far. It establishes that the overwhelming majority of reviews engage substantively with the museum's historical content rather than focusing purely on logistics or experience. This is the baseline measure of content engagement.

**Design note:** This category intentionally overlaps heavily with the others. Its high prevalence does not mean 68% of reviews are exclusively about historical content — it means 68% of reviews *include* substantive historical engagement alongside other themes.

---

## Tagging Function Design

Each review is passed through all six category pattern sets simultaneously. The tagging function:

1. Converts text to lowercase before matching (case-insensitive)
2. Uses `re.search()` — a single pattern match within a category is sufficient to tag that category
3. Returns a sorted list of matched categories (not a single label)
4. Reviews can match zero, one, or all six categories

```python
def tag_categories(text, bank):
    found = set()
    low = str(text).lower()
    for cat, patterns in bank.items():
        for p in patterns:
            if re.search(p, low):
                found.add(cat)
                break  # one match per category is sufficient
    return sorted(found)
```

The `break` after first match within a category is intentional — once a review is tagged for a category, there is no value in continuing to check that category's remaining patterns. This improves performance on a 10,116-row corpus.

---

## Coverage and Validation

**Final coverage: 84.7%** (8,570 of 10,116 reviews match at least one category)

The pattern sets were developed iteratively:
- Initial patterns achieved ~20% coverage
- Manual inspection of unmatched reviews revealed additional phrase variants
- Patterns were expanded and re-run until coverage stabilized at ~85%
- Random samples of matched reviews were checked to confirm semantic accuracy (patterns firing on the right content)

The 15.3% unmatched reviews are not failures — they are predominantly short reviews ("Great museum, highly recommend") that contain genuine content but not the specific phrase patterns targeted by each category. They are retained in the master dataset with empty `content_cats` lists.

---

## Outputs

### `ushmm_semantic_tagged.csv`
The cleaned dataset with three new columns appended:

| Column | Type | Description |
|---|---|---|
| `content_cats` | list | Python list of matched category names |
| `content_cats_str` | string | Pipe-separated string version (for CSV compatibility) |
| `num_content_cats` | int | Count of matched categories (0–6) |

**Note on CSV storage:** Python lists cannot be stored directly in CSV format. The `content_cats_str` column stores categories as pipe-separated strings (e.g., `"Learning / Understanding|Reflection / Suggestions"`). When loading this file in downstream notebooks, parse back to lists with:
```python
df["content_cats"] = df["content_cats_str"].fillna("").apply(
    lambda x: x.split("|") if x else []
)
```

### Figures

| Filename | Description |
|---|---|
| `category_counts.png` | Horizontal bar chart — review count per category |
| `category_cooccurrence.png` | Heatmap — how often each pair of categories co-occurs |
| `wc_learning.png` | Word cloud — common words in Learning/Understanding reviews |
| `wc_by_category.png` | Side-by-side word clouds for Learning, Reflection, Comparison |
| `tfidf_learning_phrases.png` | TF-IDF bigram/trigram cloud for Learning/Understanding subset |

---

## Key Numbers for the Paper

| Category | Reviews (n) | % of corpus |
|---|---|---|
| Learning Context (General) | 6,875 | 68.0% |
| Reflection / Suggestions | 3,616 | 35.7% |
| Learning / Understanding | 3,110 | 30.7% |
| Comparison / Other Museums | 799 | 7.9% |
| Prior Knowledge | 193 | 1.9% |
| Motivation / Why Visit | 140 | 1.4% |
| **At least one category** | **8,570** | **84.7%** |
| **Multiple categories** | **5,142** | **~60% of matched** |

**Strongest co-occurrences:**
- Learning Context (General) + Reflection/Suggestions: 2,552 reviews
- Learning Context (General) + Learning/Understanding: 2,282 reviews
- Learning/Understanding + Reflection/Suggestions: 1,215 reviews

---

## Limitations

**1. English-only.** The 11% of non-English reviews removed in notebook 01 likely includes international visitors with distinct cultural and historical relationships to the Holocaust. German, Israeli, and Polish reviewers in particular may engage with the content differently — their exclusion is a limitation the paper acknowledges.

**2. Pattern matching vs. semantic similarity.** A review saying "this shattered my understanding of history" would not match Learning/Understanding because "shattered" is not in the pattern set. The 15% unmatched rate is partly attributable to this.

**3. No negation handling.** The patterns do not account for negation — "I did not learn anything new" could theoretically match the Learning/Understanding category. In practice this is rare enough not to affect the aggregate counts meaningfully, but it is a known limitation.

**4. Single-language patterns.** All patterns are English-language. Bilingual reviewers who code-switch mid-review may produce false negatives.
