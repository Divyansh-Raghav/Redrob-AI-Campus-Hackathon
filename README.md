# Resume Matching Engine
### Redrob AI Campus Hackathon — Individual Competition

---

## Problem Statement

Build a program that matches 10 Indian university student resumes against 3 Job Descriptions from Korean tech companies (Kakao, Naver, Line) using TF-IDF and Cosine Similarity — using **pure Python only** (no external libraries).

---

## Project Structure

```
resume_matching_engine.py   # Main solution file
README.md                   # This file
```

---

## How It Works — Step by Step

### Step 1: Skill Normalization
- Split raw skill strings on **commas only**
- Lowercase and strip each token
- Apply exact `SKILL_ALIASES` dictionary mapping
- Discard tokens not found in the alias map
- Handles typos: `Pyhton → python`, `kubernates → kubernetes`, `JavaScrpit → javascript`

### Step 2: Deduplication
- Each canonical skill appears only once per resume
- Order is preserved during deduplication

### Step 3: Vocabulary Construction
- Collect all unique skills across all 10 resumes
- Sort alphabetically
- Same vocabulary order used for all resume and JD vectors

### Step 4: TF-IDF Vectors (Resumes only)
```
TF(skill, resume)  = 1 / N
IDF(skill)         = ln( 10 / df(skill) )
TF-IDF             = TF × IDF
```
Where:
- `N` = total unique skills in the resume
- `df(skill)` = number of resumes containing that skill
- Natural logarithm, no smoothing

### Step 5: JD Binary Vectors
- Normalize JD skills using the same alias mapping
- `1` if skill present in JD, `0` if absent
- Built over the same shared vocabulary

### Step 6: Cosine Similarity + Ranking
```
Cosine(A, B) = (A · B) / (|A| × |B|)
```
Where `A` = Resume TF-IDF vector, `B` = JD binary vector

- Top 3 candidates ranked per JD
- Ties broken alphabetically by candidate name

---

## Final Output

```
JD-1 — Kakao (Seoul) (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Seongnam) (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Seoul) (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## Requirements

| Requirement | Detail |
|---|---|
| Language | Python 3.x |
| External Libraries | ❌ None allowed |
| Standard Library Used | `math` (for `log` and `sqrt`) |
| AI Tool Used | Redrob AI |

---

## How to Run

```bash
python3 resume_matching_engine.py
```

No installations needed. Works out of the box with any Python 3.x interpreter.

---

## Key Design Decisions

- **Comma-only splitting** preserves multi-word phrases like `"feature engineering"` and `"spring boot"` as single tokens — no regex or space-splitting needed
- **Exact alias lookup** handles all typos and casing variations in one dictionary pass
- **No MULTI_WORD_KEYS preprocessing** — kept simple and compliant with hackathon rules
- **Vocabulary built from resumes only** — JD skills are not added to the vocab per problem spec

---

## Author

**Hackathon:** Redrob AI Campus Hackathon — Powered by McKinley Rice
**Competition Type:** Individual
**Allowed AI Tool:** Redrob AI only
