# Human Annotation Guidelines: Resume-to-Job Matching Evaluation

## 1. Overview & Purpose

This document provides standardized instructions for human annotators evaluating resume-to-job posting candidate pairs. Ground truth annotations collected using these guidelines form an independent, benchmark evaluation set (`data/processed/human_evaluation_pairs.parquet`) to assess the true real-world generalization of the AI Resume-to-Job Matching engine.

Weak labels derived during heuristic candidate generation are subject to rule bias and feature leakage. Human annotations serve as the gold standard for unbiased performance evaluation.

---

## 2. Evaluation Schema

Each candidate pair is evaluated on three dimensions:
1. **Match Label (`human_label`)**: Binary classification of overall suitability.
2. **Confidence Score (`human_confidence`)**: 1 to 5 rating scale.
3. **Qualitative Notes (`human_notes`)**: Brief rationale for edge cases or ambiguities.

### Label Definitions

| Label Value | Category | Definition | Criteria |
| :--- | :--- | :--- | :--- |
| **`1`** | **Good Match** | Candidate is qualified and viable for the job role. | • Candidate possesses >= 60% of critical core skills required by the job.<br>• Resume role history aligns logically with job domain.<br>• Candidate experience level is within acceptable range (e.g. ±2 years of requirement). |
| **`0`** | **Poor Match** | Candidate is unqualified or fundamentally mismatched. | • Lacks critical domain skill mandatory for role (e.g. Accounting candidate for Senior DevOps Engineer role).<br>• Significant domain mismatch.<br>• Extreme experience deficit (>3 years under minimum required experience). |
| **`null` / `-1`** | **Ambiguous / Borderline** | Insufficient information or borderline case. | • Resume text is severely truncated, corrupt, or vague.<br>• Job description lacks specific requirements.<br>• Highly transferable skills where fit depends heavily on unstated context. |

---

## 3. Specific Evaluation Protocols

### A. Core Skills vs. Secondary Skills
- **Core / Essential Skills**: Mandatory technologies or methodologies explicitly required for the role (e.g., Python/Django for a Backend Python Engineer). Missing core skills usually results in a **0 (Poor Match)**.
- **Secondary / Preferred Skills**: Nice-to-have frameworks, soft skills, or domain exposure. Missing secondary skills does NOT disqualify a candidate if core skills are present (**1 (Good Match)**).

### B. Title vs. Skill Overlap
- **Title Mismatch with Skill Fit**: If a candidate was titled "Technical Analyst" but demonstrates 80% Python/SQL/AWS skills required for "Data Engineer", label as **1 (Good Match)**. Skills take precedence over job title phrasing.
- **Title Match with Skill Deficit**: If job title matches (e.g., "Software Developer") but candidate only knows HTML/CSS and job requires Rust/C++, label as **0 (Poor Match)**.

### C. Experience Level Expectations
- **Junior Role (0-2 yrs)**: Candidate with 3-5 years is acceptable (**1**).
- **Senior Role (5+ yrs)**: Candidate with 1 year experience is a **0 (Poor Match)** unless exceptional skill match is documented.

---

## 4. Multi-Annotator Agreement & Cohen's Kappa

To maintain high dataset reliability:
1. A subset of candidate pairs must be annotated independently by two distinct annotators (`human_label_1` and `human_label_2`).
2. Inter-annotator agreement is quantified using **Cohen's Kappa ($\kappa$)**:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Where:
- $P_o$: Observed proportional agreement among annotators.
- $P_e$: Expected probability of random agreement.

**Target Benchmark**: $\kappa \ge 0.70$ (Substantial Agreement). Any pair with conflicting labels ($1$ vs $0$) undergoes expert adjudication.

---

## 5. Streamlit Annotation Utility Usage

Annotators use the interactive tool at `scripts/annotate_pairs.py`:

```bash
streamlit run scripts/annotate_pairs.py
```

### Workflow Steps:
1. Select **Annotator ID** (Annotator 1 or Annotator 2).
2. Review candidate text, extracted resume skills, job skills, shared skills, and similarity metrics.
3. Select label: **Good Match (1)**, **Poor Match (0)**, or **Ambiguous (Null)**.
4. Set Confidence rating (1-5).
5. Click **Save & Next**. Annotations persist directly to parquet storage.
