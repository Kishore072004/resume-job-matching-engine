# Production Inference Engine Documentation

The **Inference Engine** provides a clean, unified Python API and command-line interfaces for predicting resume-to-job match scores, extracting matched and missing skills, evaluating experience/education compatibility, and explaining prediction outputs using SHAP feature attributions.

---

## Architecture Overview

To guarantee zero **training-serving skew**, the inference engine wraps single resume and job posting inputs into 1-row DataFrames and runs them through the **exact same preprocessing and feature engineering pipelines** used during model training:

1. **Resume Processing** (`src/inference/resume_processor.py`):
   - Text cleaning & section detection (`src/preprocessing/resume_cleaner.py`).
   - ESCO skill extraction & alias mapping (`canonical_skills.parquet` & `canonical_skill_aliases.parquet`).
   - Experience (years) and education level extraction.
2. **Job Processing** (`src/inference/job_processor.py`):
   - Job title & description cleaning (`src/preprocessing/job_cleaner.py`).
   - Explicit & ESCO skill profile extraction.
   - Experience & education requirements extraction.
3. **Feature Generation & Enrichment** (`src/inference/engine.py`):
   - Semantic embeddings computation using `sentence-transformers/all-MiniLM-L6-v2`.
   - Feature enrichment (`src/feature_engineering/ml_features.py`).
4. **Prediction & Calibration**:
   - Imputation using the trained `SimpleImputer`.
   - Raw log-odds prediction from Optuna-tuned XGBoost model (`job_matcher_final.joblib`).
   - Probability calibration via Platt scaling logistic regression.
   - Rescaling to an estimated match score range from `0` to `100`.
5. **Explainability**:
   - Local feature attribution using `shap.TreeExplainer`.

---

## Python API Usage

```python
from src.inference.engine import JobMatchEngine

# Initialize engine (loads model artifact & sentence transformer)
engine = JobMatchEngine()

# Predict match
result = engine.predict_match(
    resume_text="Software Engineer with 6 years experience in Python, Django, Docker, AWS...",
    job_title="Senior Python Developer",
    job_description="Looking for Senior Python Engineer with Django, Docker, AWS experience...",
    category="Information-Technology"
)

print(f"Match Score: {result['estimated_match_score']}/100")
print(f"Match Decision: {result['is_match']}")
print(f"Matched Skills: {result['skills_analysis']['matched_skills']}")
print(f"Missing Skills: {result['skills_analysis']['missing_skills']}")
print(f"Top Features: {result['top_contributing_features']}")
```

---

## Command Line Usage

### Single Prediction CLI

Run predictions directly from text arguments or text files:

```bash
python scripts/predict_match.py \
  --job-title "Senior Python Engineer" \
  --job-desc "Looking for Senior Python Developer with Django and PostgreSQL..." \
  --resume-text "Software engineer with 5 years experience in Python, Django, PostgreSQL..."
```

Optionally pass file paths:

```bash
python scripts/predict_match.py \
  --job-title "Senior Backend Developer" \
  --job-file path/to/job_description.txt \
  --resume-file path/to/resume.txt \
  --output path/to/result.json
```

### Batch Prediction CLI

Predict match scores for a dataset of resume-job candidate pairs:

```bash
python scripts/batch_predict.py \
  --input-file data/sample_pairs.csv \
  --output-file data/predictions.csv
```

Required columns in input file:
- `resume_text`
- `job_title`
- `job_description`

Output columns added:
- `estimated_match_score`
- `is_match`
- `estimated_probability`
- `matched_skills`
- `missing_skills`

---

## Response Payload Schema

```json
{
  "estimated_match_score": 88.5,
  "is_match": true,
  "estimated_probability": 0.885,
  "raw_model_probability": 0.842,
  "decision_threshold": 0.10,
  "skills_analysis": {
    "matched_skills": ["python", "django", "postgresql", "docker"],
    "missing_skills": ["aws", "kubernetes"],
    "shared_skill_count": 4,
    "resume_skill_count": 6,
    "job_skill_count": 6,
    "skill_jaccard": 0.5,
    "job_skill_coverage": 0.6667
  },
  "compatibility": {
    "experience_match": 1.0,
    "education_match": true,
    "experience_gap_years": 1.0,
    "resume_years_experience": 6.0,
    "job_min_years_experience": 5.0
  },
  "top_contributing_features": [
    {
      "feature": "semantic_similarity",
      "shap_value": 0.384,
      "feature_value": 0.825
    },
    {
      "feature": "shared_skill_count",
      "shap_value": 0.215,
      "feature_value": 4.0
    }
  ],
  "metadata": {
    "model_name": "Optuna_Trained_XGBoost_Resume_Matcher",
    "model_version": "2.0.0"
  }
}
```

---

## Validation

Run the validation suite to test the inference engine end-to-end:

```bash
python scripts/validate_inference.py
```
