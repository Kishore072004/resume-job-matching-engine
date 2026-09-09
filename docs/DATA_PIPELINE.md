# AI Resume-to-Job Matching & Skill Gap Intelligence Engine
## Data Pipeline Architecture

This document describes the end-to-end data pipeline stages, data flow, and file artifacts.

---

```
  ┌────────────────┐       ┌────────────────────────┐       ┌───────────────────────────┐       ┌───────────────────────────┐
  │  RAW DATASETS  │ ────> │  STEP 2: ESCO ENGINE   │ ────> │  STEP 3: RESUME ENGINE    │ ────> │   STEP 4: JOB ENGINE      │
  │   (data/raw/)  │       │ (esco_cleaner / Py)    │       │ (resume_cleaner / Py)     │       │   (job_cleaner / Py)      │
  └────────────────┘       └────────────────────────┘       └───────────────────────────┘       └───────────────────────────┘
                                       │                                  │                                   │
                                       ▼                                  ▼                                   ▼
                           data/processed/esco_*.parquet       data/processed/resume_*.parquet     data/processed/job_*.parquet
```

---

## Stage 1: Dataset Discovery & Inventory
- **Script**: `scripts/inspect_datasets.py`
- **Output**: `data/dataset_inventory.csv`, `docs/DATA_SOURCES.md`
- **Description**: Discovers and scans raw CSV/TSV/Excel/Parquet files without modifying raw assets.

---

## Stage 2: ESCO Standardization & Taxonomy Processing
- **Scripts**: `src/preprocessing/esco_cleaner.py`, `scripts/process_esco.py`, `scripts/validate_esco.py`
- **Outputs**:
  - `data/processed/esco_skills.parquet` (13,939 skills)
  - `data/processed/esco_skill_aliases.parquet` (101,745 aliases)
  - `data/processed/esco_occupations.parquet` (3,039 occupations)
  - `data/processed/esco_occupation_skill_relations.parquet` (126,051 relationships)
  - `data/processed/esco_occupation_skill_profile.parquet` (3,039 profiles)
  - `data/processed/esco_data_quality.json`
  - `docs/ESCO_DATA.md`
- **Validation**: Schema integrity, 0 duplicate relationships, 0 broken foreign keys, 0 empty labels.

---

## Stage 3: Resume Cleaning & Feature Extraction
- **Scripts**: `src/preprocessing/resume_cleaner.py`, `scripts/process_resumes.py`, `scripts/validate_resumes.py`
- **Outputs**:
  - `data/processed/resumes_clean.parquet` (2,484 clean resumes)
  - `data/processed/resume_quality_report.parquet` (character/word counts & quality flags)
  - `data/processed/resume_sections.parquet` (rule-based section segmentation)
  - `data/processed/resume_skills.parquet` (42,324 detected skill matches mapped to ESCO URIs)
  - `data/processed/resume_experience.parquet` (regex-extracted years of experience)
  - `data/processed/resume_education.parquet` (extracted degree levels: PhD, Master, Bachelor, Associate, Diploma)
  - `data/processed/resume_job_titles.parquet` (candidate job titles)
  - `data/processed/resume_features.parquet` (master resume feature table)
  - `data/processed/resume_statistics.json`
  - `docs/RESUME_DATA.md`
- **Validation**: `resume_id` uniqueness, valid skill URIs, non-negative experience, valid education levels, 0 duplicate resume-skill pairs.
- **Data Isolation**: Resume processing is strictly independent of job postings.

---

## Stage 4: Job Postings Preprocessing & Taxonomy Mapping
- **Scripts**: `src/preprocessing/job_cleaner.py`, `scripts/process_jobs.py`, `scripts/validate_jobs.py`
- **Outputs**:
  - `data/processed/jobs_clean.parquet` (123,849 job postings)
  - `data/processed/job_quality_report.parquet` (text length metrics & quality flags)
  - `data/processed/job_skills_clean.parquet` (205,778 explicit job skills)
  - `data/processed/job_unknown_skills.parquet` (0 unknown skill abbreviations)
  - `data/processed/job_companies_clean.parquet` (24,473 company metadata profiles)
  - `data/processed/job_esco_skills.parquet` (1,278,489 ESCO skill matches extracted from text)
  - `data/processed/job_skill_profile.parquet` (combined explicit + ESCO skill profiles)
  - `data/processed/job_experience.parquet` (min/max years of experience extracted)
  - `data/processed/job_education.parquet` (required education degree levels)
  - `data/processed/job_titles.parquet` (normalized job titles)
  - `data/processed/job_features.parquet` (master job feature table)
  - `data/processed/job_statistics.json`
  - `docs/JOB_DATA.md`
- **Validation**: `job_id` uniqueness, valid explicit skill mappings, valid ESCO skill URIs, non-empty clean titles, description quality, 0 duplicate job-skill pairs.
- **Data Isolation**: Job processing is strictly independent of resume processing (0 data leakage).

---

## Upcoming Pipeline Stages
- **Stage 5**: Skill Gap Intelligence & Similarity Feature Engineering
- **Stage 6**: Matching Engine Model & Evaluation
