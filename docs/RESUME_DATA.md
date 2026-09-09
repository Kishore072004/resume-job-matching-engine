# Resume Data Lineage, Cleaning & Feature Extraction Specification

This document details the processing pipeline, feature extraction methodology, and data statistics for **STEP 3: Resume Data Cleaning and Feature Extraction**.

---

## 1. Source Dataset & Overview

- **Input Dataset**: `data/raw/resumes/Resume/Resume.csv`
- **Total Records**: 2,484 annotated candidate resumes across 24 professional categories.
- **Primary Text Column**: `Resume_str`
- **Data Isolation**: Resume processing is 100% independent of job posting data (preventing data leakage).

---

## 2. Processed Outputs & Schemas

All processed outputs are saved in Snappy-compressed Apache Parquet format under `data/processed/`:

| File Name | Primary Keys | Core Fields & Descriptions |
|---|---|---|
| `resumes_clean.parquet` | `resume_id` | `raw_text`, `clean_text`, `category` |
| `resume_quality_report.parquet` | `resume_id` | `character_count`, `word_count`, `line_count`, `quality_flag`, `is_empty`, `is_duplicate` |
| `resume_sections.parquet` | `resume_id`, `section_name` | Canonical section name (`SUMMARY`, `SKILLS`, `EXPERIENCE`, `EDUCATION`, etc.) and `section_text` |
| `resume_skills.parquet` | `resume_id`, `skill_uri` | `canonical_skill`, `matched_text`, `match_method`, `confidence` |
| `resume_experience.parquet` | `resume_id` | `years_experience`, `experience_mentions`, `confidence` |
| `resume_education.parquet` | `resume_id` | `education_level` (`PhD`, `Master`, `Bachelor`, `Associate`, `Diploma`), `education_text`, `confidence` |
| `resume_job_titles.parquet` | `resume_id`, `job_title` | Candidate job titles, `source` (`header_title`, `category`), `confidence` |
| `resume_features.parquet` | `resume_id` | Master feature table joining clean text, skills list, experience years, education level, job titles, and quality flag |
| `resume_statistics.json` | N/A | Aggregate dataset metrics and category breakdown |

---

## 3. Preprocessing & Extraction Methodology

### A. Text Cleaning & Symbol Preservation
- **Unicode Normalization**: Applied NFKC normalization and replaced non-standard unicode dashes (`\uff0d`) and non-breaking spaces (`\xa0`).
- **Whitespace & Line Breaks**: Collapsed horizontal spaces and 3+ consecutive newlines while preserving line-break structure.
- **Technical Symbol Protection**: Preserved `C++`, `C#`, `.NET`, `Node.js`, `SQL`, `AWS`, `GCP`, `Azure`, etc.
- **No Aggressive Filtering**: Retained stopwords, casing, and punctuation for natural phrase matching.

### B. Section Detection
Rule-based section segmentation mapping raw header lines to canonical section labels:
- `SUMMARY`, `SKILLS`, `EXPERIENCE`, `EDUCATION`, `PROJECTS`, `CERTIFICATIONS`, `ACHIEVEMENTS`, `LANGUAGES`.

### C. Deterministic ESCO Skill Extraction
- **Reference Database**: Matched against 101,745 normalized ESCO skill aliases from `esco_skill_aliases.parquet`.
- **Word Boundary Protection**: Prevented false-positive substring matches (e.g. avoiding matching single letter "R" inside "are").
- **Confidence Scoring**:
  - `preferred_label`: 1.0 (Exact match of ESCO preferred label)
  - `alternative_label`: 0.9 (Match of official ESCO alternative label)
  - `normalized_match`: 0.8 (Match on alias / hidden label)

### D. Experience & Education Extraction
- **Years of Experience**: Extracted numeric and word-based experience patterns (`"15+ years of experience"`, `"5 years exp"`, `"8 years work"`).
- **Education Level**: Classified highest degree achieved using hierarchical ranking: `PhD` (5) > `Master` (4) > `Bachelor` (3) > `Associate` (2) > `Diploma` (1).

---

## 4. Quality Statistics Summary

From `data/processed/resume_statistics.json`:

- **Total Resumes Processed**: 2,484
- **Valid Resumes**: 2,483 (1 raw resume was empty)
- **Duplicate Resumes**: 4
- **Average Word Count**: 811.26 words
- **Median Word Count**: 757 words
- **Total Skills Extracted**: 42,324 occurrences (1,701 unique ESCO skills)
- **Average Skills Per Resume**: 17.04 skills
- **Resumes with Extracted Experience**: 538 (21.7%)
- **Resumes with Extracted Education**: 2,344 (94.4%)
- **Resumes with Detected Skills**: 2,483 (99.96%)
- **Validation Status**: **PASS**
