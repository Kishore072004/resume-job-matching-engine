# Job Postings Data Lineage, Cleaning & Feature Extraction Specification

This document details the processing pipeline, feature extraction methodology, and quality validation for **STEP 4: Job Posting Data Cleaning and Feature Extraction**.

---

## 1. Source Datasets & Overview

- **Primary Input**: `data/raw/jobs/postings.csv` (123,849 job postings)
- **Supporting Datasets**:
  - `data/raw/jobs/jobs/job_skills.csv` (213,768 job-skill links)
  - `data/raw/jobs/mappings/skills.csv` (35 skill categories lookup table)
  - `data/raw/jobs/companies/companies.csv` (24,473 company profiles)
  - `data/raw/jobs/companies/company_industries.csv` (24,375 company-industry mappings)
- **Data Isolation**: Job preprocessing is strictly independent of resume preprocessing (0 data leakage).

---

## 2. Processed Outputs & Schemas

All processed outputs are exported in Snappy-compressed Apache Parquet format under `data/processed/`:

| File Name | Primary Keys | Core Fields & Descriptions |
|---|---|---|
| `jobs_clean.parquet` | `job_id` | `company_name`, `title`, `description`, `clean_title`, `clean_description`, `location`, `company_id`, `experience_level`, `work_type`, `remote_allowed`, `skills_description` |
| `job_quality_report.parquet` | `job_id` | `title_length`, `description_length`, `word_count`, `has_description`, `has_experience_level`, `has_skills_description`, `quality_flag` |
| `job_skills_clean.parquet` | `job_id`, `skill_abr` | Explicit job skill mapping filtered to valid postings (`job_id`, `skill_abr`, `skill_name`) |
| `job_unknown_skills.parquet` | `job_id`, `skill_abr` | Unresolved skill abbreviations (0 found) |
| `job_companies_clean.parquet` | `company_id` | Cleaned company metadata (`company_name`, `industry`, `company_size`) |
| `job_esco_skills.parquet` | `job_id`, `skill_uri` | ESCO skill extraction from text (`canonical_skill`, `matched_text`, `match_method`, `confidence`) |
| `job_skill_profile.parquet` | `job_id` | Aggregated explicit, ESCO, and combined skill counts and skill arrays |
| `job_experience.parquet` | `job_id` | Extracted experience requirement (`experience_level`, `min_years_experience`, `max_years_experience`, `experience_text`, `confidence`) |
| `job_education.parquet` | `job_id` | Education requirement (`education_requirement`, `education_text`, `confidence`) |
| `job_titles.parquet` | `job_id` | Original and normalized job title representations (`title`, `normalized_title`) |
| `job_features.parquet` | `job_id` | Master job feature table combining title, clean description, location, experience min/max years, education, skills, and quality flag |
| `job_statistics.json` | N/A | Aggregate dataset metrics, distributions, and skill counts |

---

## 3. Preprocessing & Extraction Methodology

### A. Text Cleaning & Symbol Protection
- **Unicode Normalization**: Applied NFKC normalization and replaced non-standard unicode characters.
- **Whitespace & HTML**: Stripped HTML tags/markup and collapsed whitespace while preserving sentence structure.
- **Technical Symbol Protection**: Retained technical terms and tokens (`C++`, `C#`, `.NET`, `Node.js`, `CI/CD`, `SQL`, `AWS`, `Azure`, `GCP`).
- **No Silently Dropped Records**: Retained all 123,849 job postings (including 7 jobs with missing descriptions marked with `quality_flag = 'missing_description'`).

### B. Skill Processing Strategy
- **Explicit Skills**: Matched 205,778 explicit job-skill relationships from `job_skills.csv` to standard names via `skills.csv`.
- **ESCO Text Skill Extraction**: Extracted 1,278,489 ESCO skill matches from `clean_title`, `clean_description`, and `skills_description`.
- **Combined Profiles**: Produced unified skill arrays (`explicit_skills`, `esco_skills`, `combined_skills`) per posting.

### C. Experience & Education Extraction
- **Experience Years**: Parsed explicit numerical year ranges (e.g. `3+ years`, `5-7 years`) into `min_years_experience` and `max_years_experience` while maintaining qualitative experience levels (`Entry level`, `Mid-Senior level`, `Executive`, `Director`).
- **Education Requirements**: Classified required degree levels (`PhD`, `Master`, `Bachelor`, `Associate`, `Diploma`, `Not Specified`).

---

## 4. Quality Statistics Summary

From `data/processed/job_statistics.json`:

- **Total Job Postings Processed**: 123,849
- **Valid Jobs**: 123,842 (7 jobs missing text descriptions flagged)
- **Explicit Job Skill Links**: 205,778
- **ESCO Skills Extracted**: 1,278,489
- **Jobs with Min/Max Years Experience Extracted**: 52,812
- **Jobs with Education Requirements Extracted**: 87,353
- **Unique Companies**: 24,473
- **Unique Job Titles**: 87,902
- **Validation Status**: **PASS**
