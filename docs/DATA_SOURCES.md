# Data Sources & Discovery Inventory

This document details all raw datasets discovered within `data/raw/` for the **AI Resume-to-Job Matching & Skill Gap Intelligence Engine**.

---

## 1. Overview of Raw Datasets

| Source | File Count | Primary Purpose | Key Datasets |
|---|---|---|---|
| **ESCO** | 19 CSVs | Standardized European taxonomy of occupations, skills, competencies, and relationships. | `occupations_en.csv`, `skills_en.csv`, `occupationSkillRelations_en.csv` |
| **Jobs** | 11 CSVs | Real-world job postings, company metadata, required skills, and compensation. | `postings.csv`, `job_skills.csv`, `companies.csv` |
| **O*NET** | 1 CSV | Occupational knowledge element importance and proficiency level scores. | `knowledge.csv` |
| **Resumes** | 1 CSV + 2,484 PDFs | Annotated candidate resumes across 24 job categories. | `Resume.csv`, raw PDF files in `data/data/` |

---

## 2. Discovered File Inventory & Descriptions

### A. ESCO Datasets (`data/raw/esco/`)

| File Name | Rows | Columns | Purpose |
|---|---|---|---|
| `occupations_en.csv` | 3,043 | 15 | Profiles of standardized European occupations, titles, alternative labels, and descriptions. |
| `skills_en.csv` | 13,960 | 13 | Taxonomy of skills, knowledge, competencies, and reuse levels. |
| `occupationSkillRelations_en.csv` | 126,051 | 6 | Mappings connecting occupations to essential and optional skills/competencies. |
| `skillSkillRelations_en.csv` | 5,818 | 5 | Relationships between related skills (hierarchical / association). |
| `skillGroups_en.csv` | 640 | 11 | Categorization and grouping structure of skills. |
| `ISCOGroups_en.csv` | 619 | 8 | International Standard Classification of Occupations (ISCO-08) hierarchy. |
| `skillsHierarchy_en.csv` | 640 | 14 | Hierarchical classification tree for skills and competencies. |
| `digitalSkillsCollection_en.csv` | 1,284 | 10 | Subset of digital skills and ICT competencies. |
| `greenSkillsCollection_en.csv` | 629 | 10 | Subset of green, environmental, and sustainability skills. |
| `transversalSkillsCollection_en.csv` | 95 | 10 | General/transversal soft skills and core abilities. |
| `languageSkillsCollection_en.csv` | 359 | 10 | Foreign language proficiency and communication skills. |
| `researchSkillsCollection_en.csv` | 40 | 10 | Academic and industrial research skill competencies. |
| `digCompSkillsCollection_en.csv` | 25 | 10 | Digital Competence Framework skills. |
| `greenShareOcc_en.csv` | 3,590 | 5 | Proportion of green tasks per occupation. |
| `researchOccupationsCollection_en.csv` | 122 | 8 | Occupations focused on scientific research. |
| `broaderRelationsOccPillar_en.csv` | 3,648 | 6 | Broad parent-child relations across occupations. |
| `broaderRelationsSkillPillar_en.csv` | 20,819 | 6 | Broad parent-child relations across skill nodes. |
| `conceptSchemes_en.csv` | 20 | 7 | Concept schemes and metadata definitions. |
| `dictionary_en.csv` | 160 | 4 | Terms and definitions dictionary. |

### B. Jobs & Postings Datasets (`data/raw/jobs/`)

| Relative File Path | Rows | Columns | Purpose |
|---|---|---|---|
| `data/raw/jobs/postings.csv` | 123,849 | 31 | Primary LinkedIn job postings dataset containing title, description, location, salary range, and work type. |
| `data/raw/jobs/jobs/job_skills.csv` | 213,768 | 2 | Skill tag links associated with each job posting ID. |
| `data/raw/jobs/jobs/job_industries.csv` | 164,808 | 2 | Industry classifications associated with job postings. |
| `data/raw/jobs/jobs/salaries.csv` | 40,785 | 8 | Detailed compensation breakdowns (base salary, pay period, currency). |
| `data/raw/jobs/jobs/benefits.csv` | 67,943 | 3 | Employee benefits offered per job posting (health insurance, 401k, etc.). |
| `data/raw/jobs/companies/companies.csv` | 24,473 | 10 | Profiles of hiring companies (name, size, location, description, URL). |
| `data/raw/jobs/companies/company_industries.csv` | 24,375 | 2 | Industry sector mappings for companies. |
| `data/raw/jobs/companies/company_specialities.csv` | 169,387 | 2 | Company core specialties and functional areas. |
| `data/raw/jobs/companies/employee_counts.csv` | 35,787 | 4 | Historical employee counts and company size metrics. |
| `data/raw/jobs/mappings/industries.csv` | 422 | 2 | Standard lookup table mapping industry IDs to industry names. |
| `data/raw/jobs/mappings/skills.csv` | 35 | 2 | Standard lookup table mapping skill IDs to skill names. |

### C. O*NET Dataset (`data/raw/onet/`)

| File Name | Rows | Columns | Purpose |
|---|---|---|---|
| `knowledge.csv` | 60,060 | 15 | O*NET standardized ratings (Importance `IM`, Level `LV`) for knowledge domains across SOC occupational codes. |

### D. Resumes Dataset (`data/raw/resumes/`)

| File Name / Location | Rows / Count | Columns | Purpose |
|---|---|---|---|
| `data/raw/resumes/Resume/Resume.csv` | 2,484 | 4 | Structured resume records with unique ID, plain text (`Resume_str`), HTML markup (`Resume_html`), and 24 job categories. |
| `data/raw/resumes/data/data/<CATEGORY>/*.pdf` | 2,484 PDFs | N/A | Raw PDF files corresponding to candidate resumes across 24 job category folders. |

---

## 3. Data Integrity & Verification Summary

- **Total Tabular Files Discovered**: 32 files
- **Inventory File Location**: `data/dataset_inventory.csv`
- **Data Modification**: 0 raw files modified (inspection only).
