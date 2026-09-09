# ESCO Data Lineage & Processing Specification

This document details the data lineage, cleaning, normalization, and quality validation operations performed during **STEP 2: ESCO Data Cleaning and Normalization**.

---

## 1. Input Datasets & Purpose

All source files were loaded directly from `data/raw/esco/` without modifying raw assets:

| Source File | Format | Rows (Raw) | Purpose & Usage |
|---|---|---|---|
| `skills_en.csv` | CSV | 13,960 | Primary reference taxonomy of 13,939 unique European skills, competencies, and knowledge concepts. Provides canonical labels, alternate labels, hidden labels, and concept URIs. |
| `occupations_en.csv` | CSV | 3,043 | Taxonomy of 3,039 unique European occupations, ISCO-08 occupational groups, labels, definitions, and classification codes. |
| `occupationSkillRelations_en.csv` | CSV | 126,051 | Relationships linking occupations to essential/optional skills and knowledge/competence types. |

---

## 2. Key Columns & Retained Fields

### A. ESCO Skills (`data/processed/esco_skills.parquet`)
- **`conceptUri`**: Permanent unique identifier for the skill concept.
- **`skillType`**: Classification as `skill/competence` or `knowledge`.
- **`reuseLevel`**: Reusability classification (`sector-specific`, `cross-sector`, `transversal`, etc.).
- **`preferredLabel`**: Canonical human-readable label in original title case/casing.
- **`altLabels`**: Alternative terminology and synonyms (cleaned and normalized).
- **`hiddenLabels`**: Outdated/misspelled terms or search keywords.
- **`status`**: Concept status (e.g. `released`).
- **`description`**: Detailed textual definition of the skill.
- **`normalized_label`** *(New)*: Lowercased, whitespace-collapsed version of `preferredLabel` for exact string/semantic matching.

### B. Skill Aliases Table (`data/processed/esco_skill_aliases.parquet`)
- **`skill_uri`**: Foreign key to `esco_skills.conceptUri`.
- **`canonical_skill`**: Original preferred label of the skill concept.
- **`alias`**: Raw alias string.
- **`alias_normalized`**: Lowercased and whitespace-normalized alias text.
- **`alias_type`**: Source type of the alias (`preferred`, `alternative`, or `hidden`).

### C. ESCO Occupations (`data/processed/esco_occupations.parquet`)
- **`conceptUri`**: Permanent unique identifier for the occupation concept.
- **`iscoGroup`**: 4-digit ISCO-08 group classification code.
- **`preferredLabel`**: Official title of the occupation in original casing.
- **`altLabels`**, **`hiddenLabels`**: Alternate titles and search terms.
- **`status`**, **`modifiedDate`**, **`scopeNote`**, **`definition`**, **`description`**, **`code`**, **`naceCode`**: Administrative and industry metadata.
- **`normalized_label`** *(New)*: Lowercased, whitespace-collapsed version of `preferredLabel`.

### D. Occupation-Skill Relations (`data/processed/esco_occupation_skill_relations.parquet`)
- **`occupationUri`**: Foreign key pointing to `esco_occupations.conceptUri`.
- **`occupationLabel`**: Title of the occupation.
- **`relationType`**: Relationship classification (`essential` vs `optional`).
- **`skillType`**: Type of associated concept (`skill/competence` vs `knowledge`).
- **`skillUri`**: Foreign key pointing to `esco_skills.conceptUri`.
- **`skillLabel`**: Title of the associated skill.

### E. Occupation Skill Profile (`data/processed/esco_occupation_skill_profile.parquet`)
- **`occupation_uri`**: Foreign key pointing to occupation concept URI.
- **`occupation_label`**: Preferred title of the occupation.
- **`essential_skill_count`**: Count of essential skill relations required.
- **`optional_skill_count`**: Count of optional skill relations.
- **`knowledge_count`**: Count of knowledge-based skill relations.
- **`competence_count`**: Count of competence-based skill relations.
- **`total_skill_count`**: Aggregate count of associated skills.
- **`skills`**: Array (`list[str]`) of normalized canonical skill labels.

---

## 3. Data Cleaning & Normalization Operations

1. **Whitespace & Line Break Normalization**:
   - Stripped leading and trailing whitespace.
   - Replaced carriage returns (`\r`), newlines (`\n`), and tab characters (`\t`) with single spaces.
   - Collapsed consecutive whitespace sequences (`\s+`) into single spaces.
2. **Text Casing Preservation vs Normalization**:
   - Original `preferredLabel`, `altLabels`, and `hiddenLabels` preserve exact original casing to avoid destroying proper nouns or acronyms.
   - `normalized_label` and `alias_normalized` columns are converted to lowercase specifically for downstream indexing and matching.
3. **Skill Aliases Splitting**:
   - Extracted 101,745 distinct skill alias mappings from `preferredLabel`, `altLabels` (split by linebreaks), and `hiddenLabels`.
4. **Missing Values**:
   - Converted missing string fields (`NaN`) to empty strings (`""`) without dropping records.
   - Retained 59 relation rows with missing `skillType` after reporting them in the quality log.
5. **Parquet Storage Format**:
   - Exported all processed datasets using Snappy-compressed Apache Parquet format via `pyarrow`.

---

## 4. Key Assumptions

- **ESCO Canonical Authority**: Skills and aliases provided by ESCO are assumed to be curated canonical references; no external semantic merging or clustering was performed.
- **URI Primary Keys**: `conceptUri` serves as the immutable primary key for skills and occupations.
- **Foreign Key Integrity**: All relation records in `occupationSkillRelations_en.csv` map to valid `occupationUri` and `skillUri` entities.

---

## 5. Validation Results & Quality Report

Running `scripts/validate_esco.py` verified the following metrics:

- **Total Processed Skills**: 13,939 unique concepts (`data/processed/esco_skills.parquet`)
- **Total Skill Aliases**: 101,745 rows (`data/processed/esco_skill_aliases.parquet`)
- **Total Processed Occupations**: 3,039 unique concepts (`data/processed/esco_occupations.parquet`)
- **Total Occupation-Skill Relations**: 126,051 relationships (`data/processed/esco_occupation_skill_relations.parquet`)
  - **Essential Relationships**: 67,600 (53.6%)
  - **Optional Relationships**: 58,451 (46.4%)
- **Occupation Skill Profiles**: 3,039 profiles generated (`data/processed/esco_occupation_skill_profile.parquet`)
- **URI Foreign Key Validation**: 0 invalid occupation or skill URI references.
- **Duplicate Constraints**: 0 duplicate relationships found.
- **Validation Status**: **PASS**
