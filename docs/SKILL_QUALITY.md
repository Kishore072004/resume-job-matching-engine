# Skill Quality Management

This project uses a pure Machine Learning approach to evaluate and categorize skills from the ESCO ontology to ensure high-quality job matching. By moving away from hardcoded heuristics and regex rules, the system leverages zero-shot semantic classification to adapt and scale natively.

## The Application Skill Layer

The raw ESCO ontology contains nearly 14,000 skills, many of which are broad generic actions (e.g., "stand", "use computer") or non-technical jargon ("similitude", "security panels"). Including these dilutes the value of technical skill matching.

We construct a **Relevant Skill Layer** that:
1. Filters out generic/low-value actions.
2. Categorizes remaining skills into: `technical`, `soft_skill`, and `domain`.

### Zero-Shot Semantic Classification

Instead of maintaining brittle keyword lists, we classify skills using the `SentenceTransformer("all-MiniLM-L6-v2")` model. 

1. **Anchors**: We define broad semantic anchors for each target category (e.g., `technical` = "technical programming language, software engineering framework, database technology...").
2. **Embeddings**: We compute embeddings for the semantic anchors and all ESCO skills (combining their label and description).
3. **Similarity**: We calculate the cosine similarity between each skill and the anchors.
4. **Classification**: The skill is assigned to the category with the highest similarity score.

### Metrics

After processing, we save a report to `data/processed/skill_quality_report.json`.
A healthy execution maintains roughly ~1,500 technical skills, ~2,500 professional/soft skills, and ~7,500 domain skills, while successfully filtering out ~2,000+ generic concepts.

## Validation

Run `scripts/validate_skill_quality.py` to ensure core concepts like "Python", "SQL", and "Machine Learning" route correctly to the `technical` category, while preventing noise like "similitude" from appearing in extracted skills.

```bash
python scripts/validate_skill_quality.py
```
