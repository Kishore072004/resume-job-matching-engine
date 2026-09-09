import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Essential paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Define category anchors for Zero-Shot Semantic Classification
CATEGORY_ANCHORS = {
    "technical": "programming language, software engineering framework, database technology, cloud computing platform, machine learning algorithm, data science, cybersecurity software",
    "soft_skill": "soft skill, communication, teamwork, leadership, interpersonal relations, adaptability, conflict resolution, emotional intelligence, time management",
    "domain": "industry specific domain knowledge, business administration, legal regulation, financial accounting, healthcare procedure, manufacturing process, hardware equipment, physical security, panels, machinery",
    "generic": "generic action, broad physical movement, simple daily task, common activity, vaguely defined concept, everyday object, abstract philosophy, similitude"
}

def main():
    logger.info("Starting ML-based Application Skill Layer creation...")
    
    skills_path = PROCESSED_DIR / "esco_skills.parquet"
    aliases_path = PROCESSED_DIR / "esco_skill_aliases.parquet"
    
    if not skills_path.exists() or not aliases_path.exists():
        logger.error(f"Required files not found in {PROCESSED_DIR}")
        return

    df_skills = pd.read_parquet(skills_path)
    df_aliases = pd.read_parquet(aliases_path)
    
    total_esco_skills = len(df_skills)
    logger.info(f"Loaded {total_esco_skills} ESCO skills.")

    logger.info("Loading SentenceTransformer model for zero-shot classification...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    categories = list(CATEGORY_ANCHORS.keys())
    anchor_texts = list(CATEGORY_ANCHORS.values())
    logger.info("Encoding category anchors...")
    anchor_embeddings = model.encode(anchor_texts, convert_to_tensor=True)
    
    logger.info("Encoding ESCO skills (label + description)...")
    # Combine label and description for richer semantic meaning
    skill_texts = (df_skills["normalized_label"].fillna("") + " " + df_skills["description"].fillna("")).tolist()
    
    # Process in batches to avoid OOM
    batch_size = 1000
    all_categories = []
    
    for i in range(0, len(skill_texts), batch_size):
        batch_texts = skill_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute cosine similarities [batch_size, num_categories]
        cos_scores = util.cos_sim(batch_embeddings, anchor_embeddings)
        
        # Get best category index
        best_idx = torch.argmax(cos_scores, dim=1).cpu().numpy()
        
        for idx in best_idx:
            all_categories.append(categories[idx])
            
        if i % 5000 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(skill_texts)} skills...")
            
    df_skills["relevance_category"] = all_categories
    
    # Priority mapping
    priority_map = {
        "technical": 5,
        "soft_skill": 2,
        "domain": 3,
        "generic": 0
    }
    df_skills["skill_priority"] = df_skills["relevance_category"].map(priority_map)
    df_skills["is_application_skill"] = df_skills["relevance_category"] != "generic"
    
    # Filter for application
    df_app_skills = df_skills[df_skills["is_application_skill"] == True].copy()
    app_skill_uris = set(df_app_skills["conceptUri"])
    
    # Filter aliases to match
    df_app_aliases = df_aliases[df_aliases["skill_uri"].isin(app_skill_uris)].copy()
    
    # Also create canonical_application_skills mapping
    df_canonical = df_app_skills[["conceptUri", "preferredLabel", "normalized_label", "relevance_category", "skill_priority"]].copy()
    
    # Save datasets
    logger.info("Saving application skill layer parquets...")
    df_skills.to_parquet(PROCESSED_DIR / "esco_relevant_skills.parquet", index=False, engine="pyarrow")
    df_app_aliases.to_parquet(PROCESSED_DIR / "esco_relevant_skill_aliases.parquet", index=False, engine="pyarrow")
    df_canonical.to_parquet(PROCESSED_DIR / "canonical_application_skills.parquet", index=False, engine="pyarrow")
    
    # Metrics
    cat_counts = df_skills["relevance_category"].value_counts().to_dict()
    
    metrics = {
        "total_esco_skills": total_esco_skills,
        "application_skills": len(df_app_skills),
        "excluded_skills": total_esco_skills - len(df_app_skills),
        "technical_skills": cat_counts.get("technical", 0),
        "professional_skills": cat_counts.get("soft_skill", 0),
        "domain_skills": cat_counts.get("domain", 0),
        "generic_skills": cat_counts.get("generic", 0),
    }
    
    logger.info("Skill Quality Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
        
    metrics_path = PROCESSED_DIR / "skill_quality_report.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"ML Application skill layer built successfully. Metrics saved to {metrics_path.name}")

if __name__ == "__main__":
    main()
