import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.engine import JobMatchEngine

def run_validation():
    print("Initializing JobMatchEngine with Application Skill Layer...")
    engine = JobMatchEngine()
    
    resume_text = "Experienced in Python, SQL, AWS, Docker, Machine Learning, and Pandas."
    job_desc = "Seeking engineer skilled in Python, SQL, AWS, Docker, Kubernetes, Natural Language Processing, and Statistics."
    job_title = "Machine Learning Engineer"
    
    print("Running inference...")
    result = engine.predict_match(
        resume_text=resume_text,
        job_title=job_title,
        job_description=job_desc,
        category="Engineering"
    )
    
    skills = result.get("skills_analysis", {})
    m_group = skills.get("matched_grouped", {})
    mis_group = skills.get("missing_grouped", {})
    
    matched_tech = m_group.get("technical", [])
    missing_tech = mis_group.get("technical", [])
    
    print("\nExpected Match Concepts: Python, SQL, AWS, Docker, Machine Learning")
    print(f"Actual Matched Technical: {matched_tech}")
    
    print("\nExpected Missing Concepts: Kubernetes, Natural Language Processing, Statistics")
    print(f"Actual Missing Technical: {missing_tech}")
    
    all_extracted = matched_tech + missing_tech + m_group.get("professional", []) + mis_group.get("professional", [])
    
    fail = False
    
    if "similitude" in all_extracted or "security panels" in all_extracted:
        print("FAIL: Found noise concepts in extraction.")
        fail = True
    else:
        print("Noise filtering: PASS")
        
    expected_matches = ["python", "sql", "machine learning"]
    for e in expected_matches:
        if not any(e in s.lower() for s in matched_tech):
            print(f"FAIL: Expected matched skill {e} not found.")
            fail = True
            
    expected_missing = ["natural language", "statistics"]
    
    missing_all = missing_tech + mis_group.get("professional", []) + mis_group.get("domain", [])
    for e in expected_missing:
        if not any(e in s.lower() for s in missing_all):
            print(f"FAIL: Expected missing skill {e} not found.")
            fail = True

    if not fail:
        print("\nSKILL QUALITY UPDATE COMPLETE")
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")

if __name__ == "__main__":
    run_validation()
