"""
Streamlit Web UI Dashboard for AI Resume-to-Job Matching Engine.
"""

import os
import requests
import streamlit as st

# Configuration
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide",
)

def parse_uploaded_file(uploaded_file) -> str:
    """Extract plain text from uploaded file (.txt, .pdf, .docx)."""
    if uploaded_file is None:
        return ""
    
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
                return ""
        elif filename.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(uploaded_file)
                return "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading DOCX file: {e}")
                return ""
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return ""

st.title("📄 Resume to Job Matcher")
st.write("Analyze how well a candidate's resume fits a job description using our AI engine.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    category = st.selectbox(
        "Category",
        options=["Information-Technology", "Engineering", "Healthcare", "Finance", "Sales", "Human-Resources", "General"],
        index=0
    )
    experience_level = st.selectbox(
        "Experience Level",
        options=["Not Specified", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"],
        index=3
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Candidate Resume")
    uploaded_file = st.file_uploader("Upload File (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    
    default_text = ""
    if uploaded_file is not None:
        default_text = parse_uploaded_file(uploaded_file)
        if default_text:
            st.success("File uploaded successfully!")

    resume_text = st.text_area("Resume Text", value=default_text, height=300)

with col2:
    st.subheader("Job Description")
    job_title = st.text_input("Job Title", placeholder="Software Engineer")
    job_desc = st.text_area("Job Requirements", height=300)

@st.cache_resource
def get_standalone_engine():
    """Cache and load standalone JobMatchEngine for direct inference fallback."""
    import sys
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.inference.engine import JobMatchEngine
    return JobMatchEngine()

if st.button("Analyze Match", type="primary"):
    if not resume_text.strip() or not job_title.strip() or not job_desc.strip():
        st.warning("Please fill in all the required fields.")
    else:
        with st.spinner("Analyzing match... This may take a moment."):
            payload = {
                "resume_text": resume_text,
                "job_title": job_title,
                "job_description": job_desc,
                "category": category,
                "experience_level": experience_level,
            }
            
            data = None
            used_fallback = False

            # Try API endpoint first
            try:
                endpoint = f"{api_url.rstrip('/')}/api/v1/match"
                response = requests.post(endpoint, json=payload, timeout=5)
                if response.status_code == 200:
                    data = response.json()
            except Exception:
                # If API is unreachable (e.g. hosted on Streamlit Cloud without separate backend), fall back to in-process ML engine
                used_fallback = True

            if data is None:
                try:
                    engine = get_standalone_engine()
                    data = engine.predict_match(
                        resume_text=resume_text,
                        job_title=job_title,
                        job_description=job_desc,
                        category=category,
                        experience_level=experience_level,
                    )
                except Exception as ex:
                    st.error(f"Inference error: {ex}")

            if data:
                if used_fallback:
                    st.caption("ℹ️ *Running in direct ML model mode (standalone inference engine).*")
                st.success("Analysis complete!")
                st.divider()
                
                score = data.get("estimated_match_score", 0)
                is_match = data.get("is_match", False)
                prob = data.get("estimated_probability", 0.0)

                c1, c2, c3 = st.columns(3)
                c1.metric("Match Score", f"{score}/100", "Match" if is_match else "No Match")
                c2.metric("Confidence", f"{prob*100:.1f}%")
                c3.metric("Experience Match", "Yes" if data.get("compatibility", {}).get("experience_match") == 1.0 else "No")
                
                st.progress(min(1.0, max(0.0, score / 100.0)))
                
                tab1, tab2, tab3 = st.tabs(["Skills Breakdown", "SHAP Features", "Raw Response"])
                
                with tab1:
                    sk = data.get("skills_analysis", {})
                    col_sk1, col_sk2 = st.columns(2)
                    
                    m_group = sk.get("matched_grouped", {})
                    mis_group = sk.get("missing_grouped", {})
                    m_flat = sk.get("matched_skills", [])
                    mis_flat = sk.get("missing_skills", [])
                    
                    with col_sk1:
                        st.subheader("Matched Skills")
                        has_matched = False
                        if m_group.get("technical"):
                            st.markdown("**Technical Skills:**")
                            for s in m_group["technical"]: st.markdown(f"- ✅ **{s}**")
                            has_matched = True
                        if m_group.get("professional"):
                            st.markdown("**Professional / Soft Skills:**")
                            for s in m_group["professional"]: st.markdown(f"- 🤝 **{s}**")
                            has_matched = True
                        if m_group.get("domain"):
                            st.markdown("**Domain / Industry Skills:**")
                            for s in m_group["domain"]: st.markdown(f"- 🏢 **{s}**")
                            has_matched = True
                        if m_group.get("generic"):
                            st.markdown("**General / Other Skills:**")
                            for s in m_group["generic"]: st.markdown(f"- 💡 **{s}**")
                            has_matched = True
                            
                        if not has_matched:
                            if m_flat:
                                st.markdown("**All Matched Skills:**")
                                for s in m_flat: st.markdown(f"- ✅ **{s}**")
                            else:
                                st.info("No matching ESCO skills detected in text.")
                                
                    with col_sk2:
                        st.subheader("Missing Skills (Skill Gap)")
                        has_missing = False
                        if mis_group.get("technical"):
                            st.markdown("**Technical Skills:**")
                            for s in mis_group["technical"]: st.markdown(f"- ❌ **{s}**")
                            has_missing = True
                        if mis_group.get("professional"):
                            st.markdown("**Professional / Soft Skills:**")
                            for s in mis_group["professional"]: st.markdown(f"- ❌ **{s}**")
                            has_missing = True
                        if mis_group.get("domain"):
                            st.markdown("**Domain / Industry Skills:**")
                            for s in mis_group["domain"]: st.markdown(f"- ❌ **{s}**")
                            has_missing = True
                        if mis_group.get("generic"):
                            st.markdown("**General / Other Skills:**")
                            for s in mis_group["generic"]: st.markdown(f"- ❌ **{s}**")
                            has_missing = True
                            
                        if not has_missing:
                            if mis_flat:
                                st.markdown("**All Missing Skills:**")
                                for s in mis_flat: st.markdown(f"- ❌ **{s}**")
                            else:
                                st.success("No skill gap detected! Candidate satisfies all required job skills.")
                            
                with tab2:
                    shap_feats = data.get("top_contributing_features", [])
                    if shap_feats:
                        st.dataframe(shap_feats)
                    else:
                        st.info("No SHAP features available.")
                        
                with tab3:
                    st.json(data)
