import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Resume-Job Match Human Annotator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
EVAL_PAIRS_PATH = ROOT_DIR / "data" / "processed" / "human_evaluation_pairs.parquet"

@st.cache_data(ttl=1)
def load_data():
    if not EVAL_PAIRS_PATH.exists():
        st.error(f"Dataset file not found at {EVAL_PAIRS_PATH}. Please run `python scripts/create_human_eval_set.py` first.")
        st.stop()
    return pd.read_parquet(EVAL_PAIRS_PATH)

def save_data(df: pd.DataFrame):
    df.to_parquet(EVAL_PAIRS_PATH, index=False)
    st.cache_data.clear()

def main():
    st.title("🎯 Resume-to-Job Matching: Human Evaluation Tool")
    st.caption("Label candidate pairs for independent ground truth evaluation.")

    df = load_data()

    # Sidebar setup
    st.sidebar.header("⚙️ Annotator Settings")
    annotator_role = st.sidebar.selectbox(
        "Annotator Selection",
        ["Annotator 1 (human_label_1)", "Annotator 2 (human_label_2)", "Primary Annotator (human_label)"]
    )
    
    col_mapping = {
        "Annotator 1 (human_label_1)": "human_label_1",
        "Annotator 2 (human_label_2)": "human_label_2",
        "Primary Annotator (human_label)": "human_label"
    }
    label_col = col_mapping[annotator_role]

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters & Navigation")

    strata_filter = st.sidebar.selectbox(
        "Filter by Strata",
        ["All"] + sorted(df["strata_group"].unique().tolist())
    )

    status_filter = st.sidebar.selectbox(
        "Filter by Annotation Status",
        ["All", "Unannotated Only", "Annotated Only"]
    )

    filtered_df = df.copy()
    if strata_filter != "All":
        filtered_df = filtered_df[filtered_df["strata_group"] == strata_filter]

    if status_filter == "Unannotated Only":
        filtered_df = filtered_df[filtered_df[label_col].isna()]
    elif status_filter == "Annotated Only":
        filtered_df = filtered_df[filtered_df[label_col].notna()]

    if len(filtered_df) == 0:
        st.warning("No candidate pairs match the current filter criteria.")
        st.stop()

    total_pairs = len(df)
    annotated_count = df[label_col].notna().sum()
    progress = annotated_count / total_pairs if total_pairs > 0 else 0

    st.sidebar.markdown("### 📊 Progress Summary")
    st.sidebar.progress(progress)
    st.sidebar.metric("Annotated Pairs", f"{annotated_count} / {total_pairs} ({progress*100:.1f}%)")

    # Select pair
    pair_ids = filtered_df["review_id"].tolist()
    
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Ensure index in bounds
    if st.session_state.current_index >= len(pair_ids):
        st.session_state.current_index = 0

    selected_id = st.sidebar.selectbox(
        "Jump to Pair ID",
        pair_ids,
        index=st.session_state.current_index
    )
    
    # Update current index from selectbox selection
    st.session_state.current_index = pair_ids.index(selected_id)
    idx = df[df["review_id"] == selected_id].index[0]
    row = df.loc[idx]

    # Main area pair header
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        st.subheader(f"🆔 {row['review_id']}")
        st.caption(f"Strata: `{row['strata_group']}`")
    with c2:
        st.subheader(f"👤 Category")
        st.write(f"**{row['resume_category']}**")
    with c3:
        st.subheader(f"💼 Job Title")
        st.write(f"**{row['job_title']}**")
    with c4:
        st.subheader(f"🏢 Company / Loc")
        st.write(f"{row['company_name']} | {row['location']}")

    st.markdown("---")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Semantic Similarity", f"{row['semantic_similarity']:.3f}")
    m2.metric("Skill Jaccard", f"{row['skill_jaccard']:.3f}")
    m3.metric("Resume Coverage", f"{row['skill_coverage_resume']:.1%}")
    m4.metric("Job Coverage", f"{row['skill_coverage_job']:.1%}")
    m5.metric("Weak Label", str(row['weak_label']).upper())

    st.markdown("---")

    # Skills comparison
    try:
        r_skills = json.loads(row['resume_skills'])
        j_skills = json.loads(row['job_skills'])
        s_skills = json.loads(row['shared_skills'])
    except Exception:
        r_skills, j_skills, s_skills = [], [], []

    with st.expander("🛠️ Extracted Skills Comparison", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"**Resume Skills ({len(r_skills)})**")
            st.write(", ".join([f"`{s}`" for s in r_skills]) if r_skills else "None extracted")
        with sc2:
            st.markdown(f"**Job Skills ({len(j_skills)})**")
            st.write(", ".join([f"`{s}`" for s in j_skills]) if j_skills else "None extracted")
        with sc3:
            st.markdown(f"**Shared Skills ({len(s_skills)})**")
            st.write(", ".join([f"**`{s}`**" for s in s_skills]) if s_skills else "No overlap")

    st.markdown("---")

    # Side-by-side text
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### 📄 Resume Content Snippet")
        st.text_area("Resume Text", value=row["resume_text_snippet"], height=320, disabled=True, label_visibility="collapsed")
    with t2:
        st.markdown("### 📋 Job Description Snippet")
        st.text_area("Job Description", value=row["job_description_snippet"], height=320, disabled=True, label_visibility="collapsed")

    st.markdown("---")

    # Annotation controls
    st.subheader("📝 Annotation Input")

    current_label_val = row[label_col]
    label_opts = ["Good Match (1)", "Poor Match (0)", "Ambiguous / Borderline (Null)"]
    
    default_opt_idx = 2
    if current_label_val == 1.0 or current_label_val == 1:
        default_opt_idx = 0
    elif current_label_val == 0.0 or current_label_val == 0:
        default_opt_idx = 1

    ac1, ac2 = st.columns([2, 3])
    with ac1:
        decision = st.radio("Match Quality Decision", label_opts, index=default_opt_idx, key=f"radio_{selected_id}")
        confidence = st.select_slider("Confidence Level", options=[1, 2, 3, 4, 5], value=int(row["human_confidence"]) if pd.notna(row["human_confidence"]) else 4)

    with ac2:
        notes = st.text_area("Annotation Rationale / Notes", value=str(row["human_notes"]) if pd.notna(row["human_notes"]) else "", height=120)

    btn_c1, btn_c2, btn_c3 = st.columns([2, 2, 4])

    with btn_c1:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()

    with btn_c2:
        if st.button("💾 Save & Next ➡️", type="primary", use_container_width=True):
            if "Good Match" in decision:
                val = 1.0
            elif "Poor Match" in decision:
                val = 0.0
            else:
                val = None

            df.loc[idx, label_col] = val
            df.loc[idx, "human_confidence"] = float(confidence)
            df.loc[idx, "human_notes"] = notes
            
            save_data(df)
            st.toast(f"Saved annotation for {selected_id}!", icon="✅")
            
            if st.session_state.current_index < len(pair_ids) - 1:
                st.session_state.current_index += 1
            st.rerun()

if __name__ == "__main__":
    main()
