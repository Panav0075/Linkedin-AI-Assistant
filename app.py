import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from phase1_pipeline import process_document

# ==============================
# GROQ LLM SETUP
# ==============================
from groq import Groq

client = Groq(api_key="insert your API key here")

def generate_skill_advice(resume_skills, missing_skills):
    prompt = f"""
You are an expert career coach.

Candidate skills:
{list(resume_skills)}

Missing skills from target jobs:
{list(missing_skills)}

Give:
1. Clear explanation of skill gaps
2. Priority learning order
3. Simple roadmap (step by step)
4. Keep it practical for job readiness
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful career advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(page_title="AI Career Assistant", layout="wide")

st.title("🚀 AI Career Assistant")
st.markdown("Resume → Job Matching → Skill Gap → AI Career Coach")


# ==============================
# LOAD JOB DATA
# ==============================
@st.cache_data
def load_jobs():
    df = pd.read_csv("postings.csv").head(1000)
    df = df[df["description"].notna()]
    df["text"] = df["title"] + " " + df["description"].astype(str)
    return df

df = load_jobs()

# ==============================
# MODEL (cached)
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ==============================
# UPLOAD RESUME
# ==============================
uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file:

    # save file properly with extension
    suffix = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name


    # ==============================
    # PHASE 1: PARSE RESUME
    # ==============================
    resume = process_document(tmp_path)

    st.success("✅ Resume Parsed Successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Profile")
        st.write(f"**Name:** {resume['name']}")
        st.write(f"**Email:** {resume['email']}")
        st.write(f"**Experience:** {resume['experience_years']} years")

    with col2:
        st.subheader("🧠 Skills")
        st.write(", ".join(resume["skills"]))


    # ==============================
    # PHASE 2: JOB RECOMMENDATION
    # ==============================
    st.subheader("💼 Top Job Recommendations")

    resume_text = " ".join(resume["skills"]) * 2 + " " + resume["raw_text"]

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["text"])
    resume_vec = tfidf.transform([resume_text])
    tfidf_scores = cosine_similarity(resume_vec, tfidf_matrix)[0]

    # SBERT
    job_embeddings = model.encode(df["text"].tolist(), show_progress_bar=False)
    resume_embedding = model.encode([resume_text])
    sbert_scores = cosine_similarity(resume_embedding, job_embeddings)[0]

    # combine
    final_scores = 0.5 * tfidf_scores + 0.5 * sbert_scores

    df["score"] = final_scores
    top_jobs = df.sort_values("score", ascending=False).head(10)

    selected_job_skills = set()

    for _, row in top_jobs.iterrows():
        st.markdown(f"""
        ### {row['title']}
        **Company:** {row['company_name']}  
        **Location:** {row['location']}  
        **Score:** {round(row['score'], 3)}
        """)
        st.divider()

        job_words = str(row["text"]).lower().split()
        selected_job_skills.update(job_words)


    # ==============================
    # PHASE 3: SKILL GAP
    # ==============================
    st.subheader("📉 Skill Gap Analysis")

    resume_skills = set(resume["skills"])

    missing_skills = selected_job_skills - resume_skills

    missing_skills = set([s for s in missing_skills if len(s) > 3])

    if missing_skills:

        col1, col2 = st.columns(2)

        with col1:
            st.write("### 🔥 Missing Skills")
            for s in list(missing_skills)[:15]:
                st.write(f"- {s}")

        with col2:
            st.write("### 🤖 AI Career Coach (Groq)")

            advice = generate_skill_advice(resume_skills, missing_skills)
            st.markdown(advice)

    else:
        st.success("🎉 No major skill gaps detected!")

else:
    st.info("👈 Upload a resume to get started")