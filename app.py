import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import re
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from phase1_pipeline import process_document
from groq import Groq

# ── PAGE CONFIG — MUST BE FIRST STREAMLIT CALL ──
st.set_page_config(page_title="CareerLens AI", layout="wide", initial_sidebar_state="collapsed")

# ── API KEYS ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")

client = Groq(api_key=GROQ_API_KEY)

# ── ADZUNA LIVE JOB FETCH ──
def fetch_live_jobs(query, location="us", results=20):
    url = f"https://api.adzuna.com/v1/api/jobs/{location}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": results,
        "what": query,
        "content-type": "application/json"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        jobs = []
        for job in data.get("results", []):
            jobs.append({
                "title": job.get("title", ""),
                "company_name": job.get("company", {}).get("display_name", "Unknown"),
                "location": job.get("location", {}).get("display_name", ""),
                "description": job.get("description", ""),
                "url": job.get("redirect_url", ""),
                "salary_min": job.get("salary_min", None),
                "salary_max": job.get("salary_max", None),
            })
        return pd.DataFrame(jobs)
    except Exception as e:
        st.error(f"Failed to fetch jobs: {e}")
        return pd.DataFrame()

# ── SKILL GAP AI ──
def generate_skill_advice(resume_skills, missing_skills):
    prompt = f"""
You are an expert career coach.
Candidate skills: {list(resume_skills)}
Missing skills from target jobs: {list(missing_skills)}
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

# ── KNOWN SKILLS ──
KNOWN_SKILLS = [
    "python","java","javascript","typescript","c++","c#","r","scala","go","rust","swift","kotlin","php","ruby","bash",
    "html","css","react","angular","vue","node.js","express","django","flask","fastapi","spring","rest api","graphql","bootstrap","tailwind","jquery","redux",
    "machine learning","deep learning","artificial intelligence","nlp","computer vision",
    "scikit-learn","tensorflow","pytorch","keras","xgboost","lightgbm","hugging face","transformers","llm","langchain","rag",
    "feature engineering","statistical analysis","time series","regression","classification","clustering",
    "data science","data analysis","data analytics","data engineering","data visualization","data mining","etl","a/b testing",
    "pandas","numpy","matplotlib","seaborn","plotly","scipy","pyspark","apache spark","kafka","airflow","dbt","databricks","snowflake","dask",
    "sql","mysql","postgresql","sqlite","oracle","mongodb","redis","cassandra","dynamodb","firebase","elasticsearch","bigquery","redshift",
    "aws","azure","gcp","google cloud","lambda","ec2","s3","sagemaker",
    "docker","kubernetes","jenkins","ci/cd","terraform","ansible","git","github","linux","devops","mlops","microservices",
    "power bi","tableau","looker","excel","grafana","d3.js",
    "openai","prompt engineering","fine-tuning","embeddings","pinecone","chromadb","faiss",
    "rpa","uipath","power automate",
    "agile","scrum","jira","project management","product management",
    "salesforce","sap","google analytics","seo","crm",
    "communication","leadership","problem solving","teamwork","attention to detail",
    "cybersecurity","networking","api development","web scraping","selenium","test automation",
    "matlab","sas","spss","spotfire","jupyter",
]

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Cabinet+Grotesk:wght@300;400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'Cabinet Grotesk',sans-serif;background:#050508;color:#f0eef8;}
.stApp{background:#050508}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 2.5rem 5rem 2.5rem;max-width:1280px}

.hero-wrap{padding:5rem 0 3.5rem;position:relative;text-align:center;display:flex;flex-direction:column;align-items:center;}
.hero-wrap::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:700px;height:350px;background:radial-gradient(ellipse,rgba(124,111,255,.12) 0%,transparent 70%);pointer-events:none;}
.hero-tag{font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:#7c6fff;border:1px solid rgba(124,111,255,.3);padding:.3rem .9rem;border-radius:4px;display:inline-block;margin-bottom:1.8rem;}
.hero-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(3.5rem,9vw,7.5rem);line-height:.95;letter-spacing:.04em;color:#fff;margin-bottom:1.2rem;}
.hero-title span{background:linear-gradient(90deg,#7c6fff,#ff6fd8,#7c6fff);background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:shimmer 4s linear infinite;}
@keyframes shimmer{0%{background-position:0%}100%{background-position:200%}}
.hero-sub{font-size:1.05rem;color:#6b6880;font-weight:400;max-width:500px;line-height:1.7;margin-bottom:2rem;text-align:center;}
.steps-row{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;justify-content:center;}
.step-pill{display:flex;align-items:center;gap:.5rem;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:100px;padding:.4rem 1rem;}
.step-pill.active{background:rgba(124,111,255,.15);border-color:rgba(124,111,255,.35);}
.step-num{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#7c6fff;}
.step-name{font-size:.8rem;font-weight:600;color:#c5c0d8}
.step-arrow{color:rgba(255,255,255,.15);font-size:.9rem}

.slash-divider{height:2px;background:linear-gradient(90deg,#7c6fff,#ff6fd8,transparent);margin:1rem 0 2.5rem;border-radius:4px;}

.upload-label{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:.06em;color:#fff;margin-bottom:.5rem;display:block;}
.upload-hint{font-size:.8rem;color:#6b6880;margin-bottom:1rem;font-family:'JetBrains Mono',monospace}

/* Search box */
.search-wrap{background:rgba(124,111,255,.06);border:1px solid rgba(124,111,255,.2);border-radius:16px;padding:1.5rem 1.8rem;margin:1.5rem 0;}
.search-label{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:.04em;color:#fff;margin-bottom:.3rem;}
.search-hint{font-size:.78rem;color:#6b6880;font-family:'JetBrains Mono',monospace;margin-bottom:.8rem;}
.live-badge{display:inline-flex;align-items:center;gap:.4rem;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);border-radius:100px;padding:.25rem .8rem;font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#6ee7b7;margin-left:.8rem;}
.live-dot{width:6px;height:6px;border-radius:50%;background:#10b981;animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.profile-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin:1.5rem 0}
.pcard{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:1.4rem;position:relative;overflow:hidden;transition:border-color .2s;}
.pcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#7c6fff,#ff6fd8);}
.pcard:hover{border-color:rgba(124,111,255,.3)}
.pcard-label{font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#7c6fff;margin-bottom:.6rem;}
.pcard-val{font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#fff;line-height:1}
.pcard-sub{font-size:.75rem;color:#6b6880;margin-top:.2rem}

.tag-wrap{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.8rem}
.tag{font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:.25rem .7rem;border-radius:4px;background:rgba(124,111,255,.12);border:1px solid rgba(124,111,255,.2);color:#a99fff;}
.tag.miss{background:rgba(255,111,216,.1);border-color:rgba(255,111,216,.25);color:#ff9fe8;}

.sec-head{display:flex;align-items:center;gap:1rem;margin:3rem 0 1.2rem}
.sec-num{font-family:'Bebas Neue',sans-serif;font-size:3rem;color:rgba(124,111,255,.2);line-height:1;}
.sec-title{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:.04em;color:#fff}
.sec-sub{font-size:.8rem;color:#6b6880;font-family:'JetBrains Mono',monospace}

.job-list{display:flex;flex-direction:column;gap:.6rem}
.jcard{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1rem 1.3rem;display:grid;grid-template-columns:40px 1fr auto;align-items:center;gap:1rem;transition:all .2s ease;}
.jcard:hover{background:rgba(124,111,255,.07);border-color:rgba(124,111,255,.25);transform:translateX(4px);}
.jcard-rank{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:rgba(124,111,255,.3);text-align:center;}
.jcard-title{font-size:.95rem;font-weight:700;color:#f0eef8;margin-bottom:.15rem}
.jcard-meta{font-size:.75rem;color:#6b6880;font-family:'JetBrains Mono',monospace}
.jcard-right{display:flex;flex-direction:column;align-items:flex-end;gap:.4rem}
.jcard-score{background:rgba(124,111,255,.15);border:1px solid rgba(124,111,255,.25);border-radius:6px;padding:.25rem .7rem;font-family:'JetBrains Mono',monospace;font-size:.75rem;font-weight:500;color:#a99fff;white-space:nowrap;}
.jcard-salary{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#6b6880;}
.apply-btn{display:inline-block;background:rgba(124,111,255,.15);border:1px solid rgba(124,111,255,.35);border-radius:6px;padding:.28rem .85rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:500;color:#a99fff;text-decoration:none;white-space:nowrap;transition:all .2s;}
.apply-btn:hover{background:rgba(124,111,255,.35);color:#fff;transform:translateY(-1px);}

.gap-grid{display:grid;grid-template-columns:1fr 1.5fr;gap:1.5rem;margin-top:1rem}
.gap-box{background:rgba(255,111,216,.05);border:1px solid rgba(255,111,216,.15);border-radius:14px;padding:1.5rem;}
.gap-box-title{font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.15em;text-transform:uppercase;color:#ff6fd8;margin-bottom:1rem;}
.gap-stat{font-family:'Bebas Neue',sans-serif;font-size:3.5rem;color:#ff6fd8;line-height:1;margin-bottom:.2rem;}
.gap-stat-sub{font-size:.8rem;color:#6b6880}

.coach-trigger-wrap{background:rgba(124,111,255,.05);border:1px solid rgba(124,111,255,.2);border-radius:16px;padding:1.8rem;}
.coach-badge{font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#7c6fff;border:1px solid rgba(124,111,255,.3);padding:.25rem .7rem;border-radius:4px;}
.coach-title{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:#fff;letter-spacing:.04em;margin:.6rem 0 .4rem;}
.coach-desc{font-size:.82rem;color:#6b6880;margin-bottom:1.3rem;line-height:1.6}
.coach-response{background:rgba(255,255,255,.03);border:1px solid rgba(124,111,255,.15);border-radius:12px;padding:1.5rem;margin-top:1rem;border-left:3px solid #7c6fff;}
.coach-response-head{display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#7c6fff;}

.success-bar{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);border-radius:10px;padding:.9rem 1.2rem;display:flex;align-items:center;gap:.7rem;color:#6ee7b7;font-size:.85rem;font-weight:500;margin-bottom:2rem;font-family:'JetBrains Mono',monospace;}
.empty-state{background:rgba(255,255,255,.02);border:1px dashed rgba(255,255,255,.08);border-radius:16px;padding:4rem 2rem;text-align:center;color:#3d3a52;}
.empty-state .e-icon{font-size:3rem;margin-bottom:1rem;display:block}
.empty-state .e-title{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:#4a4762;letter-spacing:.06em}
.empty-state .e-sub{font-size:.82rem;color:#3d3a52;font-family:'JetBrains Mono',monospace;margin-top:.4rem}

.stFileUploader>div{background:rgba(124,111,255,.04) !important;border:1.5px dashed rgba(124,111,255,.35) !important;border-radius:14px !important;padding:2rem !important;}
div[data-testid="stButton"] button{background:linear-gradient(135deg,#7c6fff,#a855f7) !important;color:#fff !important;border:none !important;border-radius:8px !important;font-family:'Cabinet Grotesk',sans-serif !important;font-weight:700 !important;font-size:.9rem !important;padding:.6rem 1.6rem !important;transition:all .2s !important;box-shadow:0 4px 20px rgba(124,111,255,.3) !important;}
div[data-testid="stButton"] button:hover{transform:translateY(-2px) !important;box-shadow:0 8px 30px rgba(124,111,255,.5) !important;}
</style>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">// Live Job Search · AI-Powered · Career Intelligence</div>
    <div class="hero-title">CAREER &nbsp;<span>LENS</span>&nbsp; AI</div>
    <p class="hero-sub">Upload your resume. Search live jobs. Discover what skills to learn next.</p>
    <div class="steps-row">
        <div class="step-pill"><span class="step-num">01</span>&nbsp;<span class="step-name">Parse Resume</span></div>
        <div class="step-arrow">&#8594;</div>
        <div class="step-pill"><span class="step-num">02</span>&nbsp;<span class="step-name">Live Job Search</span></div>
        <div class="step-arrow">&#8594;</div>
        <div class="step-pill"><span class="step-num">03</span>&nbsp;<span class="step-name">Gap Analysis</span></div>
        <div class="step-arrow">&#8594;</div>
        <div class="step-pill active"><span class="step-num">04</span>&nbsp;<span class="step-name">AI Coach</span></div>
    </div>
</div>
<div class="slash-divider"></div>
""", unsafe_allow_html=True)


# ── UPLOAD ──
st.markdown('<span class="upload-label">DROP YOUR RESUME</span>', unsafe_allow_html=True)
st.markdown('<span class="upload-hint">// PDF · DOCX · TXT — processed locally, never stored</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf", "docx", "txt"], label_visibility="collapsed")

if uploaded_file:
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Parsing resume..."):
        resume = process_document(tmp_path)

    st.markdown(f"""
    <div class="success-bar">
        ✓ &nbsp; Parsed — <strong>{uploaded_file.name}</strong> &nbsp;·&nbsp; Ready to search live jobs
    </div>
    """, unsafe_allow_html=True)

    # ── PROFILE ──
    st.markdown("""
    <div class="sec-head">
        <div class="sec-num">01</div>
        <div>
            <div class="sec-title">YOUR PROFILE</div>
            <div class="sec-sub">// extracted from resume</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    skill_tags = "".join([f'<span class="tag">{s}</span>' for s in resume["skills"][:24]])
    st.markdown(f"""
    <div class="profile-grid">
        <div class="pcard">
            <div class="pcard-label">Candidate</div>
            <div style="font-size:1.2rem;font-weight:800;color:#fff;margin-bottom:.2rem">{resume['name']}</div>
            <div style="font-size:.78rem;color:#6b6880;font-family:'JetBrains Mono',monospace">{resume['email']}</div>
        </div>
        <div class="pcard">
            <div class="pcard-label">Experience</div>
            <div class="pcard-val">{resume['experience_years']}</div>
            <div class="pcard-sub">years detected</div>
        </div>
        <div class="pcard">
            <div class="pcard-label">Skills Found</div>
            <div class="pcard-val">{len(resume['skills'])}</div>
            <div class="pcard-sub">unique skills</div>
        </div>
    </div>
    <div class="pcard" style="margin-bottom:1.5rem">
        <div class="pcard-label">Skill Inventory</div>
        <div class="tag-wrap">{skill_tags}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="slash-divider"></div>', unsafe_allow_html=True)

    # ── LIVE JOB SEARCH ──
    st.markdown("""
    <div class="sec-head">
        <div class="sec-num">02</div>
        <div>
            <div class="sec-title">LIVE JOB SEARCH <span class="live-badge"><span class="live-dot"></span> LIVE</span></div>
            <div class="sec-sub">// powered by Adzuna · real-time job postings</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-suggest query from resume skills
    top_skills = resume["skills"][:3] if resume["skills"] else ["data science"]
    default_query = " ".join(top_skills[:2])

    st.markdown("""
    <div class="search-wrap">
        <div class="search-label">Search Live Jobs</div>
        <div class="search-hint">// enter job title or skills — we'll match them to your resume</div>
    </div>
    """, unsafe_allow_html=True)

    col_q, col_loc, col_btn = st.columns([3, 1.5, 1])
    with col_q:
        job_query = st.text_input("", value=default_query, placeholder="e.g. data scientist, python developer...", label_visibility="collapsed")
    with col_loc:
        location = st.selectbox("", ["us", "gb", "au", "ca", "de", "fr"], label_visibility="collapsed",
                                format_func=lambda x: {"us":"🇺🇸 USA","gb":"🇬🇧 UK","au":"🇦🇺 Australia","ca":"🇨🇦 Canada","de":"🇩🇪 Germany","fr":"🇫🇷 France"}[x])
    with col_btn:
        search_clicked = st.button("🔍 Search Jobs", use_container_width=True)

    if search_clicked or "live_jobs" not in st.session_state:
        with st.spinner(f"Fetching live jobs for '{job_query}'..."):
            live_df = fetch_live_jobs(job_query, location=location, results=20)
            st.session_state["live_jobs"] = live_df
            st.session_state["job_query"] = job_query

    live_df = st.session_state.get("live_jobs", pd.DataFrame())

    if not live_df.empty:
        # Score jobs against resume
        live_df = live_df[live_df["description"].notna() & (live_df["description"] != "")]
        live_df["text"] = live_df["title"] + " " + live_df["description"].astype(str)

        resume_text = " ".join(resume["skills"]) * 2 + " " + resume["raw_text"]

        with st.spinner("Matching jobs to your profile..."):
            tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
            all_texts = live_df["text"].tolist() + [resume_text]
            tfidf.fit(all_texts)
            tfidf_matrix = tfidf.transform(live_df["text"].tolist())
            resume_vec = tfidf.transform([resume_text])
            tfidf_scores = cosine_similarity(resume_vec, tfidf_matrix)[0]

            job_embeddings = model.encode(live_df["text"].tolist(), show_progress_bar=False)
            resume_embedding = model.encode([resume_text])
            sbert_scores = cosine_similarity(resume_embedding, job_embeddings)[0]

            final_scores = 0.5 * tfidf_scores + 0.5 * sbert_scores
            live_df["score"] = final_scores
            top_jobs = live_df.sort_values("score", ascending=False).head(10)

        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#6b6880;margin-bottom:1rem;">// found {len(live_df)} live jobs · showing top 10 matches</div>', unsafe_allow_html=True)

        selected_job_skills = set()
        st.markdown('<div class="job-list">', unsafe_allow_html=True)
        for i, (_, row) in enumerate(top_jobs.iterrows()):
            score_pct = int(round(row["score"] * 100))
            company = row.get("company_name", "Unknown")
            location_str = row.get("location", "")
            job_url = row.get("url", "#")
            sal_min = row.get("salary_min")
            sal_max = row.get("salary_max")
            salary_str = ""
            if sal_min and sal_max:
                salary_str = f"${int(sal_min):,} – ${int(sal_max):,}"
            elif sal_min:
                salary_str = f"From ${int(sal_min):,}"

            st.markdown(f"""
            <div class="jcard">
                <div class="jcard-rank">#{i+1}</div>
                <div>
                    <div class="jcard-title">{row['title']}</div>
                    <div class="jcard-meta">{company} &nbsp;·&nbsp; {location_str}</div>
                </div>
                <div class="jcard-right">
                    <div class="jcard-score">{score_pct}% match</div>
                    {f'<div class="jcard-salary">{salary_str}</div>' if salary_str else ''}
                    <a href="{job_url}" target="_blank" class="apply-btn">View Job ↗</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

            job_text_lower = str(row["text"]).lower()
            for skill in KNOWN_SKILLS:
                if re.search(r'\b' + re.escape(skill) + r'\b', job_text_lower):
                    selected_job_skills.add(skill)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="slash-divider" style="margin-top:2rem"></div>', unsafe_allow_html=True)

        # ── SKILL GAP ──
        resume_skills = set(resume["skills"])
        missing_skills = selected_job_skills - resume_skills
        top_missing = sorted(list(missing_skills))[:15]

        st.markdown("""
        <div class="sec-head">
            <div class="sec-num">03</div>
            <div>
                <div class="sec-title">SKILL GAP ANALYSIS</div>
                <div class="sec-sub">// what live job postings need that your resume is missing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if top_missing:
            miss_tags = "".join([f'<span class="tag miss">{s}</span>' for s in top_missing])
            st.markdown(f"""
            <div class="gap-grid">
                <div class="gap-box">
                    <div class="gap-box-title">// Missing Skills</div>
                    <div class="gap-stat">{len(top_missing)}</div>
                    <div class="gap-stat-sub">priority skills to acquire</div>
                    <div class="tag-wrap" style="margin-top:1.2rem">{miss_tags}</div>
                </div>
                <div class="coach-trigger-wrap">
                    <span class="coach-badge">// AI · Groq · LLaMA</span>
                    <div class="coach-title">GET YOUR CAREER ROADMAP</div>
                    <div class="coach-desc">
                        Our AI career coach will analyze your skill gaps based on <strong>live job postings</strong>
                        and generate a personalized, step-by-step learning roadmap to make you job-ready.
                    </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 3])
            with col1:
                generate_clicked = st.button("⚡ Generate Roadmap", key="coach_btn", use_container_width=True)

            st.markdown('</div></div>', unsafe_allow_html=True)

            if generate_clicked:
                with st.spinner("AI Coach is analyzing your profile against live jobs..."):
                    advice = generate_skill_advice(resume_skills, set(top_missing))
                st.markdown("""
                <div class="coach-response">
                    <div class="coach-response-head"><span>▶</span> AI CAREER COACH RESPONSE</div>
                """, unsafe_allow_html=True)
                st.markdown(advice)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(124,111,255,.04);border:1px dashed rgba(124,111,255,.2);
                    border-radius:12px;padding:2rem;text-align:center;margin-top:1rem;color:#4a4762;
                    font-family:'JetBrains Mono',monospace;font-size:.78rem;">
                    ↑ &nbsp; Click the button to generate your personalized AI roadmap
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-bar" style="justify-content:center;font-size:.95rem;padding:1.5rem">
                🎉 &nbsp; No major skill gaps — you are a strong match for these live job postings!
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <span class="e-icon">📡</span>
            <div class="e-title">NO JOBS FOUND</div>
            <div class="e-sub">// try a different search query or check your API keys</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <span class="e-icon">◈</span>
        <div class="e-title">AWAITING RESUME</div>
        <div class="e-sub">// Drop your PDF, DOCX, or TXT above to begin</div>
    </div>
    """, unsafe_allow_html=True)
