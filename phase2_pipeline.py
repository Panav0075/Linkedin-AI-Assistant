import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- LOAD JOBS ----------------
def load_jobs():
    df = pd.read_csv("postings.csv").head(1000)
    df = df[df["description"].notna()]
    df["text"] = df["title"] + " " + df["description"].astype(str)
    return df


# ---------------- RECOMMEND ----------------
def recommend_jobs(resume, df):
    resume_text = " ".join(resume["skills"]) + " " + resume["raw_text"]

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["text"])
    resume_vec = tfidf.transform([resume_text])
    tfidf_scores = cosine_similarity(resume_vec, tfidf_matrix)[0]

    # SBERT
    job_emb = model.encode(df["text"].tolist(), show_progress_bar=False)
    res_emb = model.encode([resume_text])
    sbert_scores = cosine_similarity(res_emb, job_emb)[0]

    # combine
    scores = 0.5 * tfidf_scores + 0.5 * sbert_scores

    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(10)

    return top