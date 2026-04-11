import re
import spacy
import os
import json
from pdfminer.high_level import extract_text as pdf_extract
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel
from typing import List, Optional

nlp = spacy.load("en_core_web_sm")

# ---------------- FILE LOADER ----------------
def load_document(file_path):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        text = pdf_extract(file_path)
    elif ext == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported format")

    return text


# ---------------- CLEAN ----------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


# ---------------- EXTRACT ----------------
def extract_name(text):
    doc = nlp(text[:200])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Unknown"


def extract_email(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return m.group() if m else None


def extract_phone(text):
    m = re.search(r'(\+?\d{1,3}[-.\s]?)?\d{10}', text)
    return m.group() if m else None


SKILLS = [
    "python","sql","aws","docker","kafka","machine learning",
    "deep learning","excel","tableau","power bi","communication"
]


def extract_skills(text):
    text = text.lower()
    return list(set([s for s in SKILLS if s in text]))


def extract_experience(text):
    m = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())
    return max([int(x[0]) for x in m]) if m else None


def extract_education(text):
    edu = ["bachelor","master","phd","b.tech","m.tech"]
    text = text.lower()
    return [e for e in edu if e in text]


# ---------------- MAIN ----------------
def process_document(file_path):
    text = load_document(file_path)
    text = clean_text(text)

    data = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_experience(text),
        "education": extract_education(text),
        "raw_text": text
    }

    return data