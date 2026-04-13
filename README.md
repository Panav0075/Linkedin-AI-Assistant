
## System Architecture

### 1. Resume Parser (`phase1_pipeline.py`)
**The NLP Layer.** Extracts entities from unstructured documents.
* **Input:** PDF/Docx Resume.
* **Core Logic:** Uses **spaCy** and custom NLP patterns to identify key entities.
* **Output:** Structured JSON (Name, Email, Skills, Experience, Education).

### 2. Job Recommender (`phase2_pipeline.py`)
**The ML Layer.** Ranks jobs from a `jobs.csv` database using a hybrid approach.
* **TF-IDF:** Calculates keyword-based similarity.
* **SBERT:** Utilizes Sentence-BERT for deep semantic matching (understanding context beyond keywords).
* **Output:** Top 10 ranked job recommendations.

### 3. Intelligence Dashboard (`app.py`)
**The Product Layer.** A **Streamlit** UI that orchestrates the entire pipeline.
* **Integration:** Calls Phase 1 for parsing and Phase 2 for recommendations.
* **Groq LLM:** Performs a real-time **Skill Gap Analysis** and generates personalized career advice based on the matched jobs.

---

## Project Structure

```text
├── app.py                # Streamlit UI & Orchestration
├── phase1_pipeline.py    # NLP Resume Parsing Logic
├── phase2_pipeline.py    # ML Job Recommendation Logic
├── requirements.txt      # Project Dependencies
└── jobs.csv              # Job Listings Dataset (link below)


TO start :

jobs.csv from : https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

then,
pip install -r requirements.txt
python -m spacy download en_core_web_sm

and GROQ_API_KEY=your_key_here ( replace with yours )
to create one visit : https://console.groq.com/keys


last : streamlit run app.py

## 🎥 Demo Video

Click below to watch:



https://github.com/user-attachments/assets/835f34a3-61b4-4517-8271-1648e0fc5549



https://github.com/user-attachments/assets/79f2e3fa-9e6a-4fa5-b13f-7867e8da7025


