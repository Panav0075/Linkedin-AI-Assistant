import re
import spacy
import os
from pdfminer.high_level import extract_text as pdf_extract
import docx

nlp = spacy.load("en_core_web_sm")

# ── FILE LOADER ──
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

# ── CLEAN ──
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

# ── EXTRACT NAME ──
def extract_name(text):
    doc = nlp(text[:300])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Unknown"

# ── EXTRACT EMAIL ──
def extract_email(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return m.group() if m else None

# ── EXTRACT PHONE ──
def extract_phone(text):
    m = re.search(r'(\+?\d{1,3}[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}', text)
    return m.group() if m else None

# ── COMPREHENSIVE SKILLS LIST ──
SKILLS = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "r", "scala",
    "go", "golang", "rust", "swift", "kotlin", "php", "ruby", "perl", "matlab",
    "julia", "dart", "bash", "shell scripting", "powershell", "vba", "assembly",

    # Web Development
    "html", "css", "react", "react.js", "reactjs", "angular", "vue", "vue.js",
    "node.js", "nodejs", "express", "express.js", "next.js", "nuxt.js",
    "django", "flask", "fastapi", "spring", "spring boot", "asp.net",
    "rest api", "graphql", "bootstrap", "tailwind", "jquery", "webpack",
    "redux", "sass", "less",

    # Data Science & ML
    "machine learning", "deep learning", "artificial intelligence", "ai", "ml",
    "neural networks", "natural language processing", "nlp", "computer vision",
    "reinforcement learning", "supervised learning", "unsupervised learning",
    "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "xgboost",
    "lightgbm", "catboost", "hugging face", "transformers", "llm",
    "large language models", "langchain", "rag", "vector database",
    "feature engineering", "model training", "model deployment",
    "statistical analysis", "statistical modeling", "time series",
    "regression", "classification", "clustering", "dimensionality reduction",
    "random forest", "gradient boosting", "svm", "decision tree",
    "data science", "data analysis", "data analytics", "data engineering",
    "data visualization", "data mining", "data wrangling", "data cleaning",
    "data pipeline", "etl", "elt", "a/b testing", "hypothesis testing",

    # Data Tools
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy", "statsmodels",
    "pyspark", "apache spark", "spark", "hadoop", "hive", "pig", "kafka",
    "airflow", "apache airflow", "dbt", "databricks", "snowflake", "dask",

    # Databases
    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "sql server",
    "mssql", "mongodb", "nosql", "redis", "cassandra", "dynamodb", "firebase",
    "elasticsearch", "neo4j", "couchdb", "mariadb", "bigquery", "redshift",
    "hbase", "influxdb",

    # Cloud Platforms
    "aws", "amazon web services", "azure", "microsoft azure", "gcp",
    "google cloud", "google cloud platform", "heroku", "digital ocean",
    "cloudflare", "vercel", "netlify", "lambda", "ec2", "s3", "rds",
    "sagemaker", "azure ml", "vertex ai", "cloud functions",

    # DevOps & Tools
    "docker", "kubernetes", "k8s", "jenkins", "ci/cd", "github actions",
    "gitlab ci", "terraform", "ansible", "chef", "puppet", "vagrant",
    "linux", "unix", "git", "github", "gitlab", "bitbucket", "jira",
    "confluence", "agile", "scrum", "kanban", "devops", "mlops",
    "nginx", "apache", "microservices", "serverless",

    # BI & Visualization
    "power bi", "tableau", "looker", "qlik", "excel", "google sheets",
    "data studio", "metabase", "superset", "grafana", "kibana",
    "d3.js", "plotly", "bokeh",

    # Microsoft Office
    "microsoft excel", "microsoft word", "microsoft powerpoint",
    "microsoft outlook", "microsoft office", "word", "powerpoint", "outlook",

    # AI & LLM Stack
    "openai", "openai api", "gpt", "chatgpt", "claude", "gemini",
    "prompt engineering", "fine-tuning", "embeddings", "pinecone",
    "chromadb", "faiss", "weaviate", "langchain", "llamaindex",
    "agentic ai", "ai agents", "rag pipeline",

    # Automation & RPA
    "rpa", "uipath", "automation anywhere", "blue prism", "power automate",

    # Project Management
    "project management", "product management", "agile", "scrum", "kanban",
    "jira", "trello", "asana", "notion", "slack", "microsoft teams",
    "stakeholder management", "risk management",

    # Analytics & Marketing
    "google analytics", "seo", "sem", "digital marketing", "crm",
    "salesforce", "hubspot", "mixpanel", "amplitude",

    # Communication & Soft Skills
    "communication", "presentation", "leadership", "teamwork",
    "problem solving", "critical thinking", "attention to detail",
    "time management", "collaboration", "cross-functional",

    # Other Technical
    "sap", "erp", "tableau", "spotfire", "sas", "spss",
    "matlab", "simulink", "autocad", "solidworks",
    "cybersecurity", "networking", "tcp/ip", "rest", "soap",
    "api development", "web scraping", "selenium", "playwright",
    "unit testing", "test automation", "pytest", "junit",
]

# Remove duplicates and sort by length (longer first for better matching)
SKILLS = sorted(list(set(SKILLS)), key=len, reverse=True)


def extract_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in SKILLS:
        # Match whole word / phrase
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    # Return cleaned, title-cased list
    return sorted(list(found))


def extract_experience(text):
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
        r'(\d+)\+?\s*(?:years?|yrs?)',
    ]
    all_nums = []
    for p in patterns:
        matches = re.findall(p, text.lower())
        all_nums.extend([int(x) for x in matches if int(x) < 50])
    return max(all_nums) if all_nums else 0


def extract_education(text):
    edu = ["bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e",
           "mba", "bsc", "msc", "associate", "diploma"]
    text = text.lower()
    return [e for e in edu if e in text]


# ── MAIN ──
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
