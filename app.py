import streamlit as st
import sqlite3
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import hashlib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Resume AI", layout="wide")
nlp = spacy.load("en_core_web_sm")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------- AUTH ----------------
def signup(username, password):
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

# ---------------- DATA ----------------
jobs = pd.read_csv("jobs.csv")
skills_master = set(open("skills_list.txt").read().split())

# ---------------- RESUME PARSER ----------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join(p.extract_text() for p in reader.pages if p.extract_text())
    if file.name.endswith(".docx"):
        d = docx.Document(file)
        return " ".join(p.text for p in d.paragraphs)
    return ""

def extract_skills(text):
    doc = nlp(text.lower())
    return {t.text for t in doc if t.text in skills_master}

# ---------------- MATCHING ----------------
def recommend_jobs(resume_text, resume_skills):
    vectorizer = TfidfVectorizer()
    job_vec = vectorizer.fit_transform(jobs["job_description"])
    res_vec = vectorizer.transform([resume_text])
    scores = cosine_similarity(res_vec, job_vec)[0]

    results = []
    for i, score in enumerate(scores):
        job_skills = set(jobs.iloc[i]["skills"].split())
        results.append({
            "Company": jobs.iloc[i]["company"],
            "Job Role": jobs.iloc[i]["job_title"],
            "Category": jobs.iloc[i]["category"],
            "Match %": round(score * 100, 2),
            "Skills to Improve": list(job_skills - resume_skills)
        })
    return sorted(results, key=lambda x: x["Match %"], reverse=True)

# ---------------- UI ----------------
st.markdown("## 🚀 Smart Resume Analyzer & Company Recommender")

menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])

if "user" not in st.session_state:
    st.session_state.user = None

# -------- LOGIN --------
if menu == "Login":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(u, p):
            st.session_state.user = u
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

# -------- SIGNUP --------
if menu == "Sign Up":
    u = st.text_input("Create Username")
    p = st.text_input("Create Password", type="password")
    if st.button("Sign Up"):
        if signup(u, p):
            st.success("Account created. Login now.")
        else:
            st.error("Username already exists")

# -------- DASHBOARD --------
if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user}")

    resume = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

    if resume:
        text = extract_text(resume)
        skills = extract_skills(text)

        st.subheader("🧠 Extracted Skills")
        st.write(skills)

        results = recommend_jobs(text, skills)

        st.subheader("🏢 Recommended Companies & Roles")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        st.subheader("📌 Top Matches")
        for r in results[:5]:
            with st.expander(f"{r['Company']} – {r['Job Role']} ({r['Match %']}%)"):
                st.write("Category:", r["Category"])
                st.write("Skills to Improve:", r["Skills to Improve"])
