import re
import spacy
from PyPDF2 import PdfReader

MODEL_NAME = "en_core_web_sm"

try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    nlp = None

MOCK_SKILLS = [
    "Python", "Java", "SQL", "React", "TensorFlow", "scikit-learn",
    "NLP", "Machine Learning", "Deep Learning", "AWS", "Docker",
    "Data Science", "Cybersecurity", "Django", "PostgreSQL", "JavaScript",
    "R", "Pandas", "Matplotlib", "Linux", "Network Security", "Cryptography",
    "Keras", "CNNs", "REST APIs", "Statistical Analysis", "Cloud Computing",
    "HTML", "CSS", "Visualization", "Penetration Testing",
    "SEO", "Google Analytics", "Google Ads", "Facebook Ads",
    "Email Marketing", "Content Creation", "Social Media", "Analytics",
    "Campaign Management", "Instagram Marketing", "LinkedIn Marketing",
    "UI/UX", "Project Management", "Communication", "Teamwork"
]

def extract_text_from_pdf(pdf_file_path: str) -> str:
    try:
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""

def simple_resume_parser(text: str) -> dict:
    if not text:
        return {"extracted_skills": [], "email": "N/A", "summary": "N/A"}

    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "N/A"

    found_skills = set()
    text_lower = text.lower()

    for skill in MOCK_SKILLS:
        skill_lower = skill.lower()
        if " " in skill_lower:
            if skill_lower in text_lower:
                found_skills.add(skill)
        else:
            if re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
                found_skills.add(skill)

    summary = "Cannot parse summary reliably yet."
    if nlp:
        doc = nlp(text)
        for sent in doc.sents:
            if "profile" in sent.text.lower() or "summary" in sent.text.lower() or "objective" in sent.text.lower():
                summary = sent.text.strip()
                break

    return {
        "email": email,
        "extracted_skills": list(found_skills),
        "summary": summary
    }

if __name__ == '__main__':
    print(f"Is spaCy NLP loaded? {'Yes' if nlp else 'No'}")
