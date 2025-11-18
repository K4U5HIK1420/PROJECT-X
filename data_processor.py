import re
import spacy
from PyPDF2 import PdfReader
from io import BytesIO

# Define the model name. This should match the package name installed via pip URL.
MODEL_NAME = "en_core_web_sm"

# Load the spaCy model
try:
    nlp = spacy.load(MODEL_NAME)
    print(f"spaCy model '{MODEL_NAME}' loaded successfully.")
except OSError as e:
    # If loading fails, it usually means the model was not installed or linked correctly.
    print(f"Warning: spaCy model '{MODEL_NAME}' not found.")
    print("If you encounter errors, ensure you run the direct 'pip install [URL]' command.")
    nlp = None

# Mock list of industry standard skills for matching
MOCK_SKILLS = [
    # --- Programming / Technical ---
    "Python", "Java", "SQL", "React", "TensorFlow", "scikit-learn",
    "NLP", "Machine Learning", "Deep Learning", "AWS", "Docker",
    "Data Science", "Cybersecurity", "Django", "PostgreSQL", "JavaScript",
    "R", "Pandas", "Matplotlib", "Linux", "Network Security", "Cryptography",
    "Keras", "CNNs", "REST APIs", "Statistical Analysis", "Cloud Computing",
    "HTML", "CSS", "Visualization", "Penetration Testing",

    # --- Digital Marketing Skills ---
    "SEO", "Google Analytics", "Google Ads", "Facebook Ads",
    "Email Marketing", "Content Creation", "Social Media", "Analytics",
    "Campaign Management", "Instagram Marketing", "LinkedIn Marketing",

    # --- Soft/General Skills (Optional) ---
    "UI/UX", "Project Management", "Communication", "Teamwork"
]


def extract_text_from_pdf(pdf_file_path: str) -> str:
    """MOCK: Extracts text from a PDF file."""
    # This is a placeholder utility for file stream handling in the final app.
    try:
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        # If running the mock test, this will fail as no file is provided.
        # This function is not used in the current API flow but kept for completeness.
        # print(f"Error reading PDF: {e}") 
        return ""

def simple_resume_parser(text: str) -> dict:
    """
    Performs basic extraction of skills and email from a resume text.
    """
    if not text:
        return {"extracted_skills": [], "email": "N/A", "summary": "N/A"}

    # 1. Extract Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "N/A"

    # 2. Extract Skills (Keyword Matching)
    found_skills = set()
    text_lower = text.lower()
    
    for skill in MOCK_SKILLS:
        skill_lower = skill.lower()

        # Multi-word skill handling (e.g., "google analytics", "content creation")
        if " " in skill_lower:
            # simple containment match
            if skill_lower in text_lower:
                found_skills.add(skill)
        else:
            # word-boundary match for single-word skills
            if re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
                found_skills.add(skill)


    # 3. Extract Summary/Profile (using rough heuristic)
    summary = "Cannot parse summary reliably yet."
    if nlp:
        doc = nlp(text)
        # Simple heuristic: look for sentences near keywords
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
    # Test your installation
    print(f"Is spaCy NLP loaded? {'Yes' if nlp else 'No'}")