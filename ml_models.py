import pandas as pd
import numpy as np
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# --- MOCK DATA FOR TRAINING ---
MOCK_DATA = {
    'profile_text': [
        "Strong in Python, TensorFlow, and Deep Learning. Experienced with CNNs and Keras.", 
        "Expert in Java, SQL, and database design. Experience with REST APIs and unit testing.", 
        "Familiar with Python, data cleaning, and statistical analysis. Used Pandas and Matplotlib.", 
        "Worked with Linux, network security, and penetration testing. Knows cryptography and firewalls.", 
        "Used React and JavaScript for front-end development. Good UI/UX sense.", 
        "Bachelors in CS. Focus on network protocol analysis and threat modeling.",
        "Experienced in backend development using Django and PostgreSQL."
    ],
    'domain': [
        ['AI/ML', 'Software Development'], 
        ['Software Development'], 
        ['Data Science'], 
        ['Cybersecurity'], 
        ['Software Development', 'Frontend Development'],
        ['Cybersecurity'],
        ['Software Development']
    ]
}

# --- 1. Career Domain Classification Model ---

class CareerClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = LogisticRegression(solver='liblinear', random_state=42)
        self.mlb = MultiLabelBinarizer()
        self.is_trained = False

    def train_model(self):
        """Trains the classification model using mock data."""
        df = pd.DataFrame(MOCK_DATA)
        X_vec = self.vectorizer.fit_transform(df['profile_text'])
        Y_bin = self.mlb.fit_transform(df['domain'])
        # Convert to 1D array for binary classification, but since it's multi-label, we need to handle differently
        # For simplicity, let's use a different approach or fix the labels
        # Actually, LogisticRegression doesn't support multi-label directly; we need to use MultiOutputClassifier or OneVsRestClassifier
        from sklearn.multiclass import OneVsRestClassifier
        self.classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42))
        self.classifier.fit(X_vec, Y_bin)
        self.is_trained = True
        print("Career Classifier Trained successfully.")
    
    def predict_domain(self, text: str) -> list:
        """Predicts the best-fit career domains for a given profile text."""
        if not self.is_trained:
            self.train_model() 
        
        text_vec = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(text_vec)[0]
        
        # Identify labels that meet a low threshold (0.2) or take the single highest one
        predicted_labels_indices = np.where(probabilities > 0.2)[0] 
        
        if len(predicted_labels_indices) == 0:
            top_index = np.argmax(probabilities)
            predicted_domains = self.mlb.classes_[top_index:top_index+1].tolist()
        else:
            predicted_domains = self.mlb.classes_[predicted_labels_indices].tolist()

        return predicted_domains

# --- 2. Skill Gap Analyzer ---

MOCK_JOB_REQUIREMENTS = {
    "AI/ML Engineer": ["Python", "TensorFlow", "Deep Learning", "SQL", "Cloud Computing"],
    "Data Science": ["Python", "R", "Pandas", "Statistical Analysis", "SQL", "Visualization"],
    "Frontend Development": ["React", "JavaScript", "HTML", "CSS", "REST APIs"],
    "Cybersecurity": ["Linux", "Network Security", "Cryptography", "Penetration Testing"],
    # New additions
    "Backend Developer": ["Python", "Django", "PostgreSQL", "Docker", "REST APIs", "AWS"],
    "Full Stack Developer": ["JavaScript", "React", "Node.js", "Express", "MongoDB", "HTML", "CSS"],
    "DevOps Engineer": ["Linux", "Docker", "Kubernetes", "AWS", "Jenkins", "CI/CD", "Terraform"],
    "Mobile App Developer": ["Dart", "Flutter", "Firebase", "REST APIs", "Git"],
    "Data Analyst": ["SQL", "Excel", "Tableau", "Python", "Data Visualization"],
    "Cloud Architect": ["AWS", "Networking", "Security", "Python", "Terraform"],
    "QA Automation Engineer": ["Java", "Python", "Selenium", "Jenkins", "Git", "SQL"],
    "Blockchain Developer": ["Solidity", "Ethereum", "Smart Contracts", "Web3.js", "JavaScript"]
}

def analyze_skill_gap(required_skills: list, student_skills: list) -> dict:
    """Compares required skills for a role against student's extracted skills."""
    required_set = set(s.lower() for s in required_skills)
    student_set = set(s.lower() for s in student_skills)
    
    missing_skills = list(required_set - student_set)
    matched_skills = list(required_set.intersection(student_set))
    
    total_required = len(required_set)
    if total_required == 0:
        completeness_percentage = 100
    else:
        completeness_percentage = (len(matched_skills) / total_required) * 100
        
    return {
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "completeness_percentage": round(completeness_percentage, 1)
    }

# --- 3. Virtual Mentor Chatbot (Simple NLP Responder) ---

def virtual_mentor_response(query: str, domain: str) -> str:
    """Generates a simple rule-based response for career guidance."""
    query_lower = query.lower()
    
    # Safely get a key tech for the domain
    key_tech = MOCK_JOB_REQUIREMENTS.get(domain, ['key technologies'])[0]

    if "study next" in query_lower or "course" in query_lower or "certifications" in query_lower:
        return f"Since your profile aligns with **{domain}**, I strongly recommend focusing on advanced courses in **{key_tech}**. Look into specialized certifications or online learning paths on platforms like Coursera/Udemy to fill those gaps."
    
    elif "job market" in query_lower or "salary" in query_lower:
        return f"The job market for **{domain}** is experiencing high demand. To stand out, ensure you have a strong project portfolio showcasing skills in {key_tech}. This will significantly increase your appeal to top recruiters."
    
    elif "interview" in query_lower or "prepare" in query_lower:
        return f"To prepare for **{domain}** interviews, focus on mastering technical concepts related to **{key_tech}** and practicing behavioral questions. Remember to maintain strong confidence and clarity, as noted in your mock session."
        
    elif "alumni" in query_lower or "mentor" in query_lower:
        return "Connecting with alumni is crucial! I suggest filtering your university's alumni network to find professionals currently working as a **{domain}** expert and asking them for a brief informational interview."
    
    else:
        return f"That's a great question! As your Virtual Mentor, I'm here to guide you. Try asking about your next learning steps, potential salary, or how to improve your interview skills for your target domain: **{domain}**."

if __name__ == '__main__':
    classifier = CareerClassifier()
    classifier.train_model()
    
    # Test Chatbot
    print("\n--- Testing Virtual Mentor ---")
    response = virtual_mentor_response("What course should I study next?", "Data Science")
    print(f"Mentor: {response}")