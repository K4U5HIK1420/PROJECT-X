import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

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

def rule_based_domain(text: str) -> str:
    t = text.lower()
    marketing_keywords = [
        "seo", "search engine", "google analytics", "google ads", "facebook ads",
        "social media", "content marketing", "email marketing", "mailchimp",
        "instagram", "linkedin", "campaign", "adwords", "ad campaigns", "marketing"
    ]
    if any(k in t for k in marketing_keywords):
        return "Digital Marketing"

    frontend_keywords = ["react", "javascript", "html", "css", "tailwind", "material ui", "ui/ux", "frontend"]
    if any(k in t for k in frontend_keywords):
        return "Frontend Development"

    ml_keywords = ["machine learning", "tensorflow", "scikit-learn", "deep learning", "cnn", "neural network", "keras"]
    if any(k in t for k in ml_keywords):
        return "AI/ML"

    data_keywords = ["pandas", "dataframe", "statistical", "visualization", "tableau"]
    if any(k in t for k in data_keywords):
        return "Data Science"

    return None

class CareerClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42))
        self.mlb = MultiLabelBinarizer()
        self.is_trained = False

    def train_model(self):
        df = pd.DataFrame(MOCK_DATA)
        X_vec = self.vectorizer.fit_transform(df['profile_text'])
        Y_bin = self.mlb.fit_transform(df['domain'])
        self.classifier.fit(X_vec, Y_bin)
        self.is_trained = True

    def predict_domain(self, text: str) -> list:
        if not self.is_trained:
            self.train_model()

        rule_label = rule_based_domain(text)
        if rule_label:
            return [rule_label]

        text_vec = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(text_vec)[0]

        sorted_indices = np.argsort(probabilities)[::-1]
        best_index = sorted_indices[0]
        best_domain = self.mlb.classes_[best_index]

        threshold = 0.12
        predicted_domains = [self.mlb.classes_[i] for i in sorted_indices if probabilities[i] >= threshold]
        if best_domain not in predicted_domains:
            predicted_domains.insert(0, best_domain)

        return predicted_domains

MOCK_JOB_REQUIREMENTS = {
    "AI/ML": ["Python", "TensorFlow", "Deep Learning", "SQL", "Cloud Computing"],
    "Software Development": ["Python", "Django", "PostgreSQL", "Docker", "REST APIs", "AWS"],
    "Digital Marketing": ["SEO", "Google Analytics", "Google Ads", "Facebook Ads", "Email Marketing", "Content Creation", "Social Media", "Analytics"],
    "AI/ML Engineer": ["Python", "TensorFlow", "Deep Learning", "SQL", "Cloud Computing"],
    "Data Science": ["Python", "R", "Pandas", "Statistical Analysis", "SQL", "Visualization"],
    "Frontend Development": ["React", "JavaScript", "HTML", "CSS", "REST APIs"],
    "Cybersecurity": ["Linux", "Network Security", "Cryptography", "Penetration Testing"],
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

def virtual_mentor_response(query: str, domain: str) -> str:
    query_lower = query.lower()

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

    response = virtual_mentor_response("What course should I study next?", "Data Science")
    print(f"Mentor: {response}")
