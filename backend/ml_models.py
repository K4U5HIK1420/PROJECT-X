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

def virtual_mentor_response(
    query: str,
    domain: str,
    employability_score: float = 0,
    missing_skills: list = None
) -> str:

    query = query.lower()
    missing_skills = missing_skills or []

    # Determine confidence level
    if employability_score >= 85:
        level = "job-ready"
    elif employability_score >= 65:
        level = "intermediate"
    else:
        level = "beginner"

    intro = f"As your Virtual Career Mentor for **{domain}**, here’s my honest guidance:\n\n"

    # -------------------------------
    # Career Roadmap (NEW)
    # -------------------------------
    if any(k in query for k in ["roadmap", "career path", "step by step", "plan"]):
        roadmap = generate_career_roadmap(domain, employability_score, missing_skills)

        response = f"Here’s a **step-by-step career roadmap for {domain}** 👇\n\n"
        response += f"🎯 Current Level: **{roadmap['current_level']}**\n\n"

        for phase in roadmap["roadmap"]:
            response += f"🔹 **{phase['phase']}**\n"
            response += "Focus:\n"
            for item in phase["focus"]:
                response += f"• {item}\n"
            response += f"Goal: {phase['goal']}\n\n"

        response += "Follow this sequentially — don’t rush phases."
        return response

    # -------------------------------
    # Learning / Courses
    # -------------------------------
    if any(k in query for k in ["learn", "study", "course", "next", "skill"]):
        if missing_skills:
            return (
                intro +
                f"You’re currently at an **{level}** stage.\n\n"
                f"Your biggest growth opportunity right now is:\n"
                f"👉 **" + ", ".join(missing_skills[:3]) + "**\n\n"
                "My advice:\n"
                "• Pick ONE skill and go deep\n"
                "• Build a small but real project\n"
                "• Document what you learn (GitHub / notes)\n\n"
                "This approach compounds fast."
            )
        else:
            return (
                intro +
                "Your skill alignment is already strong 👍\n\n"
                "Now focus on:\n"
                "• Advanced projects\n"
                "• Mock interviews\n"
                "• System design & real-world scenarios\n\n"
                "You’re closer than you think."
            )

    # -------------------------------
    # Interview Preparation
    # -------------------------------
    if any(k in query for k in ["interview", "prepare", "confidence"]):
        return (
            intro +
            "Interview success comes down to clarity, not memorization.\n\n"
            "Focus on:\n"
            "• Explaining your projects end-to-end\n"
            "• Why you made certain design choices\n"
            "• Common mistakes & what you learned\n\n"
            "Mock interviews + self-review = confidence."
        )

    # -------------------------------
    # Jobs / Market / Salary
    # -------------------------------
    if any(k in query for k in ["job", "salary", "market", "hiring"]):
        return (
            intro +
            f"The job market for **{domain}** is competitive, but fair.\n\n"
            "What actually works:\n"
            "• Strong fundamentals\n"
            "• 2–3 solid projects\n"
            "• A clean, honest resume\n\n"
            "Avoid chasing trends blindly — consistency wins."
        )

    # -------------------------------
    # Confusion / Motivation
    # -------------------------------
    if any(k in query for k in ["confused", "lost", "stuck", "direction"]):
        return (
            intro +
            "Feeling confused is normal — it means you care.\n\n"
            "Here’s how to reset:\n"
            "• Pick ONE domain\n"
            "• Pick ONE roadmap\n"
            "• Pick ONE project\n\n"
            "Progress beats perfection. Keep moving."
        )

    # -------------------------------
    # Default fallback
    # -------------------------------
    return (
        intro +
        "You can ask me about:\n"
        "• What to learn next\n"
        "• Interview preparation\n"
        "• Job readiness\n"
        "• Career roadmap\n\n"
        "Ask freely — that’s how growth happens."
    )


def generate_career_roadmap(domain: str, employability_score: float, missing_skills: list):
    
    # Determine level
    if employability_score >= 85:
        level = "Job-Ready"
    elif employability_score >= 65:
        level = "Intermediate"
    else:
        level = "Beginner"

    roadmap = []

    # Phase 1: Foundations
    roadmap.append({
        "phase": "Phase 1: Strengthen Fundamentals",
        "focus": missing_skills[:3] if missing_skills else ["Core concepts review"],
        "goal": "Build strong conceptual clarity"
    })

    # Phase 2: Projects
    roadmap.append({
        "phase": "Phase 2: Project Building",
        "focus": [
            f"Build 2 real-world projects in {domain}",
            "Use GitHub with clean README",
            "Apply best practices"
        ],
        "goal": "Demonstrate practical ability"
    })

    # Phase 3: Interview Prep
    roadmap.append({
        "phase": "Phase 3: Interview Preparation",
        "focus": [
            "Explain projects end-to-end",
            "Practice technical + HR questions",
            "Mock interviews"
        ],
        "goal": "Improve confidence and communication"
    })

    # Phase 4: Job Readiness
    roadmap.append({
        "phase": "Phase 4: Job Readiness",
        "focus": [
            "Optimize resume for ATS",
            "Apply consistently",
            "Network on LinkedIn"
        ],
        "goal": "Convert skills into opportunities"
    })

    return {
        "domain": domain,
        "current_level": level,
        "roadmap": roadmap
    }


