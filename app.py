import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from data_processor import simple_resume_parser
from ml_models import CareerClassifier, analyze_skill_gap, virtual_mentor_response
from interview_analyzer import mock_facial_analysis, analyze_communication, calculate_employability_score
import psycopg2
from psycopg2 import sql

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# --- Initialize ML Models Globally ---
# These models are trained/loaded once when the server starts
career_classifier = CareerClassifier()
career_classifier.train_model() 

# Import the MOCK_JOB_REQUIREMENTS for use across endpoints
from ml_models import MOCK_JOB_REQUIREMENTS

# --- Database Setup (PostgreSQL) ---
def get_db_connection():
    """Establishes connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            # IMPORTANT: 'localhost' works if running Flask outside Docker, accessing Docker DB.
            host="localhost",
            port="5432",
            database=os.getenv("DB_NAME")
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# --- API Endpoints ---

@app.route('/api/status', methods=['GET'])
def status():
    """Basic health check."""
    return jsonify({"status": "running", "ml_backend": "online"})

@app.route('/api/v1/analyze_profile', methods=['POST'])
def analyze_profile():
    """
    (PHASE 2) Endpoint for comprehensive profile analysis: parsing, domain prediction, and skill gap analysis.
    """
    data = request.json
    if not data or 'resume_text' not in data:
        return jsonify({"error": "Please provide 'resume_text' in the request body."}), 400
    
    resume_text = data['resume_text']
    
    # 1. Resume Parsing
    parsed_data = simple_resume_parser(resume_text)
    student_skills = parsed_data['extracted_skills']
    
    # 2. Career Domain Classification & Skill Gap Analysis
    predicted_domains = career_classifier.predict_domain(resume_text)
    gap_analysis = None
    profile_match_percentage = 0
    
    if predicted_domains:
        # We assume the top predicted domain is the target for gap analysis
        target_domain = predicted_domains[0]
        required_skills = MOCK_JOB_REQUIREMENTS.get(target_domain)
        
        if required_skills:
            gap_analysis = analyze_skill_gap(required_skills, student_skills)
            profile_match_percentage = gap_analysis['completeness_percentage']
            gap_analysis['target_role'] = target_domain
            gap_analysis['required_skills'] = required_skills
            
    
    # 3. Save/Update Data to DB (Slightly simplified for project scope)
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            user_id = 'test_user_123' # Hardcoded placeholder for now
            
            cur.execute(
                sql.SQL("""
                    INSERT INTO student_profiles (user_id, name, email, raw_resume_text, extracted_skills, profile_match_percentage) 
                    VALUES (%s, %s, %s, %s, %s, %s) 
                    ON CONFLICT (user_id) 
                    DO UPDATE SET 
                        raw_resume_text=EXCLUDED.raw_resume_text, 
                        extracted_skills=EXCLUDED.extracted_skills,
                        email=EXCLUDED.email,
                        profile_match_percentage=EXCLUDED.profile_match_percentage
                """),
                [user_id, 'Dummy Student', parsed_data['email'], resume_text, student_skills, profile_match_percentage]
            )
            conn.commit()
        except Exception as e:
            # NOTE: If your student_profiles table doesn't have the 'profile_match_percentage' column yet, this will fail.
            print(f"Error saving to DB (Check your table schema!): {e}")
            conn.rollback()
        finally:
            if conn: conn.close()

    return jsonify({
        "message": "Profile analyzed successfully.",
        "student_data": parsed_data,
        "career_recommendations": predicted_domains,
        "skill_gap_analysis": gap_analysis,
        "profile_match_percentage": profile_match_percentage
    })

@app.route('/api/v1/mock_interview', methods=['POST'])
def mock_interview():
    """
    (PHASE 3) Endpoint for simulating a mock interview and generating the final employability score.
    Requires 'transcript' and 'profile_match_percentage' (from analyze_profile result).
    """
    data = request.json
    if not data or 'transcript' not in data or 'profile_match_percentage' not in data:
        return jsonify({"error": "Please provide 'transcript' and 'profile_match_percentage' (as a number)."}), 400
    
    transcript = data['transcript']
    try:
        profile_match_percentage = float(data['profile_match_percentage'])
    except ValueError:
        return jsonify({"error": "'profile_match_percentage' must be a valid number."}), 400
    
    # --- 1. Facial Analysis (MOCK) ---
    facial_result = mock_facial_analysis(video_input_data=None)
    
    # --- 2. Communication Analysis ---
    communication_result = analyze_communication(transcript)
    
    # Combined Interview Score
    interview_score = int(round((facial_result['facial_score'] + communication_result['score']) / 2, 0))
    
    # --- 3. Employability Score ---
    employability_score = calculate_employability_score(profile_match_percentage, interview_score)
    
    return jsonify({
        "message": "Mock interview analyzed successfully.",
        "interview_score": interview_score,
        "employability_score": employability_score,
        "facial_analysis": facial_result,
        "communication_analysis": communication_result
    })


@app.route('/api/v1/mentor_chat', methods=['POST'])
def mentor_chat():
    """
    (PHASE 3) Endpoint for interactive guidance with the Virtual Mentor Chatbot.
    Requires 'query' (user question) and 'domain' (predicted career path).
    """
    data = request.json
    if not data or 'query' not in data or 'domain' not in data:
        return jsonify({"error": "Please provide 'query' and 'domain'."}), 400
    
    query = data['query']
    domain = data['domain']
    
    response = virtual_mentor_response(query, domain)
    
    return jsonify({
        "mentor_response": response
    })


if __name__ == '__main__':
    # REMINDER: Uncomment and run this once to create your table schema if needed.
    # init_db() 
    app.run(debug=True, port=5000)