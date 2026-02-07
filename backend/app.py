import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from data_processor import simple_resume_parser
from ml_models import CareerClassifier, analyze_skill_gap, virtual_mentor_response
from interview_analyzer import analyze_video_faces, analyze_communication, calculate_employability_score, transcribe_video
import psycopg2
from psycopg2 import sql
from flask import send_file, request
from report_generator import generate_interview_report


load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route("/api/v1/mock_facial_interview", methods=["POST"])
def mock_facial_interview():
    if "video" not in request.files:
        return jsonify({"error": "No video file received"}), 400

    video = request.files["video"]
    # 1. Save video temporarily so Whisper can read it
    temp_filename = "temp_interview.webm"
    video.save(temp_filename)

    try:
        # 2. REAL AI TRANSCRIPTION
        print("Starting transcription...")
        real_transcript = transcribe_video(temp_filename)
        print(f"Transcribed Text: {real_transcript}")
        
        # 3. Analyze the REAL transcript
        facial_result = analyze_video_faces(temp_filename)
        comm_result = analyze_communication(real_transcript)

        # 4. Calculate Score
        response = {
            "employability_score": 75, # You can make this dynamic later
            "interview_score": comm_result['score'],
            "communication_analysis": comm_result,
            "facial_analysis": facial_result
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Error processing interview: {e}")
        return jsonify({"error": "Processing failed"}), 500
        
    finally:
        # 5. Cleanup: Delete the temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

career_classifier = CareerClassifier()
career_classifier.train_model()

from ml_models import MOCK_JOB_REQUIREMENTS

def get_db_connection():
    try:
        conn = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host="localhost",
            port="5432",
            database=os.getenv("DB_NAME")
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"status": "running", "ml_backend": "online"})

@app.route('/api/v1/analyze_profile', methods=['POST'])
def analyze_profile():
    data = request.json
    if not data or 'resume_text' not in data:
        return jsonify({"error": "Please provide 'resume_text' in the request body."}), 400

    resume_text = data['resume_text']

    parsed_data = simple_resume_parser(resume_text)
    student_skills = parsed_data['extracted_skills']

    predicted_domains = career_classifier.predict_domain(resume_text)
    if not predicted_domains:
        predicted_domains = ["Unknown"]

    gap_analysis = None
    profile_match_percentage = 0

    if predicted_domains:
        target_domain = predicted_domains[0]
        required_skills = MOCK_JOB_REQUIREMENTS.get(target_domain)

    if required_skills:
        gap_analysis = analyze_skill_gap(required_skills, student_skills)
        profile_match_percentage = gap_analysis['completeness_percentage']
        gap_analysis['target_role'] = target_domain
        gap_analysis['required_skills'] = required_skills

    return jsonify({
        "message": "Profile analyzed successfully.",
        "student_data": parsed_data or {},
        "career_recommendations": predicted_domains or [],
        "skill_gap_analysis": gap_analysis or {
            "missing_skills": [],
            "matched_skills": [],
            "completeness_percentage": 0,
            "target_role": "Unknown",
            "required_skills": []
        },
        "profile_match_percentage": profile_match_percentage or 0
    })

@app.route('/api/v1/mock_interview', methods=['POST'])
def mock_interview():
    data = request.json
    if not data or 'transcript' not in data or 'profile_match_percentage' not in data:
        return jsonify({"error": "Please provide 'transcript' and 'profile_match_percentage' (as a number)."}), 400

    transcript = data['transcript']
    try:
        profile_match_percentage = float(data['profile_match_percentage'])
    except ValueError:
        return jsonify({"error": "'profile_match_percentage' must be a valid number."}), 400

    facial_result = analyze_video_faces(video_input_data=None)
    communication_result = analyze_communication(transcript)
    interview_score = int(round((facial_result['facial_score'] + communication_result['score']) / 2, 0))
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
    data = request.json
    if not data or 'query' not in data or 'domain' not in data:
        return jsonify({"error": "Please provide 'query' and 'domain'."}), 400

    query = data['query']
    domain = data['domain']

    response = virtual_mentor_response(
    query=query,
    domain=domain,
    employability_score=data.get("employability_score", 0),
    missing_skills=data.get("missing_skills", [])
)


    return jsonify({
        "mentor_response": response
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

    @app.route("/api/v1/interview_report", methods=["POST"])
    def interview_report():
        data = request.json  # interview result data
        pdf_path = generate_interview_report(data)
        return send_file(
        pdf_path,
        as_attachment=True,
        download_name="AI_Interview_Report.pdf"
    )
