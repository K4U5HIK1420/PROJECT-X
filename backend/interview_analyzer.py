import random
import re
import os
from textblob import TextBlob
import whisper  # <--- IMPORT WHISPER

# --- 0. LOAD WHISPER MODEL (Runs once at startup) ---
print("Loading Whisper model... (this may take a moment)")
# "base" is a good balance of speed and accuracy. Use "tiny" for faster results.
model = whisper.load_model("base")
print("Whisper model loaded!")

# --- 1. NEW TRANSCRIPTION FUNCTION ---
def transcribe_video(video_path):
    """
    Uses OpenAI Whisper to transcribe the audio from the video file.
    """
    try:
        # Whisper handles audio extraction automatically
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return ""

# --- 2. HIGHLIGHTER FUNCTION ---
def highlight_transcript_issues(transcript: str):
    if not transcript:
        return {"text": "", "issues": []}

    issues = []
    
    # Detect Filler Words
    fillers = ["um", "uh", "umm", "like", "you know", "sort of", "kind of"]
    for filler in fillers:
        for match in re.finditer(r'\b' + re.escape(filler) + r'\b', transcript, re.IGNORECASE):
            issues.append({
                "type": "filler",
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "feedback": "Filler word detected"
            })

    # Detect Repetitions
    for match in re.finditer(r'\b(\w+)\s+\1\b', transcript, re.IGNORECASE):
        issues.append({
            "type": "repetition",
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
            "feedback": "Repetition detected"
        })

    issues.sort(key=lambda x: x["start"])
    return {"text": transcript, "issues": issues}

# --- 3. ANALYSIS FUNCTION ---
def analyze_communication(transcript: str) -> dict:
    highlight_data = highlight_transcript_issues(transcript)
    
    if not transcript or len(transcript.strip()) == 0:
        return {
            "score": 0,
            "clarity_feedback": "No speech detected.",
            "sentiment": "Neutral",
            "transcript_analysis": highlight_data
        }

    blob = TextBlob(transcript)
    polarity = blob.sentiment.polarity
    
    # Scoring Logic
    issue_count = len(highlight_data['issues'])
    base_score = 100
    deduction = issue_count * 5
    final_score = max(50, base_score - deduction)

    if final_score >= 85:
        feedback = "Excellent! Clear and concise communication."
    elif final_score >= 70:
        feedback = "Good, but try to reduce filler words."
    else:
        feedback = "Needs practice. You used many filler words."

    return {
        "score": final_score,
        "clarity_feedback": feedback,
        "sentiment": "Positive" if polarity > 0 else "Neutral",
        "transcript_analysis": highlight_data
    }

# --- 4. MOCK FACIAL ANALYSIS (Keep Mock for now) ---
def mock_facial_analysis(video_input_data=None):
    return {
        "facial_score": random.randint(75, 90),
        "emotions": {
            "dominant_emotion": "Confident",
            "confidence": 0.85
        },
        "feedback": "Good eye contact maintained."
    }

def calculate_employability_score(profile_match_percentage: float, interview_score: int) -> int:
    return int((profile_match_percentage * 0.6) + (interview_score * 0.4))