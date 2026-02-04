import random
import re
import os
import cv2  # <--- NEW: For video processing
from textblob import TextBlob
import whisper
from deepface import DeepFace # <--- NEW: For emotion detection

# --- 0. LOAD WHISPER MODEL ---
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded!")

# --- 1. TRANSCRIPTION (Existing) ---
def transcribe_video(video_path):
    try:
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return ""

# --- 2. FACIAL ANALYSIS (NEW & REAL) ---
def analyze_video_faces(video_path):
    """
    Scans the video, takes a frame every 1 second, and detects emotions.
    """
    print(f"Starting facial analysis on {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    emotions_list = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30 # Default to 30 if unknown
    frame_interval = int(frame_rate) # Process 1 frame per second
    
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only analyze once per second (to speed it up)
        if current_frame % frame_interval == 0:
            try:
                # DeepFace analyze
                # 'actions'=['emotion'] tells it to only look for emotions (faster)
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # DeepFace returns a list (in case of multiple faces), get the first one
                if isinstance(analysis, list):
                    dominant_emotion = analysis[0]['dominant_emotion']
                    emotions_list.append(dominant_emotion)
                else:
                    emotions_list.append(analysis['dominant_emotion'])
                    
            except Exception as e:
                # If no face is found in a specific frame, just skip it
                pass
        
        current_frame += 1
        
    cap.release()
    
    # --- CALCULATE RESULTS ---
    if not emotions_list:
        return {
            "facial_score": 50,
            "emotions": {"dominant_emotion": "No Face Detected", "confidence": 0},
            "feedback": "Could not detect a face. Ensure good lighting."
        }
        
    # Count most frequent emotion
    emotion_counts = {e: emotions_list.count(e) for e in set(emotions_list)}
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # Calculate a "Facial Score" based on positive vs negative emotions
    positive_emotions = ["happy", "neutral", "surprise"]
    negative_emotions = ["sad", "angry", "fear", "disgust"]
    
    positive_count = sum(emotion_counts.get(e, 0) for e in positive_emotions)
    total_frames = len(emotions_list)
    
    # Simple score: % of time spent looking positive/neutral
    facial_score = int((positive_count / total_frames) * 100)
    
    # Generate Feedback
    if dominant_emotion in ["fear", "sad"]:
        feedback = f"You looked mostly {dominant_emotion}. Try to smile and relax."
    elif dominant_emotion == "happy":
        feedback = "Great energy! You looked happy and confident."
    elif dominant_emotion == "neutral":
        feedback = "You maintained a calm, professional demeanor."
    else:
        feedback = f"Dominant expression was {dominant_emotion}. Work on eye contact."

    print(f"Facial Analysis Complete. Dominant: {dominant_emotion}")

    return {
        "facial_score": facial_score,
        "emotions": {
            "dominant_emotion": dominant_emotion.capitalize(),
            "confidence": round(positive_count / total_frames, 2)
        },
        "feedback": feedback
    }

# --- 3. TRANSCRIPT HIGHLIGHTER (Existing) ---
def highlight_transcript_issues(transcript: str):
    if not transcript: return {"text": "", "issues": []}
    issues = []
    fillers = ["um", "uh", "umm", "like", "you know"]
    for filler in fillers:
        for match in re.finditer(r'\b' + re.escape(filler) + r'\b', transcript, re.IGNORECASE):
            issues.append({"type": "filler", "text": match.group(), "start": match.start(), "end": match.end(), "feedback": "Filler word"})
    for match in re.finditer(r'\b(\w+)\s+\1\b', transcript, re.IGNORECASE):
        issues.append({"type": "repetition", "text": match.group(), "start": match.start(), "end": match.end(), "feedback": "Repetition"})
    issues.sort(key=lambda x: x["start"])
    return {"text": transcript, "issues": issues}

# --- 4. COMMUNICATION ANALYZER (Existing) ---
def analyze_communication(transcript: str) -> dict:
    highlight_data = highlight_transcript_issues(transcript)
    if not transcript:
        return {"score": 0, "clarity_feedback": "No speech.", "sentiment": "Neutral", "transcript_analysis": highlight_data}
    
    blob = TextBlob(transcript)
    issue_count = len(highlight_data['issues'])
    final_score = max(50, 100 - (issue_count * 5))
    
    return {
        "score": final_score,
        "clarity_feedback": "Good clarity." if final_score > 80 else "Reduce filler words.",
        "sentiment": "Positive" if blob.sentiment.polarity > 0 else "Neutral",
        "transcript_analysis": highlight_data
    }

def calculate_employability_score(profile, interview):
    return int((profile * 0.6) + (interview * 0.4))