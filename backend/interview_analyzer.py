import random
import re
from textblob import TextBlob

def mock_facial_analysis(video_input_data=None):
    facial_score = random.randint(70, 95)

    if facial_score >= 90:
        dominant_emotion = "Confident"
        feedback = "Excellent eye contact and positive expressions."
    elif facial_score >= 80:
        dominant_emotion = "Slight Anxiety"
        feedback = "Maintain strong eye contact. Dominant state was Slight Anxiety."
    else:
        dominant_emotion = "Nervous"
        feedback = "Try to relax and smile more. Work on maintaining eye contact."

    attention = round(random.uniform(0.8, 1.0), 2)
    confidence = round(random.uniform(0.7, 0.95), 2)
    stress = round(random.uniform(0.05, 0.2), 2)

    return {
        "facial_score": facial_score,
        "emotions": {
            "attention": attention,
            "confidence": confidence,
            "dominant_emotion": dominant_emotion,
            "stress": stress
        },
        "feedback": feedback
    }

def analyze_communication(transcript: str) -> dict:
    if not transcript or len(transcript.strip()) == 0:
        return {
            "score": 0,
            "clarity_feedback": "No transcript provided.",
            "sentiment": "Neutral",
            "sentiment_scores": {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0}
        }

    blob = TextBlob(transcript)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]

    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    filler_words = ["um", "uh", "like", "you know", "so", "well"]
    filler_count = sum(transcript.lower().count(word) for word in filler_words)

    words = re.findall(r'\b\w+\b', transcript.lower())
    unique_words = set(words)
    vocab_diversity = len(unique_words) / len(words) if words else 0

    sentiment_score = max(0, (polarity + 1) * 50)
    length_score = max(0, 100 - abs(avg_sentence_length - 15) * 5)
    filler_score = max(0, 100 - filler_count * 10)
    vocab_score = min(100, vocab_diversity * 200)

    overall_score = int((sentiment_score + length_score + filler_score + vocab_score) / 4)
    overall_score = min(100, max(0, overall_score))

    if overall_score >= 85:
        clarity_feedback = "Excellent communication skills. Clear, confident, and engaging."
    elif overall_score >= 70:
        clarity_feedback = "Good communication. Vocabulary usage was clear and professional."
    elif overall_score >= 50:
        clarity_feedback = "Average communication. Work on reducing fillers and improving sentence structure."
    else:
        clarity_feedback = "Needs improvement. Practice speaking clearly and confidently."

    return {
        "score": overall_score,
        "clarity_feedback": clarity_feedback,
        "sentiment": sentiment,
        "sentiment_scores": {
            "compound": polarity,
            "neg": 0.0,
            "neu": 1 - abs(polarity),
            "pos": max(0, polarity)
        }
    }

def calculate_employability_score(profile_match_percentage: float, interview_score: int) -> int:
    if profile_match_percentage < 0 or profile_match_percentage > 100:
        raise ValueError("Profile match percentage must be between 0 and 100.")
    if interview_score < 0 or interview_score > 100:
        raise ValueError("Interview score must be between 0 and 100.")

    profile_weight = 0.6
    interview_weight = 0.4

    employability_score = (profile_match_percentage * profile_weight) + (interview_score * interview_weight)

    return int(round(employability_score, 0))

if __name__ == '__main__':
    facial = mock_facial_analysis()
    print(facial)

    test_transcript = "I am excited about this opportunity. I have experience in Python and machine learning. Um, I think I can contribute a lot to the team."
    comm = analyze_communication(test_transcript)
    print(comm)

    score = calculate_employability_score(85.0, 78)
    print(f"Employability Score: {score}")
