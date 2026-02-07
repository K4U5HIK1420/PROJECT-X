from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import os
from datetime import datetime

def generate_interview_report(data):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "interview_report.pdf")

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI Mock Interview Report")

    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated on: {datetime.now().strftime('%d %b %Y %H:%M')}")

    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Employability Score: {data.get('employability_score', 'N/A')}")

    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Sentiment: {data.get('communication_analysis', {}).get('sentiment')}")

    y -= 20
    c.drawString(50, y, f"Dominant Emotion: {data.get('facial_analysis', {}).get('emotions', {}).get('dominant_emotion')}")

    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Transcript:")

    y -= 20
    c.setFont("Helvetica", 10)
    transcript = data.get("communication_analysis", {}).get("full_transcript", "")

    for line in transcript.split("\n"):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line[:100])
        y -= 14

    c.save()
    return file_path
