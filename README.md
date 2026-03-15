This is a multimodal performance analysis tool designed to evaluate both verbal delivery and non-verbal presence. By combining deep audio transcription analysis with computer vision, this project provides a holistic view of how a person communicates during interviews, presentations, or speeches.
The system processes audio to detect fluency issues like filler words and pauses, while simultaneously analyzing video frames to track eye contact, head stability, and body language.

Key Features
1. Verbal Fluency Analysis (Audio.py)
Automated Transcription: Uses OpenAI’s Whisper model to convert speech to text.
WPM Tracking: Calculates speaking rate in words per minute to ensure your pace is neither too fast nor too slow.
Filler Word Detection: Identifies and counts common "crutch" words such as "um," "uh," "like," and "actually".
Pause Analysis: Distinguishes between natural short pauses and long hesitations, providing specific timestamps for the latter.
Fluency Scoring: Generates a score based on the "Clarity," "Consistency," and "Delivery" of your speech.

2. Visual Confidence Tracking (Video.py)
Eye Contact Monitoring: Measures the symmetry and consistency of gaze toward the camera.
Biometric Markers: Tracks blink rates (to detect nervousness) and Mouth Aspect Ratio (MAR) to monitor expression.
Posture & Gesture: Analyzes head movement stability, shoulder posture, and hand fidgeting.
Confidence Classification: Categorizes performance into levels such as "Confident," "Neutral," or "Nervous".

Detailed Feedback & Outputs
The Annotated Video
The system generates a processed video file (annotated_output.mp4) that acts as a visual "highlight reel" of your mistakes:
Real-time Overlays: Displays live metrics for eye contact, head angles, and blink counts directly on the screen.
Visual Indicators: Shows a "YES/NO" indicator for eye contact so you can see exactly when you looked away.
Technical Landmarks: Plots nose tips and facial bounds to show how much you are moving during the session.

Comprehensive Reports
You receive a two-part breakdown of your performance:
Audio Report: Printed to the console, detailing your "Fillers per 100 words" and a final speech score out of 100.
JSON Data Report (interview_report.json): A deep-dive data file containing raw metrics for every subscore, including mean head motion in degrees and exact blink frequency.
Actionable Feedback: Both scripts generate specific text recommendations, such as "Reduce head movement" or "Slowing down may improve clarity".

Technical Requirements
Python 3.x
Core Libraries: opencv-python, mediapipe, librosa, numpy, openai-whisper.
Hardware: A GPU is recommended for faster Whisper transcription and MediaPipe processing, though not strictly required.
