import sys
import os
import librosa
import librosa.effects
import numpy as np
import whisper
import re

if len(sys.argv) < 2:
    print("Usage: python audio_assistant.py <audio_file_path>")
    sys.exit(1)

audio_path = sys.argv[1]

if not os.path.exists(audio_path):
    print(f"Error: File not found -> {audio_path}")
    sys.exit(1)

def load_and_preprocess_audio(path, target_sr=16000):
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio, sr

audio, sr = load_and_preprocess_audio(audio_path)

audio_duration = len(audio) / sr

print(f"Loaded audio file: {audio_path}")
print("Sample rate:", sr)
print("Audio duration (seconds):", round(audio_duration, 2))
print("Min amplitude:", np.min(audio))
print("Max amplitude:", np.max(audio))

model = whisper.load_model("base")

result = model.transcribe(
    audio_path,
    language="en",
    fp16=False
)

transcription_text = result.get("text", "")

print("\nTranscription:\n")
print(transcription_text)

def count_words(text):
    return len(text.strip().split())

total_words = count_words(transcription_text)

def speaking_rate_wpm(word_count, duration_seconds):
    minutes = duration_seconds / 60
    return word_count / minutes if minutes > 0 else 0

wpm = speaking_rate_wpm(total_words, audio_duration)

print("\nTotal words:", total_words)
print("Speaking rate (WPM):", round(wpm, 1))

def detect_pauses_with_timestamps(audio, sr, silence_db=25, short_pause=(0.3, 0.7), long_pause_threshold=0.7):
    intervals = librosa.effects.split(audio, top_db=silence_db)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pause_duration = (start - prev_end) / sr
        pause_start_time = prev_end / sr
        if pause_duration > 0:
            pauses.append({"duration": pause_duration, "start_time": pause_start_time})
        prev_end = end
    short_pauses = [p for p in pauses if short_pause[0] <= p["duration"] < short_pause[1]]
    long_pauses = [p for p in pauses if p["duration"] >= long_pause_threshold]
    total_pause_time = sum(p["duration"] for p in pauses)
    return short_pauses, long_pauses, total_pause_time

short_pauses, long_pauses, total_pause_time = detect_pauses_with_timestamps(audio, sr)

pause_ratio = total_pause_time / audio_duration if audio_duration > 0 else 0

print("\nPause analysis:")
print("Short pauses:", len(short_pauses))
print("Long pauses:", len(long_pauses))
print("Total pause time:", round(total_pause_time, 2))
print("Pause ratio:", round(pause_ratio, 3))

print("\nLong pause timestamps:")
for p in long_pauses:
    print(f"- {round(p['start_time'], 2)}s → {round(p['duration'], 2)}s")

FILLER_WORDS = [
    "uh", "um", "erm",
    "like", "you know",
    "actually", "basically",
    "i mean"
]

def detect_fillers(text, filler_list):
    text = text.lower()
    filler_counts = {}
    for filler in filler_list:
        pattern = r"\b" + re.escape(filler) + r"\b"
        filler_counts[filler] = len(re.findall(pattern, text))
    total_fillers = sum(filler_counts.values())
    return filler_counts, total_fillers

filler_counts, total_fillers = detect_fillers(transcription_text, FILLER_WORDS)

fillers_per_100_words = (total_fillers / total_words) * 100 if total_words > 0 else 0

print("\nFiller breakdown:", filler_counts)
print("Total fillers:", total_fillers)
print("Fillers per 100 words:", round(fillers_per_100_words, 2))

segments = result.get("segments", [])
segment_lengths = [(seg["end"] - seg["start"]) for seg in segments] if segments else []
consistency_std = np.std(segment_lengths) if segment_lengths else 0.0

print("\nSpeaking consistency (std):", round(consistency_std, 2))

def clarity_score(pause_ratio, fillers_per_100, wpm):
    score = 100
    if pause_ratio > 0.35:
        score -= 30
    elif pause_ratio > 0.25:
        score -= 15
    if fillers_per_100 > 8:
        score -= 30
    elif fillers_per_100 > 4:
        score -= 15
    if wpm < 90 or wpm > 180:
        score -= 15
    return max(score, 0)

WEIGHTS = {
    "fluency": 0.35,
    "clarity": 0.30,
    "consistency": 0.20,
    "delivery": 0.15
}

fluency_score = max(
    100 - (pause_ratio * 100) - (fillers_per_100_words * 5) - (len(long_pauses) * 5),
    0
)

clarity_numeric = clarity_score(pause_ratio, fillers_per_100_words, wpm)

consistency_score = max(100 - (consistency_std * 10), 0)

delivery_score = 100 if 90 <= wpm <= 180 else 70

final_score = (
    fluency_score * WEIGHTS["fluency"]
    + clarity_numeric * WEIGHTS["clarity"]
    + consistency_score * WEIGHTS["consistency"]
    + delivery_score * WEIGHTS["delivery"]
)

print("\n--- FINAL SPEECH EVALUATION ---")
print("Fluency score:", round(fluency_score, 1))
print("Clarity score:", round(clarity_numeric, 1))
print("Consistency score:", round(consistency_score, 1))
print("Delivery score:", round(delivery_score, 1))
print("\nFinal Speech Score:", round(final_score, 1), "/ 100")

def generate_feedback(pause_ratio, fillers_per_100, long_pause_count, wpm):
    feedback = []
    if pause_ratio > 0.25:
        feedback.append("Frequent pauses were observed, which may indicate hesitation.")
    if fillers_per_100 > 5:
        feedback.append("Frequent filler words reduced overall speech fluency.")
    if long_pause_count > 2:
        feedback.append("Long pauses during responses suggest a need for smoother flow.")
    if wpm < 90:
        feedback.append("Speaking rate was slow; increasing pace may improve confidence.")
    if wpm > 180:
        feedback.append("Speaking rate was very fast; slowing down may improve clarity.")
    if not feedback:
        feedback.append("Speech delivery was clear and fluent overall.")
    return feedback

def interpret_score(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Needs Improvement"
    else:
        return "Poor"

feedback = generate_feedback(pause_ratio, fillers_per_100_words, len(long_pauses), wpm)
score_label = interpret_score(final_score)

print("\n================ FINAL SPEECH REPORT ================")
print(f"Final Speech Score : {round(final_score,1)} / 100")
print(f"Performance Level  : {score_label}")
print("\nKey Feedback:")
for f in feedback:
    print("-", f)
