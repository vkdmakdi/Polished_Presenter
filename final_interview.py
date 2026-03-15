import argparse
import json
import math
import time
from typing import List, Tuple, Dict
import cv2
import numpy as np
from tqdm import tqdm

# Direct imports to bypass the MediaPipe solutions bug
import mediapipe as mp
# For your specific installation, the path is:
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import hands as mp_hands
# Config
VIDEO_OUT = "annotated_output.mp4"
REPORT_JSON = "interview_report.json"
SAMPLE_FPS = 5

# Thresholds 
EYE_IDEAL_LOW = 0.6
EYE_IDEAL_HIGH = 0.8
HEAD_LOW = 2.0
HEAD_HIGH = 15.0
BLINK_EAR_THRESH = 0.22
BLINK_MIN_FRAMES = 1

# New weights
WEIGHT_EYE = 0.25
WEIGHT_HEAD = 0.20
WEIGHT_BLINK = 0.15
WEIGHT_POSTURE = 0.15
WEIGHT_HAND = 0.15
WEIGHT_FACE = 0.10

# Landmark indices
LEFT_EYE = [33, 133, 159, 145, 153, 154]
RIGHT_EYE = [362, 263, 386, 374, 380, 381]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]
NOSE_TIP = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FOREHEAD = 10
LEFT_EYE_H = (33, 133)
LEFT_EYE_V = (159, 145)
RIGHT_EYE_H = (362, 263)
RIGHT_EYE_V = (386, 374)
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
MOUTH_VERT = (13, 14)  
MOUTH_HORZ = (78, 308)    
LEFT_BROW_PAIR = (65, 158)  
RIGHT_BROW_PAIR = (295, 385)

# Hand processing
HAND_WRIST_IDX = 0

# Utility
def to_py(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v

# Scoring helpers
def eye_contact_score(ratio: float) -> float:
    if ratio < EYE_IDEAL_LOW:
        return max(0.0, ratio / EYE_IDEAL_LOW)
    if ratio <= EYE_IDEAL_HIGH:
        return 1.0
    return max(0.0, 1 - (ratio - EYE_IDEAL_HIGH) / (1 - EYE_IDEAL_HIGH))

def head_movement_score(m: float) -> float:
    if m < HEAD_LOW:
        return max(0.0, m / HEAD_LOW)
    if m <= HEAD_HIGH:
        return 1.0
    return max(0.0, 1 - (m - HEAD_HIGH) / HEAD_HIGH)

def blink_rate_score(blinks_per_min: float) -> float:
    IDEAL_LOW = 8.0
    IDEAL_HIGH = 20.0
    if blinks_per_min < IDEAL_LOW:
        return max(0.0, blinks_per_min / IDEAL_LOW)
    if blinks_per_min <= IDEAL_HIGH:
        return 1.0
    return max(0.0, 1 - (blinks_per_min - IDEAL_HIGH) / IDEAL_HIGH)

def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(pts: np.ndarray, horiz_idx: Tuple[int, int], vert_idx: Tuple[int, int]) -> float:
    h0 = pts[horiz_idx[0]]
    h1 = pts[horiz_idx[1]]
    v0 = pts[vert_idx[0]]
    v1 = pts[vert_idx[1]]
    horiz = euclid(h0, h1)
    vert = euclid(v0, v1)
    if horiz <= 1e-6:
        return 0.0
    return vert / horiz

def mouth_aspect_ratio(pts: np.ndarray) -> float:
    v = euclid(pts[MOUTH_VERT[0]], pts[MOUTH_VERT[1]])
    h = euclid(pts[MOUTH_HORZ[0]], pts[MOUTH_HORZ[1]])
    if h <= 1e-6:
        return 0.0
    return v / h

def brow_distance_norm(pts: np.ndarray, face_width: float) -> float:
    d_left = euclid(pts[LEFT_BROW_PAIR[0]], pts[LEFT_BROW_PAIR[1]])
    d_right = euclid(pts[RIGHT_BROW_PAIR[0]], pts[RIGHT_BROW_PAIR[1]])
    avg = (d_left + d_right) / 2.0
    if face_width <= 1e-6:
        return 0.0
    return avg / face_width

def expression_score(mar: float, brow_dist_norm: float) -> float:
    score = 0.6  # baseline
    if mar > 0.35:
        score += 0.35
    elif mar < 0.22:
        score -= 0.25

    if brow_dist_norm < 0.14:
        score -= 0.25
    elif brow_dist_norm > 0.18:
        score += 0.1

    return float(np.clip(score, 0.0, 1.0))

def posture_score(pts: np.ndarray) -> float:
    forehead_y = pts[FOREHEAD][1]
    cheeks_y = (pts[LEFT_CHEEK][1] + pts[RIGHT_CHEEK][1]) / 2.0
    chin_y = pts[CHIN][1]
    face_height = max(chin_y - forehead_y, 1e-6)
    delta = (cheeks_y - forehead_y) / face_height 
    if delta < 0.9:
        return 0.4
    if delta < 1.05:
        return 0.75
    return 1.0

def hand_movement_score(mean_motion: float, frame_diag: float) -> float:
    if frame_diag <= 1e-6:
        return 1.0
    norm = mean_motion / frame_diag
    if norm < 0.01:
        return 1.0
    if norm < 0.03:
        return 0.7
    return 0.3

# Main processing
def run(video_path: str, video_out: str = VIDEO_OUT, report_json: str = REPORT_JSON) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    # metrics
    eye_hits = 0
    eye_score_sum = 0.0
    valid_frames = 0
    head_moves: List[float] = []
    prev_angle = None
    blink_count = 0
    blink_frame_counter = 0

    # new accumulators
    mar_values: List[float] = []
    brow_norms: List[float] = []
    posture_vals: List[float] = []
    hand_motion_vals: List[float] = []

    # video properties
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0.0
    frame_diag = math.hypot(w, h)

    out = cv2.VideoWriter(
        video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    eye_ok_flags: List[int] = []
    sampled_times: List[float] = []

    pbar = tqdm(total=total_frames or None, unit="fr")

    idx = 0
    sampled_idx = 0

    required_indices = (
        LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS +
        LEFT_EYE + RIGHT_EYE +
        LEFT_IRIS + RIGHT_IRIS +
        [NOSE_TIP, CHIN, LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD] +
        list(LEFT_EYE_H) + list(LEFT_EYE_V) + list(RIGHT_EYE_H) + list(RIGHT_EYE_V) +
        list(MOUTH_VERT) + list(MOUTH_HORZ) + list(LEFT_BROW_PAIR) + list(RIGHT_BROW_PAIR)
    )
    max_idx_needed = max(required_indices)

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    prev_wrist_pos = None
    prev_wrist_pos = None

# Use the modules we imported at the very top of the script
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                draw = frame.copy()

                if idx % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = face_mesh.process(rgb)
                    res_hands = hands.process(rgb)

                    eye_ok = False
                    eye_score_frame = 0.0

                    if res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0].landmark
                        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

                        if pts.shape[0] <= max_idx_needed:
                            eye_ok = False
                        else:
                            nose_tip = pts[NOSE_TIP]
                            chin = pts[CHIN]
                            angle = math.degrees(math.atan2(chin[1] - nose_tip[1], chin[0] - nose_tip[0]))
                            if prev_angle is not None:
                                head_moves.append(abs(angle - prev_angle))
                            prev_angle = angle
                            left_eye = pts[LEFT_EYE_CORNERS]
                            right_eye = pts[RIGHT_EYE_CORNERS]

                            left_eye_width = np.linalg.norm(left_eye[0] - left_eye[1])
                            right_eye_width = np.linalg.norm(right_eye[0] - right_eye[1])
                            denom = max(left_eye_width, right_eye_width, 1e-6)
                            eye_symmetry = abs(left_eye_width - right_eye_width) / denom
                            eye_score_frame = np.clip(1.0 - (eye_symmetry / 0.18), 0.0, 1.0)
                            if abs(angle) > 20.0:
                                eye_score_frame *= 0.4
                            eye_ok = eye_score_frame >= 0.25
                            ear_left = eye_aspect_ratio(pts, LEFT_EYE_H, LEFT_EYE_V)
                            ear_right = eye_aspect_ratio(pts, RIGHT_EYE_H, RIGHT_EYE_V)
                            ear = (ear_left + ear_right) / 2.0

                            if ear < BLINK_EAR_THRESH:
                                blink_frame_counter += 1
                            else:
                                if blink_frame_counter >= BLINK_MIN_FRAMES:
                                    blink_count += 1
                                blink_frame_counter = 0
                            mar = mouth_aspect_ratio(pts)
                            face_width = euclid(pts[LEFT_CHEEK], pts[RIGHT_CHEEK])
                            brow_norm = brow_distance_norm(pts, face_width)

                            mar_values.append(mar)
                            brow_norms.append(brow_norm)
                            pscore = posture_score(pts)
                            posture_vals.append(pscore)
                            try:
                                cv2.circle(draw, tuple(nose_tip.astype(int)), 3, (255, 0, 0), -1)
                                cv2.putText(draw, f"Angle: {angle:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                                cv2.putText(draw, f"EAR L:{ear_left:.2f} R:{ear_right:.2f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                                cv2.putText(draw, f"MAR:{mar:.2f} BrowN:{brow_norm:.3f} Post:{pscore:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)
                            except Exception:
                                pass
                            eye_hits += int(eye_ok)
                            eye_score_sum += float(eye_score_frame)
                            valid_frames += 1
                    else:
                        eye_ok = False

                    eye_ok_flags.append(1 if eye_ok else 0)
                    sampled_times.append(idx / fps)
                    sampled_idx += 1
                    wrist_centroid = None
                    if res_hands.multi_hand_landmarks:
                        wrists = []
                        for hl in res_hands.multi_hand_landmarks:
                            wrist = hl.landmark[HAND_WRIST_IDX]
                            wrists.append(np.array([wrist.x * w, wrist.y * h], dtype=np.float32))
                        if wrists:
                            wrist_centroid = np.mean(np.stack(wrists), axis=0)

                    if wrist_centroid is not None:
                        if prev_wrist_pos is not None:
                            hand_motion_vals.append(np.linalg.norm(wrist_centroid - prev_wrist_pos))
                        prev_wrist_pos = wrist_centroid
                    else:
                        pass

                valid_sampled = max(1, len(eye_ok_flags))
                running_eye_pct = (sum(eye_ok_flags) / valid_sampled) * 100.0
                last_eye = "YES" if (len(eye_ok_flags) and eye_ok_flags[-1]) else "NO"

                cv2.putText(draw, f"Eye contact: {last_eye}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0) if last_eye=='YES' else (0,0,255), 2)
                cv2.putText(draw, f"Eye %: {running_eye_pct:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(draw, f"Blink count: {blink_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

                out.write(draw)
                idx += 1
                pbar.update(1)

        finally:
            pbar.close()
            cap.release()
            out.release()

    if blink_frame_counter >= BLINK_MIN_FRAMES:
        blink_count += 1

    # Aggregate & scoring
    if valid_frames == 0:
        eye_contact_pct = 0.0
        binary_eye_ratio = 0.0
    else:
        eye_contact_pct = (eye_score_sum / valid_frames) * 100.0
        binary_eye_ratio = eye_hits / valid_frames

    mean_head_motion = float(np.mean(head_moves)) if head_moves else 0.0
    blinks_per_min = (blink_count / (duration_sec / 60.0)) if duration_sec > 0 else 0.0

    mean_mar = float(np.mean(mar_values)) if mar_values else 0.0
    mean_brow_norm = float(np.mean(brow_norms)) if brow_norms else 0.0
    mean_posture = float(np.mean(posture_vals)) if posture_vals else 0.0
    mean_hand_motion = float(np.mean(hand_motion_vals)) if hand_motion_vals else 0.0

    # Subscores
    s_eye = eye_contact_score(eye_contact_pct / 100.0)
    s_head = head_movement_score(mean_head_motion)
    s_blink = blink_rate_score(blinks_per_min)
    s_face = expression_score(mean_mar, mean_brow_norm)
    s_posture = float(np.clip(mean_posture, 0.0, 1.0))
    s_hand = hand_movement_score(mean_hand_motion, frame_diag)

    final_percent = round(float(
        WEIGHT_EYE * s_eye +
        WEIGHT_HEAD * s_head +
        WEIGHT_BLINK * s_blink +
        WEIGHT_POSTURE * s_posture +
        WEIGHT_HAND * s_hand +
        WEIGHT_FACE * s_face
    ) * 100, 1)

    if final_percent >= 75:
        classification = "Confident"
    elif final_percent >= 50:
        classification = "Somewhat confident / neutral"
    else:
        classification = "Nervous / needs improvement"

    feedback: List[str] = []
    if s_eye < 0.6:
        feedback.append("Improve eye contact: look toward the camera more consistently.")
    if s_head < 0.6:
        feedback.append("Reduce head movement; keep your head steady.")
    if s_blink < 0.6:
        feedback.append("Blink rate suggests nervousness; try to relax.")
    if s_hand < 0.6:
        feedback.append("Hands show fidgeting; rest them calmly or place them on the table.")
    if s_posture < 0.6:
        feedback.append("Posture appears slouched; sit upright and open your chest.")
    if s_face < 0.6:
        feedback.append("Facial expression appears tense; try gentle smiles and relax forehead.")
    if not feedback:
        feedback.append("Good presence: steady eye contact, posture and expression.")

    # JSON report
    report = {
        "video_path": video_path,
        "duration_sec": to_py(duration_sec),
        "sampled_frames": int(valid_frames),
        "eye_contact_pct_soft": to_py(eye_contact_pct),
        "eye_contact_pct_binary": to_py(binary_eye_ratio * 100.0),
        "mean_head_motion_deg": to_py(mean_head_motion),
        "blink_count": int(blink_count),
        "blinks_per_min": to_py(blinks_per_min),
        "face_metrics": {
            "mean_mar": to_py(mean_mar),
            "mean_brow_norm": to_py(mean_brow_norm),
            "expression_score": to_py(s_face),
        },
        "posture_score": to_py(s_posture),
        "hand_motion_mean_px": to_py(mean_hand_motion),
        "hand_movement_score": to_py(s_hand),
        "subscores": {
            "eye_score": to_py(s_eye),
            "head_score": to_py(s_head),
            "blink_score": to_py(s_blink),
            "posture_score": to_py(s_posture),
            "hand_score": to_py(s_hand),
            "face_score": to_py(s_face),
        },
        "final_score_percent": to_py(final_percent),
        "classification": classification,
        "feedback": feedback,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Interview Analysis ---")
    print(f"Duration: {duration_sec:.1f} s   Sampled frames: {valid_frames}")
    print(f"Eye contact (soft avg): {eye_contact_pct:.1f}%   (binary %: {binary_eye_ratio*100:.1f}%)")
    print(f"Head movement (avg Δ°): {mean_head_motion:.2f}")
    print(f"Blink count: {blink_count}  ({blinks_per_min:.1f}/min)")
    print(f"MAR: {mean_mar:.3f}   BrowNorm: {mean_brow_norm:.3f}")
    print(f"Posture score: {s_posture:.2f}   Hand motion mean(px): {mean_hand_motion:.1f}  Hand score: {s_hand:.2f}")
    print(f"Expression score: {s_face:.2f}")
    print(f"Confidence score: {final_percent} / 100 -> {classification}")
    print(f"Annotated video saved to: {video_out}")
    print(f"JSON report saved to: {report_json}")
    print("\nFeedback:")
    for fmsg in feedback:
        print(f"- {fmsg}")

# CLI
def main():
    parser = argparse.ArgumentParser(description="Analyze interview video for eye contact/head movement/blinks + posture/face/hands.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--out", "-o", default=VIDEO_OUT, help="Path to annotated output video")
    parser.add_argument("--report", "-r", default=REPORT_JSON, help="Path to output JSON report")
    args = parser.parse_args()
    run(args.video, video_out=args.out, report_json=args.report)

if __name__ == "__main__":
    main()
