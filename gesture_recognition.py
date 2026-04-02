import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Finger landmark indices ───────────────────────────────────────────────────
FINGER_TIPS   = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
FINGER_BASES  = [2, 5,  9, 13, 17]   # corresponding base/knuckle landmarks

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_BG      = (15,  15,  25)
COLOR_ACCENT  = (0,  220, 180)
COLOR_WHITE   = (240, 240, 240)
COLOR_GREEN   = (0,  200,  80)
COLOR_RED     = (0,   60, 220)
COLOR_YELLOW  = (0,  210, 255)


def get_finger_states(landmarks, handedness):
    """Returns a list of booleans: True if finger is extended."""
    states = []

    # Thumb (special case: compare x-axis based on hand side)
    tip  = landmarks[4]
    base = landmarks[3]
    if handedness == "Right":
        states.append(tip.x < base.x)   # right hand: tip is LEFT of base when up
    else:
        states.append(tip.x > base.x)   # left hand: tip is RIGHT of base when up

    # Other four fingers (compare y-axis: tip above pip joint = extended)
    for tip_id, base_id in zip(FINGER_TIPS[1:], [6, 10, 14, 18]):
        states.append(landmarks[tip_id].y < landmarks[base_id].y)

    return states  # [thumb, index, middle, ring, pinky]


def classify_gesture(finger_states, landmarks):
    """Map finger states to a gesture name."""
    thumb, index, middle, ring, pinky = finger_states

    # ── All fingers closed ─────────────────────────────────────────────────
    if not any(finger_states):
        return "✊  Fist"

    # ── All fingers open ──────────────────────────────────────────────────
    if all(finger_states):
        return "🖐  Open Hand"

    # ── Thumbs up/down (only thumb extended) ──────────────────────────────
    if thumb and not index and not middle and not ring and not pinky:
        thumb_tip_y = landmarks[4].y
        wrist_y     = landmarks[0].y
        if thumb_tip_y < wrist_y - 0.05:
            return "👍  Thumbs Up"
        else:
            return "👎  Thumbs Down"

    # ── Peace / Victory (index + middle only) ─────────────────────────────
    if not thumb and index and middle and not ring and not pinky:
        return "✌️  Peace"

    # ── Pointing (index only) ─────────────────────────────────────────────
    if not thumb and index and not middle and not ring and not pinky:
        return "👆  Pointing"

    # ── Spider-Man / Rock (index + pinky) ─────────────────────────────────
    if not thumb and index and not middle and not ring and pinky:
        return "🤘  Rock On"

    # ── OK sign (thumb + index close, others open) ────────────────────────
    if middle and ring and pinky:
        dx = abs(landmarks[4].x - landmarks[8].x)
        dy = abs(landmarks[4].y - landmarks[8].y)
        if dx < 0.05 and dy < 0.05:
            return "👌  OK"

    # ── Three fingers (index + middle + ring) ────────────────────────────
    if not thumb and index and middle and ring and not pinky:
        return "🤟  Three"

    return "🤔  Unknown"


def draw_ui(frame, gesture_name, hand_count, fps):
    """Overlay gesture label, FPS, and hand count on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "Gesture Recognition", (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_ACCENT, 2, cv2.LINE_AA)

    # FPS counter (top right)
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(frame, fps_text, (w - 130, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 1, cv2.LINE_AA)

    # Hand count
    hc_text = f"Hands: {hand_count}"
    cv2.putText(frame, hc_text, (w - 130, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)

    # Bottom gesture banner
    if gesture_name:
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 80), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(banner_overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, gesture_name, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_GREEN, 3, cv2.LINE_AA)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check if it is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0

    print("=" * 50)
    print("  Gesture Recognition System — Running")
    print("  Press  Q  to quit")
    print("=" * 50)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)   # mirror for natural feel
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_label = ""
            hand_count    = 0

            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)

                for hand_landmarks, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    lm_list    = hand_landmarks.landmark
                    handedness = hand_info.classification[0].label  # "Left" / "Right"

                    finger_states = get_finger_states(lm_list, handedness)
                    gesture_label = classify_gesture(finger_states, lm_list)

            # FPS calculation
            import time
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            frame = draw_ui(frame, gesture_label, hand_count, fps)

            cv2.imshow("Gesture Recognition System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
