import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
from screeninfo import get_monitors

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        height, width, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def calculate_vectors(hand_landmarks):
    vectors = []
    for connection in solutions.hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        p1 = hand_landmarks[start_idx]
        p2 = hand_landmarks[end_idx]
        vector = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        vectors.append((start_idx, end_idx, vector))
    return vectors

def resize_to_fit_screen(image, max_fraction=0.9):
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height
    img_h, img_w = image.shape[:2]
    scale = min((screen_w * max_fraction) / img_w, (screen_h * max_fraction) / img_h)
    return cv2.resize(image, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA)

def save_detection_data(file_base, detection_result):
    txt_path = f"{file_base}_landmarks.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0].category_name
            f.write(f"Dłoń {idx + 1} ({handedness}):\n")
            for i, lm in enumerate(hand_landmarks):
                f.write(f"  Punkt {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}\n")

            f.write("  Wektory:\n")
            for start_idx, end_idx, vector in calculate_vectors(hand_landmarks):
                vx, vy, vz = vector
                f.write(f"    {start_idx}→{end_idx}: ({vx:.3f}, {vy:.3f}, {vz:.3f})\n")
            f.write("\n")

# === GŁÓWNA CZĘŚĆ ===
image_path = "Dataset/y/y_000002.jpg"
output_base = "output/y/y_000002"

os.makedirs("output", exist_ok=True)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

image = mp.Image.create_from_file(image_path)
detection_result = detector.detect(image)

if detection_result.hand_landmarks:
    annotated = draw_landmarks_on_image(image.numpy_view(), detection_result)
    resized = resize_to_fit_screen(annotated)

    # Zapis obrazu
    cv2.imwrite(f"{output_base}_annotated.jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # Zapis danych
    save_detection_data(output_base, detection_result)

    # Podgląd
    cv2.imshow("Dłoń - wykrycie", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Dłoń wykryta, dane i obraz zapisane.")
else:
    print("Nie wykryto dłoni na obrazie.")
