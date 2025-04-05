import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np



MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#Kopiuje obraz wejściowy. Iteruje po wykrytych dłoniach, rysując landmarki (punkty dłoni).
#Oznacza na obrazie, czy dłoń jest lewa czy prawa.
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Wyznacza gdzie napisy left/right
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Wypisuje napisy left/right na obrazie
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

#Wypisuje w terminalu informacje o wykrytych dłoniach:Indeks dłoni, Czy to prawa czy lewa dłoń,
#Lista współrzędnych (x, y, z) landmarków
def print_detection_result(detection_result):
    # Loop through each hand in the detection result
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[idx]
        print(f"\nHand {idx + 1}:")
        print(f"Handedness: {handedness[0].category_name}")

        # Print landmarks coordinates (normalized)
        for i, landmark in enumerate(hand_landmarks):
            print(f"Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")

#Oblicza wektory między punktami dłoni, bazując na standardowych połączeniach dłoni w Mediapipe.
def calculate_vectors(hand_landmarks):
    vectors = []
    for connection in solutions.hands.HAND_CONNECTIONS:  # Połączenia między punktami
        start_idx, end_idx = connection
        p1 = hand_landmarks[start_idx]
        p2 = hand_landmarks[end_idx]

        vector = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        vectors.append((start_idx, end_idx, vector))

    return vectors

# Wypisuje w terminalu listę wektorów między landmarkami, ułatwiając analizę ruchu dłoni.
def print_vectors(detection_result):
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[idx]
        print(f"\n Dłoń {idx + 1} ({handedness[0].category_name}):")

        vectors = calculate_vectors(hand_landmarks)
        for start_idx, end_idx, vector in vectors:
            print(f"Wektor {start_idx} → {end_idx}: ({vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f})")


# STEP 2: Create an HandLandmarker object, Tworzenie detektora dłoni
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2) #max 2 dlonie w obrazie
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("photos/woman_hands.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)
if detection_result.hand_landmarks: #jeżeli została wykryta dłoń
    # rysowanie landmarkow na obrazie
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # wyświetla obraz z zaznaczonymi punktami i połączeniami.
    cv2.imshow('Hand Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)  # Wait for a key press before closing the image window
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # wypisuje punkty i wektory
    print_detection_result(detection_result)
    print_vectors(detection_result)
else:
    print("Nie wykryto dłoni na obrazie.")