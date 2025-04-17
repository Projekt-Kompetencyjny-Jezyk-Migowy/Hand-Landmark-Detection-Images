import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
from PIL import Image, ExifTags

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

#By nie obracało obrazu - Poprawne załadowanie zdjęcia z uwzględnieniem jego obrotu zapisanego w danych EXIF
def load_image_correct_orientation(path):
    image = Image.open(path)
    try:
        for orientation in ExifTags.TAGS: #orientation określa w jaki sposób aparat "obracał" zdjęcie
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None: #jeśli istnieje
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3: #obrócone o 180
                image = image.rotate(180, expand=True) #obraca odpowiednio obraz
            elif orientation_value == 6: #obrócone o 270
                image = image.rotate(270, expand=True)
            elif orientation_value == 8: #obrócone o 90
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return np.array(image.convert("RGB")) #Konwertuje obraz do RGB i zwraca jako tablicę NumPy

#rysuje landmarki i wektory dłoni na obrazie
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

#oblicza wektory
def calculate_vectors(hand_landmarks):
    vectors = []
    for connection in solutions.hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        p1 = hand_landmarks[start_idx]
        p2 = hand_landmarks[end_idx]
        vector = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        vectors.append((start_idx, end_idx, vector))
    return vectors



#Do formatowania wyników detekcji dłoni(czytelny zapis w txt)
def format_detection_data(detection_result, is_flipped=False):
    lines = [] # lista stringów
    #Przejście po każdej wykrytej dłoni
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[idx][0].category_name
        flip_note = " (flipped)" if is_flipped else ""
        lines.append(f"Dłoń {idx + 1} ({handedness}{flip_note}):")
        for i, lm in enumerate(hand_landmarks):
            lines.append(f"  Punkt {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
        lines.append("  Wektory:")
        for start_idx, end_idx, vector in calculate_vectors(hand_landmarks):
            vx, vy, vz = vector
            lines.append(f"    {start_idx}→{end_idx}: ({vx:.3f}, {vy:.3f}, {vz:.3f})")
        lines.append("")  # pusta linia między dłońmi
    return lines

#Przetwarzanie obrazów dla jednego ze znaku, zapisuje wyniki detekcji do pliku tekstowego
def process_and_label_images_for_sign(sign_label, input_dir, output_dir, detector):
    # Nowy folder wyjściowy na dane
    os.makedirs(output_dir, exist_ok=True)

    # Lista plików .jpg w katalogu
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"{sign_label}: Brak obrazów do przetworzenia.") #czy pusty
        return

    all_lines = []
    #przechodzi przez każdy obraz w katalogu input_dir
    for img_name in image_files:
        input_path = os.path.join(input_dir, img_name)
        base_filename = os.path.splitext(img_name)[0] #nazwa pliku bez rozszerzenia

        #Ładowanie i wykrywanie obrazu danego znaku.
        image = mp.Image.create_from_file(input_path)
        detection_result = detector.detect(image)

        if detection_result.hand_landmarks:
            all_lines.append(f"==== {base_filename} ====")
            all_lines.extend(format_detection_data(detection_result, is_flipped=False))

            # Flip (odbicie lustrzane)
            flipped_np = cv2.flip(image.numpy_view(), 1)
            flipped_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped_np)
            flipped_result = detector.detect(flipped_mp)

            if flipped_result.hand_landmarks:
                all_lines.append(f"==== {base_filename}_flipped ====")
                all_lines.extend(format_detection_data(flipped_result, is_flipped=True))

            print(f"{img_name}: dane dodane.")
        else:
            print(f"{img_name}: NIE wykryto dłoni")

    # Jeśli mamy dane, zapisujemy je do jednego pliku na symbol
    if all_lines:
        # Tworzy pełną ścieżkę do pliku wyjściowego
        output_path = os.path.join(output_dir, f"{sign_label}.txt")
        #otwiera plik to zapisu
        with open(output_path, "w", encoding="utf-8") as f:
            #zapisuje całe zebrane wczesniej dane
            f.write("\n".join(all_lines))
    else:
        #Usuwa folder wyjściowy, ponieważ jest pusty
        os.rmdir(output_dir)

#przechodzi przez wszystkie foldery w katalogu wejściowym (np. „a”, „b”, „a+” itd.)
def process_full_dataset(base_input_dir, base_output_dir, detector):

    for symbol_name in os.listdir(base_input_dir):
        input_dir = os.path.join(base_input_dir, symbol_name)
        if os.path.isdir(input_dir): #Sprawdzanie, czy element jest katalogiem
            print(f" Przetwarzanie symbolu: {symbol_name}")

            # Folder wyjściowy dla danego symbolu
            output_dir = os.path.join(base_output_dir, symbol_name)

            # Przetwarza obrazy dla danego znaku
            process_and_label_images_for_sign(symbol_name, input_dir, output_dir, detector)

#przetwarza wszystkie zdjęcia z datasetu (tworzy obrazy z landmarkami i wektorami i zapisuje je do folderu)
def detection_all_images_from_folder(input_base="Dataset Migowy PJM", output_base="outputImages"):

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    os.makedirs(output_base, exist_ok=True)

    for label_folder in os.listdir(input_base):
        input_folder = os.path.join(input_base, label_folder)
        if not os.path.isdir(input_folder):
            continue

        output_folder = os.path.join(output_base, label_folder)
        os.makedirs(output_folder, exist_ok=True)

        for img_file in os.listdir(input_folder):
            if not img_file.lower().endswith((".jpg")):
                continue

            input_path = os.path.join(input_folder, img_file)
            output_name = os.path.splitext(img_file)[0]
            output_base_path = os.path.join(output_folder, output_name)

            # Wczytaj obraz z korektą orientacji
            img_np = load_image_correct_orientation(input_path)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
            detection_result = detector.detect(image)

            if detection_result.hand_landmarks:
                annotated = draw_landmarks_on_image(image.numpy_view(), detection_result)

                # Zapis obrazu i danych
                cv2.imwrite(f"{output_base_path}_annotated.jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                print(f"{img_file}: WYKRYTO dłonie i zapisano dane.")
            else:
                print(f"{img_file}: Brak wykrycia dłoni.")

#pobieranie wektorow i landmarkow do txt
def generate_hand_landmark_data(input_base = "Dataset Migowy PJM", output_base = "outputdetection_data"):

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Przetwarzanie całego datasetu
    process_full_dataset(input_base, output_base, detector)

#detection_all_images_from_folder()
generate_hand_landmark_data()