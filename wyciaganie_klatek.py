import os
import cv2
import numpy as np

def pad_to_square(image):
    height, width = image.shape[:2]
    size = max(height, width)
    padded_image = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - width) // 2
    y_offset = (size - height) // 2
    padded_image[y_offset:y_offset+height, x_offset:x_offset+width] = image
    return padded_image

def correct_rotation(frame):
    # If the frame is taller than wide, assume it was rotated and fix it
    h, w = frame.shape[:2]
    if h > w:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def extract_frames(video_path, output_folder, num_frames=10):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing video: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")

    if frame_count < num_frames:
        print(f"Video {video_name} has fewer than {num_frames} frames. Skipping.")
        cap.release()
        return

    frame_indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)

    for i, idx in enumerate(frame_indices, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            square_frame = pad_to_square(frame)
            frame_filename = f"{video_name}_frame{i}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            square_frame = cv2.rotate(square_frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(frame_path, square_frame)
            print(f"Saved frame {i} at index {idx} -> {frame_filename}")


            cv2.rotate(square_frame, cv2.ROTATE_90_CLOCKWISE)
            mirrored_frame = cv2.flip(square_frame, 1)
            mirror_filename = f"m_{video_name}_frame{i}.jpg"
            mirror_path = os.path.join(output_folder, mirror_filename)
            cv2.imwrite(mirror_path, mirrored_frame)
            print(f"Saved mirrored frame {i} -> {mirror_filename}")
        else:
            print(f"Failed to read frame {idx} from {video_name}")

    cap.release()

    #os.remove(video_path)
    print(f"Deleted original video: {video_path}")

def process_pjm_folder(root_folder):
    print(f"\nStarting processing for root folder: {root_folder}")

    for dir_entry in os.scandir(root_folder):
        if dir_entry.is_dir():
            folder_name = dir_entry.name
            video_folder = dir_entry.path
            images_folder = os.path.join(root_folder, folder_name + "_images")

            print(f"\nFound folder: {folder_name}")
            print(f"Creating/using images folder: {images_folder}")
            os.makedirs(images_folder, exist_ok=True)

            for file_entry in os.scandir(video_folder):
                if file_entry.is_file() and file_entry.name.lower().endswith(('.mp4', '.mov')):
                    video_path = file_entry.path
                    extract_frames(video_path, images_folder)

    print("\nAll videos processed.\n")

# Example usage
pjm_root = "PJM"  # Adjust path as needed
process_pjm_folder(pjm_root)
