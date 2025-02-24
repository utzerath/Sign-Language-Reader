import os 
import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

def get_reference_image(label, dataset_dir="asl_dataset"):
    """
    Look inside dataset_dir/label/ for any JPEG file.
    Return the path of the first one we find, or None if none found.
    """
    label_folder = os.path.join(dataset_dir, label)
    if not os.path.exists(label_folder):
        return None

    for fname in os.listdir(label_folder):
        if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg"):
            return os.path.join(label_folder, fname)
    return None

def capture_user_signs(dataset_dir="asl_dataset", labels=None, num_images=5):
    """
    Captures user hand signs and saves images.
    """
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(26)]

    for label in labels:
        os.makedirs(os.path.join(dataset_dir, label), exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for label in labels:
            ref_path = get_reference_image(label, dataset_dir=dataset_dir)
            ref_img = cv2.imread(ref_path) if ref_path and os.path.exists(ref_path) else None

            if ref_img is not None:
                cv2.imshow("Reference Image", ref_img)
                cv2.waitKey(1)

            for countdown in [3, 2, 1]:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Sign '{label}' in {countdown}...", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Capture Hand Sign", frame)
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    return

            captured_count = 0
            while captured_count < num_images:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        x_min, x_max = w, 0
                        y_min, y_max = h, 0
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min, x_max = min(x_min, x), max(x_max, x)
                            y_min, y_max = min(y_min, y), max(y_max, y)

                        margin = 20
                        x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
                        x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

                        hand_crop = frame[y_min:y_max, x_min:x_max]
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        if hand_crop.size > 0:
                            save_filename = f"{label}_{time.time():.5f}.jpg"
                            save_path = os.path.join(dataset_dir, label, save_filename)
                            cv2.imwrite(save_path, hand_crop)
                            captured_count += 1
                            time.sleep(1)

                cv2.imshow("Capture Hand Sign", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if ref_img is not None:
                cv2.destroyWindow("Reference Image")

    cap.release()
    cv2.destroyAllWindows()
    print("Done capturing user signs.")

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_accuracy(accuracy_values):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.grid()
    plt.show()

def main():
    labels = [chr(ord('a') + i) for i in range(26)]
    capture_user_signs(dataset_dir="asl_dataset", labels=labels, num_images=10)
    
    # Example data for evaluation
    y_true = np.random.choice(labels, 100)
    y_pred = np.random.choice(labels, 100)
    accuracy_values = np.linspace(0.5, 0.95, num=10)
    
    plot_confusion_matrix(y_true, y_pred, labels)
    plot_accuracy(accuracy_values)

if __name__ == "__main__":
    main()
