# Preprocess and segment chemical structure images
import cv2
import os

def segment_structures(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_folder, f"structure_{i+1}.png"), roi)
