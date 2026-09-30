import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def clean_and_crop_mri(image_path, img_size=(224, 224)):
    """Loads an MRI scan, crops black borders, and enhances tissue contrast using CLAHE."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to separate brain tissue from dark background
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find brain boundary contour
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = gray[y:y+h, x:x+w]
    else:
        cropped = gray

    resized = cv2.resize(cropped, img_size)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    return enhanced

def process_and_split_dataset(raw_dir="data/raw", processed_dir="data/processed", test_ratio=0.2):
    """Reads raw images, processes them, and splits into train/test sets (80/20)."""
    classes = ["glioma", "healthy"]
    
    for cls in classes:
        raw_path = os.path.join(raw_dir, cls)
        image_files = glob.glob(os.path.join(raw_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(raw_path, "*.[pP][nN][gG]")) + \
                      glob.glob(os.path.join(raw_path, "*.[jJ][pP][eE][gG]"))

        if not image_files:
            print(f"Warning: No raw images found in {raw_path}.")
            continue

        print(f"Found {len(image_files)} images for '{cls}'. Processing...")

        # Stratified Train-Test Split (80% Train, 20% Test)
        train_files, test_files = train_test_split(image_files, test_size=test_ratio, random_state=42)

        def save_set(file_list, split_type):
            output_dir = os.path.join(processed_dir, split_type, cls)
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, filepath in enumerate(tqdm(file_list, desc=f"Saving {split_type}/{cls}")):
                processed_img = clean_and_crop_mri(filepath)
                if processed_img is not None:
                    out_path = os.path.join(output_dir, f"{cls}_{idx:04d}.png")
                    cv2.imwrite(out_path, processed_img)

        save_set(train_files, "train")
        save_set(test_files, "test")

    print("\nPhase 1 Complete: Dataset cleaned and split into train/test sets.")

if __name__ == "__main__":
    process_and_split_dataset()