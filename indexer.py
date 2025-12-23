import face_recognition
import os
import pickle
import cv2
from tqdm import tqdm

SEARCH_DIR = "/Volumes/One Touch/Google_Photos"
OUTPUT_FILE = "face_data.pkl"

face_encodings_db = []

print("Building file list...")
image_files = []
for root, dirs, files in os.walk(SEARCH_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(root, f))

print(f"Processing {len(image_files)} images. This will take time.")

for path in tqdm(image_files):
    try:
        img = face_recognition.load_image_file(path)
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        encs = face_recognition.face_encodings(small_img)
        for e in encs:
            face_encodings_db.append({"path": path, "encoding": e})
    except Exception:
        continue

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(face_encodings_db, f)

print(f"Finished. Saved {len(face_encodings_db)} faces to {OUTPUT_FILE}")
