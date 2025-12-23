import face_recognition
import pickle
import numpy as np
import os
import shutil
import time

def search_person_vectorized(pkl_path, query_image_path, tolerance=0.4):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_folder = os.path.join(desktop_path, "Search_Results")
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 1. Load Data
    print(f"[*] Loading {os.path.basename(pkl_path)} into RAM...")
    start_load = time.time()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[+] Loaded {len(data)} entries in {time.time()-start_load:.2f}s")

    # 2. Extract Query Signature
    print(f"[*] Analyzing query image...")
    query_image = face_recognition.load_image_file(query_image_path)
    query_encodings = face_recognition.face_encodings(query_image)
    if not query_encodings:
        print("[-] Error: No face detected in query photo.")
        return
    target_encoding = np.array(query_encodings[0])

    # 3. Vectorized Math (The AI way)
    print(f"[*] Performing high-speed matrix search...")
    start_math = time.time()
    
    # Filter out any corrupted entries (like the 'encoding' strings we saw earlier)
    valid_data = [e for e in data if isinstance(e.get('encoding'), np.ndarray)]
    
    # Create a single large Matrix of all 94k signatures
    all_signatures = np.array([e['encoding'] for e in valid_data])
    
    # Calculate all distances at once
    distances = np.linalg.norm(all_signatures - target_encoding, axis=1)
    
    # Find indices where distance < tolerance
    match_indices = np.where(distances < tolerance)[0]
    best_dist = np.min(distances) if len(distances) > 0 else 1.0
    
    print(f"[+] Math complete in {time.time()-start_math:.4f}s")

    # 4. Physical Copying
    match_count = len(match_indices)
    print(f"[*] Found {match_count} matches. Copying files...")
    
    for i, idx in enumerate(match_indices):
        img_path = valid_data[idx]['path']
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(output_folder, f"match_{i+1}.jpg"))
        
        # Progress update every 10 files
        if (i + 1) % 10 == 0:
            print(f"    -> Progress: {i+1}/{match_count} files copied.")

    print(f"\n--- RESULTS ---")
    print(f"Closest distance in database: {best_dist:.4f}")
    print(f"Total files saved to Desktop: {match_count}")
    if match_count == 0:
        print(f"Tip: Closest was {best_dist:.4f}. If this is ~0.65, try tolerance 0.7.")

if __name__ == "__main__":
    PKL = "/Users/shreyasl/Desktop/Offline Recognition/face_data.pkl"
    IMG = "/Users/shreyasl/Downloads/PXL_20250731_205708140.MP.jpg"
    search_person_vectorized(PKL, IMG)
