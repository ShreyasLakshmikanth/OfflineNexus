# OfflineNexus 🛡️

**OfflineNexus** is a privacy-focused AI engine model designed to index and query massive image archives locally. This project was personally developed to solve the "Digital Orphan" problem—managing 114,000+ unorganized images without relying on cloud-based AI providers.

## 🚀 The Engineering Challenge
Processing **114,736 images** (approx. 9 hours of indexing) creates significant data bottlenecks. `OfflineNexus` addresses these through:

- **Vectorized Search:** Instead of slow Python loops, we utilize **NumPy Vectorization** to calculate Euclidean distances ($L^2$ norm) across 94,000+ 128-dimensional embeddings in under 0.5 seconds.
- **Hardware Optimization:** Specifically tuned to handle the I/O latency of **external mechanical HDDs** (One Touch drives) by decoupling the mathematical search from the physical file retrieval.
- **Biometric Precision:** Defaulted to a `0.4` tolerance threshold. This "High Precision" mode minimizes false positives, ensuring that from a database of 94k images, only the most statistically certain matches are retrieved.



## 🛠 Project Structure
- `indexer.py`: Scans directories recursively, extracts facial features, and serializes them into a `face_data.pkl` database.
- `searcher.py`: Performs high-speed similarity searches using a target query image.

## 🔐 Privacy & Security
- **Air-Gapped Logic:** No data is sent to external APIs.
- **Zero-Data GitHub:** The `.gitignore` ensures that biometric signatures (`.pkl`) and raw image data never leave the local machine.

## ⚙️ Setup

1. **Create a Virtual Environment:** `python3 -m venv face_offline`
2. **Activate the Environment:** `source face_offline/bin/activate`
3. **Install AI Libraries & Dependencies:** `pip install -r requirements.txt`

---
*Developed as part of an exploration into solving daily problems by building AI models.*
