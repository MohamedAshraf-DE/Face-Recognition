This is an excellent README. Below is the same structure and style, but adapted to your ORL Face Recognition Lab project with the background image at the very top, just like the MAGIC README.

text
# ORL Face Recognition Lab (PCA / LDA)

![ORL Face Lab](face.jpeg)

## Overview

This project implements a full face-recognition lab using the classic **ORL (AT&T) face dataset**, showcasing both **PCA (Eigenfaces)** and **LDA (Fisherfaces)** for identity recognition and similarity search.

The app is built as a modern Streamlit dashboard with a dark, high-tech UI and a face-recognition background image. It is designed for both **non-technical users** (simple “Identity Match” view) and **professors/engineers** (detailed metrics, confusion matrices, eigenfaces).

### Why This Matters

Biometric systems such as phone unlock and access control depend on accurate face recognition.  
This project turns a traditional academic PCA/LDA assignment into an interactive lab where users can:

- Test **1:1 identity verification** with distance thresholds and confidence.
- Explore **1:N similarity search** for “look-alike” faces.
- Visualize **how PCA/LDA see faces** through mean faces, eigenfaces, and reconstructions.
- Study **system reliability** via accuracy curves, FAR/FRR, and confusion matrices.

---

## Project Details

### Dataset

- **Source**: ORL / AT&T Face Database  
- **Subjects**: 40 persons  
- **Images per Subject**: 10 (frontal grayscale faces)  
- **Resolution**: 92 × 112 pixels (grayscale, resized if necessary)  
- **Format Expected**:

orl_dataset/
├── s1/1.pgm ... 10.pgm
├── s2/1.pgm ... 10.pgm
...
└── s40/1.pgm ... 10.pgm

text

- Each subject folder (`s1` … `s40`) represents one identity and contains 10 images.

### Data Preprocessing

1. **Loading**  
 - Reads all `.pgm` images from `orl_dataset/s1` … `s40`.  
 - Converts each image to grayscale (if needed) and resizes to 92 × 112.

2. **Vectorization**  
 - Each image is flattened into a vector of 10,304 pixels.  
 - Images are stacked into a data matrix \(X \in \mathbb{R}^{400 \times 10304}\).  
 - Labels \(y \in \{1,\dots,40\}\) encode the subject ID.

3. **Train–Test Splits**  
 - **5/5 Odd–Even**: odd indices for train, even for test (5 train + 5 test per subject).  
 - **7/3 First 7 Train**: first 7 images train, last 3 images test.

4. **Centering for PCA**  
 - Subtracts the mean face from each training image before computing covariance (snapshot method).

---

## Models Implemented

| Model              | Role                         | Notes                              |
|--------------------|-----------------------------|------------------------------------|
| PCA (Eigenfaces)   | Dimensionality reduction    | Snapshot method, α-controlled k   |
| LDA (Fisherfaces)  | Discriminative projection   | PCA → LDA pipeline, up to 39 dims |
| KNN (1,3,5,7)      | Classifier in feature space | Euclidean distance                 |

### PCA (Eigenfaces)

- Uses snapshot PCA: eigen-decomposition of \(A A^T\) (A = centered training data).  
- Keeps the smallest **k** such that cumulative variance ≥ α, with α ∈ {0.80, 0.85, 0.90, 0.95}.  
- Projects faces to a low-dimensional eigenspace and reconstructs images for visualization.

### LDA (Fisherfaces)

- Applies PCA first to reduce dimensionality and avoid singular scatter matrices.  
- Computes within-class (SW) and between-class (SB) scatter in PCA space.  
- Solves eigenproblem for \(S_W^{-1}S_B\) and keeps up to 39 Fisher directions.  
- Produces a feature space optimized for class separation.

---

## App Features

### 1. Identity Check / Verification (1:1)

- **Claimed ID input**: user selects a subject ID (1–40).  
- App picks a random test image for that ID as the **probe**.  
- Computes distance between probe and all enrolled samples of the claimed ID.  
- **Decision threshold** slider controls Accept / Reject behavior:
- Lower threshold → stricter security (lower FAR, higher FRR).
- Shows:
- Identity Match: **Yes/No** with a confidence progress bar.
- Distance score, **FAR** (False Accept Rate), **FRR** (False Reject Rate).
- Probe image, best enrolled match, and a difference map.

### 2. Similarity Search / Look‑Alikes (1:N)

- Select a **query index** from the test set.  
- Shows the query face and its true ID.  
- Finds the **Top‑K nearest neighbors** in the training set using KNN in PCA/LDA space.  
- For technical mode, shows rank, predicted ID, and distance; correct matches highlighted in green.

### 3. Model Lab / “How AI Sees Faces”

- **Mean Face** visualization.  
- **Top Eigenfaces** (principal components) displayed as images.  
- **Reconstruction demo**:
- Choose an image and number of components k.
- Compare original vs reconstructed image.
- Show reconstruction MSE and compression ratio.

### 4. Metrics & System Reliability

- Overall accuracy on the current split and method (PCA or LDA).  
- **Accuracy vs α** (for PCA) curve.  
- **Accuracy vs K** (KNN) curve for the selected method.  
- **Confusion matrix** heatmap (IDs 1–40) with PCA/LDA-specific color maps.  

---

## UI & Design

- **Framework**: Streamlit.  
- **Theme**: Dark, “Canva-style” dashboard with:
- Rounded cards and metrics.
- Gradient sliders and tab buttons.
- Neon progress bar for confidence.
- **Background Image**:
- Global background set from `face.jpeg` using base64 embedding.
- Dark overlay (`rgba(3,7,18,0.88)`) ensures charts and text remain readable.
- **Dual Mode Sidebar**:
- 👤 **General User**: Friendly wording, focus on decisions and confidence.  
- 🎓 **Technical / Professor**: Extra metrics, scores, curves, and technical captions.

---

## Installation

### System Requirements

- **OS**: Windows, macOS, or Linux  
- **Python**: 3.8+  
- **RAM**: ≥ 2 GB (ORL is small; more helps with plotting)  

### Python Dependencies

See `requirements.txt` for exact versions. Typical stack:

streamlit
numpy
opencv-python-headless
scikit-learn
matplotlib
seaborn

text

---

## Setup Instructions

### Step 1: Clone the Repository

git clone https://github.com/MohamedAshraf-DE/Face-Recognition.git
cd Face-Recognition

text

### Step 2: (Optional) Create Virtual Environment

**Windows**

python -m venv venv
venv\Scripts\activate

text

**macOS / Linux**

python3 -m venv venv
source venv/bin/activate

text

### Step 3: Install Dependencies

pip install -r requirements.txt

text

### Step 4: Verify Installation

python -c "import streamlit; print('Streamlit version:', streamlit.version)"

text

### Step 5: Run the Application

streamlit run app.py

text

The app will open in your browser at `http://localhost:8501`.

---

## Project Structure

Face-Recognition/
│
├── app.py # Streamlit ORL Face Lab
├── face_rec.ipynb # Full PCA/LDA notebook
├── requirements.txt # Python dependencies
├── face.jpeg # Background/banner image
│
├── orl_dataset/ # ORL dataset (40 × 10 faces)
│ ├── s1/1.pgm ... 10.pgm
│ ├── ...
│ └── s40/1.pgm ... 10.pgm
│
└── .gitignore # Git ignore configuration

text

---

## Usage Guide

### 1. Running the Lab

streamlit run app.py

text

Choose **Mode**, **Dataset path**, **Split**, **Method**, and **K** from the sidebar to explore the different tabs.

### 2. Tabs & Typical Workflow

1. **Identity Check / Verification**  
   - Pick a claimed ID.  
   - Move the threshold slider and observe how Accept/Reject, confidence, FAR, and FRR change.

2. **Similarity Search / Look‑Alikes**  
   - Slide through test indices to see different query faces.  
   - Inspect top‑K most similar training faces and distances.

3. **Model Lab**  
   - Switch Method to PCA in sidebar.  
   - Inspect mean face and eigenfaces.  
   - Try different k values for reconstruction and watch MSE drop.

4. **Metrics & Evaluation**  
   - Compare PCA vs LDA behavior using accuracy and confusion matrices.  
   - Analyze how α (variance kept) and K (KNN) influence performance.

---

## Machine Learning Workflow

### Phase 1: Data Loading & Splitting

- Load all ORL images from `orl_dataset/s1` … `s40`.  
- Flatten into vectors and build data matrix \(X\) and labels \(y\).  
- Apply one of the two train–test split strategies (5/5 odd–even or 7/3 first 7).

### Phase 2: PCA (Eigenfaces)

- Compute mean face and subtract it from training images.  
- Use snapshot method to find eigenvalues/eigenvectors of covariance.  
- Select k components according to chosen variance threshold α.  
- Project training and test sets into PCA space.

### Phase 3: LDA (Fisherfaces)

- Start from PCA‑projected features.  
- Compute between-class and within-class scatter matrices.  
- Solve generalized eigenproblem to obtain Fisherfaces (up to 39 dims).  
- Project training and test sets into LDA space.

### Phase 4: Classification & Evaluation

- Run KNN classifier (k ∈ {1, 3, 5, 7}) in PCA or LDA space.  
- Compute:
  - Accuracy
  - Confusion matrix
  - FAR/FRR (for 1:1 verification scenario)
  - Accuracy–α and Accuracy–K curves (technical mode).

---

## Technologies Used

| Category           | Technologies                 |
|--------------------|-----------------------------|
| Dimensionality Red.| NumPy (manual PCA), LDA     |
| ML / Metrics       | scikit-learn (KNN, metrics) |
| Image Processing   | OpenCV (cv2)                |
| Visualization      | Matplotlib, Seaborn         |
| Web UI             | Streamlit                   |
| Version Control    | Git, GitHub                 |
| Development        | Jupyter Notebook, VS Code   |

---

## Author

**Mohamed Ashraf**  
- **GitHub**: [@MohamedAshraf-DE](https://github.com/MohamedAshraf-DE)  
- **Email**: mohammed.ashraf.m.w@gmail.com  
- **Focus Areas**: Machine Learning, Signal Processing, Face Recognition  
- **Location**: Egypt  

---

## License

This project is open source and available under the **MIT License**.  
See the `LICENSE` file for details.

---

## Future Enhancements

- [x] PCA & LDA implementation with ORL dataset  
- [x] Streamlit interactive lab (verification, similarity, model insights)  
- [ ] Camera / upload support for real-world photos  
- [ ] Live face detection and alignment before recognition  
- [ ] Web deployment (Streamlit Community Cloud / other)  
- [ ] More classifiers (SVM, modern deep features)  
- [ ] Detailed report export (PDF/HTML)  

---

## Troubleshooting

### Issue: Dataset load failed (Missing file: `orl_dataset/s1/1.pgm`)
- Ensure the ORL dataset is present under `orl_dataset/s1` … `s40`.  
- Check the root path in the sidebar and adjust if your dataset is elsewhere.

### Issue: `ModuleNotFoundError` (e.g., `cv2`, `streamlit`)
- Run `pip install -r requirements.txt` in your environment.

### Issue: App does not start
- Verify Python 3.8+ is installed and on PATH.  
- Try `streamlit hello` to ensure Streamlit is installed.

---

**Last Updated**: December 2025  
**Version**: 1.0.0  
**Status**: ✅ Completed ORL Face Lab with interactive Streamlit UI
