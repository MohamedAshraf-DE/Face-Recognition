import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import base64  # for background image

# -----------------------------
# Global layout + theming
# -----------------------------
st.set_page_config(
    page_title="ORL Face Lab (PCA/LDA)",
    layout="wide",
    page_icon="🧠",
)

# ---------- Background image (face.jpeg) ----------
def get_base64(bin_file: str) -> str:
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# IMPORTANT: app.py and face.jpeg must be in the same folder
BG_IMG = "face.jpeg"   # exact filename

bg_img = get_base64(BG_IMG)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(3,7,18,0.88);
        z-index: -1;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
# --------------------------------------------------

# Canva-style dark theme
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0b1020;
        color: #f5f5f7;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main > div:first-child h1 {
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #f9fafb;
    }

    .stMetric, .stAlert {
        border-radius: 14px;
        padding: 8px 12px;
        background: #111827;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #6366f1, #ec4899) !important;
    }

    section[data-testid="stSidebar"] {
        background: #020617;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {
        color: #e5e7eb;
    }

    button[data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 6px 18px !important;
        background: #020617;
        color: #9ca3af;
        border: 1px solid #1f2937;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg,#6366f1,#ec4899) !important;
        color: white !important;
        border: none !important;
    }

    .stProgress > div > div {
        border-radius: 999px;
        background: linear-gradient(90deg,#22c55e,#eab308);
    }

    img {
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Constants (match notebook)
# -----------------------------
RANDOMSEED = 42
random.seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

IMGW, IMGH = 92, 112
NSUBJECTS = 40
IMGSPERSUBJECT = 10

ALPHAS = [0.80, 0.85, 0.90, 0.95]
KVALUES = [1, 3, 5, 7]

# -----------------------------
# Utilities
# -----------------------------
def to_img(x_flat: np.ndarray) -> np.ndarray:
    img = x_flat.reshape(IMGH, IMGW)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def knn_predict(trainW: np.ndarray, ytrain: np.ndarray, testW: np.ndarray, k: int = 1) -> np.ndarray:
    preds = []
    for w in testW:
        d = np.linalg.norm(trainW - w, axis=1)
        idx = np.argsort(d)[:k]
        labels = ytrain[idx]
        values, counts = np.unique(labels, return_counts=True)
        preds.append(values[np.argmax(counts)])
    return np.array(preds, dtype=int)


def min_dist_to_class(trainW: np.ndarray, ytrain: np.ndarray, w: np.ndarray, cls: int) -> float:
    mask = (ytrain == cls)
    if np.sum(mask) == 0:
        return float("inf")
    d = np.linalg.norm(trainW[mask] - w.reshape(1, -1), axis=1)
    return float(np.min(d))


def min_dist_not_class(trainW: np.ndarray, ytrain: np.ndarray, w: np.ndarray, cls: int) -> float:
    mask = (ytrain != cls)
    d = np.linalg.norm(trainW[mask] - w.reshape(1, -1), axis=1)
    return float(np.min(d))


# -----------------------------
# ORL loader
# -----------------------------
@st.cache_data(show_spinner=False)
def load_orl_from_root(root_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    imgs, labels, paths = [], [], []

    for s in range(1, NSUBJECTS + 1):
        subj_dir = os.path.join(root_dir, f"s{s}")
        for i in range(1, IMGSPERSUBJECT + 1):
            p = os.path.join(subj_dir, f"{i}.pgm")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise RuntimeError(f"Failed reading: {p}")
            if im.shape != (IMGH, IMGW):
                im = cv2.resize(im, (IMGW, IMGH), interpolation=cv2.INTER_AREA)
            imgs.append(im.flatten().astype(np.float32))
            labels.append(s)
            paths.append(p)

    X = np.vstack(imgs)
    y = np.array(labels, dtype=int)
    return X, y, paths


def split_5_5_odd_even(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = [], []
    for s in range(NSUBJECTS):
        base = s * IMGSPERSUBJECT
        for j in range(IMGSPERSUBJECT):
            idx = base + j
            if (j + 1) % 2 == 1:
                train_idx.append(idx)
            else:
                test_idx.append(idx)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def split_7_3_first7(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = [], []
    for s in range(NSUBJECTS):
        base = s * IMGSPERSUBJECT
        for j in range(IMGSPERSUBJECT):
            idx = base + j
            if j < 7:
                train_idx.append(idx)
            else:
                test_idx.append(idx)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


# -----------------------------
# PCA (snapshot method)
# -----------------------------
@dataclass
class PCAModel:
    mean: np.ndarray
    eigvecs: np.ndarray
    eigvals: np.ndarray
    k: int
    explained_ratio: np.ndarray

    def project(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        return Xc @ self.eigvecs

    def reconstruct(self, W: np.ndarray) -> np.ndarray:
        return (W @ self.eigvecs.T) + self.mean


@st.cache_data(show_spinner=False)
def fit_pca_snapshot(Xtrain: np.ndarray, alpha: float) -> PCAModel:
    n, D = Xtrain.shape
    mu = np.mean(Xtrain, axis=0)
    A = (Xtrain - mu).astype(np.float64)
    S = A @ A.T
    evals, evecs_small = np.linalg.eigh(S)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs_small = evecs_small[:, idx]

    eps = 1e-10
    keep = evals > eps
    evals = evals[keep]
    evecs_small = evecs_small[:, keep]

    V = (A.T @ evecs_small)
    V = V / np.sqrt(evals.reshape(1, -1) + eps)

    total = np.sum(evals)
    ratio = (evals / total).astype(np.float64)
    cum = np.cumsum(ratio)
    k = int(np.searchsorted(cum, alpha) + 1)
    k = min(k, V.shape[1])
    V_k = V[:, :k].astype(np.float32)
    evals_k = evals[:k].astype(np.float32)
    ratio_k = ratio[:k].astype(np.float32)

    return PCAModel(
        mean=mu.astype(np.float32),
        eigvecs=V_k,
        eigvals=evals_k,
        k=k,
        explained_ratio=ratio_k,
    )


# -----------------------------
# LDA (Fisherfaces)
# -----------------------------
@dataclass
class LDAModel:
    mean: np.ndarray
    U: np.ndarray
    n_dims: int = 39

    def project(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        return Xc @ self.U


@st.cache_data(show_spinner=False)
def fit_fisherfaces(Xtrain: np.ndarray, ytrain: np.ndarray, pca_alpha: float = 0.95) -> LDAModel:
    pca = fit_pca_snapshot(Xtrain, pca_alpha)
    W = pca.project(Xtrain)
    classes = np.unique(ytrain)
    k = W.shape[1]

    mean_overall = np.mean(W, axis=0).reshape(-1, 1)
    SW = np.zeros((k, k), dtype=np.float64)
    SB = np.zeros((k, k), dtype=np.float64)

    for c in classes:
        Xc = W[ytrain == c]
        mc = np.mean(Xc, axis=0).reshape(-1, 1)
        Dc = (Xc - mc.reshape(1, -1)).astype(np.float64)
        SW += Dc.T @ Dc
        nc = Xc.shape[0]
        md = (mc - mean_overall)
        SB += nc * (md @ md.T)

    eps = 1e-4
    SWinv = np.linalg.pinv(SW + eps * np.eye(k))
    T = SWinv @ SB
    evals, evecs = np.linalg.eigh(T)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    n_dims = min(39, evecs.shape[1])
    Wlda = evecs[:, :n_dims].astype(np.float32)

    U_total = (pca.eigvecs @ Wlda).astype(np.float32)
    return LDAModel(mean=pca.mean, U=U_total, n_dims=n_dims)


# -----------------------------
# Evaluation helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def run_pca_eval(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, alpha: float, knn_k: int) -> Dict:
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    pca = fit_pca_snapshot(Xtr, alpha)
    Wtr = pca.project(Xtr)
    Wte = pca.project(Xte)
    ypred = knn_predict(Wtr, ytr, Wte, k=knn_k)
    acc = accuracy_score(yte, ypred)
    cm = confusion_matrix(yte, ypred, labels=np.arange(1, NSUBJECTS + 1))
    return {
        "acc": float(acc),
        "cm": cm,
        "pca": pca,
        "Wtr": Wtr,
        "Wte": Wte,
        "ytr": ytr,
        "yte": yte,
        "ypred": ypred,
    }


@st.cache_data(show_spinner=False)
def run_lda_eval(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, knn_k: int) -> Dict:
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    lda = fit_fisherfaces(Xtr, ytr, pca_alpha=0.95)
    Wtr = lda.project(Xtr)
    Wte = lda.project(Xte)
    ypred = knn_predict(Wtr, ytr, Wte, k=knn_k)
    acc = accuracy_score(yte, ypred)
    cm = confusion_matrix(yte, ypred, labels=np.arange(1, NSUBJECTS + 1))
    return {
        "acc": float(acc),
        "cm": cm,
        "lda": lda,
        "Wtr": Wtr,
        "Wte": Wte,
        "ytr": ytr,
        "yte": yte,
        "ypred": ypred,
    }


# -----------------------------
# Header
# -----------------------------
st.title("ORL Face Recognition Lab")
st.markdown(
    "<span style='font-size:0.95rem; color:#9ca3af;'>"
    "Interactive PCA / Fisherfaces playground for verification, similarity search, and model insight."
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Mode")
    user_mode = st.radio(
        "Who are you?",
        ["👤 General User", "🎓 Technical / Professor"],
        index=0,
    )

    st.markdown(
        f"<div style='margin-top:4px; margin-bottom:12px; padding:6px 10px; "
        "border-radius:999px; background:rgba(99,102,241,0.15); color:#a5b4fc; "
        "font-size:0.8rem;'>"
        f"{'✨ Friendly view for demos' if user_mode.startswith('👤') else '🧪 Full technical view'}"
        "</div>",
        unsafe_allow_html=True,
    )

    st.header("Dataset")
    root = st.text_input("ORL root folder (contains s1..s40)", value="orl_dataset")
    st.markdown(
        "<span style='font-size:0.75rem; color:#6b7280;'>"
        "AT&T / ORL faces (40 subjects × 10 images)"
        "</span>",
        unsafe_allow_html=True,
    )
    split_mode = st.selectbox("Train/Test split", ["5/5 (odd-even)", "7/3 (first 7 train)"])

    st.header("Methods")
    method = st.selectbox("Method", ["PCA", "LDA (Fisherfaces)"])
    knn_k = st.selectbox("KNN K", KVALUES, index=0)


try:
    X, y, paths = load_orl_from_root(root)
except Exception as e:
    st.error(f"Dataset load failed: {e}\n\nExpected structure: {root}/s1/1.pgm ... {root}/s40/10.pgm")
    st.stop()


if split_mode.startswith("5/5"):
    train_idx, test_idx = split_5_5_odd_even(y)
else:
    train_idx, test_idx = split_7_3_first7(y)


if method == "PCA":
    alpha_ui = st.sidebar.select_slider("PCA variance threshold (alpha)", options=ALPHAS, value=0.95)
    eval_out = run_pca_eval(X, y, train_idx, test_idx, alpha=alpha_ui, knn_k=knn_k)
    proj_train = eval_out["Wtr"]
    proj_test = eval_out["Wte"]
    y_train = eval_out["ytr"]
    y_test = eval_out["yte"]
    base_model = ("PCA", eval_out["pca"])
else:
    eval_out = run_lda_eval(X, y, train_idx, test_idx, knn_k=knn_k)
    proj_train = eval_out["Wtr"]
    proj_test = eval_out["Wte"]
    y_train = eval_out["ytr"]
    y_test = eval_out["yte"]
    base_model = ("LDA", eval_out["lda"])


# Tabs
if user_mode.startswith("👤"):
    tabs = st.tabs(["Identity Check", "Find Look-Alikes", "How AI Sees Faces", "System Reliability"])
else:
    tabs = st.tabs(["Verification Simulator", "Similarity Search", "Model Lab", "Metrics & Evaluation"])


# -----------------------------
# Tab 1: Verification / Identity Check
# -----------------------------
with tabs[0]:
    if user_mode.startswith("👤"):
        st.subheader("Identity Check (1:1)")
    else:
        st.subheader("Verification Simulator (1:1)")

    colA, colB = st.columns([1, 2], gap="large")

    with colA:
        claim_id = st.number_input("Claimed ID", min_value=1, max_value=NSUBJECTS, value=1, step=1)

        ref_d = []
        for w in proj_test[: min(50, proj_test.shape[0])]:
            ref_d.append(float(np.min(np.linalg.norm(proj_train - w.reshape(1, -1), axis=1))))
        ref_d = np.array(ref_d)
        dmin, dmax = float(np.min(ref_d)), float(np.max(ref_d))
        thr = st.slider(
            "Decision threshold",
            min_value=0.0,
            max_value=max(1.0, dmax * 2.0),
            value=float(dmax),
            step=float(max(0.1, dmax / 50.0)),
        )

        if user_mode.startswith("👤"):
            st.caption("Lower threshold = stricter system (fewer accepts).")
        else:
            st.caption("Lower threshold increases security (lower FAR) but increases rejection (higher FRR).")

        test_mask = (y_test == claim_id)
        if np.sum(test_mask) == 0:
            st.warning("No test samples for this ID in the current split.")
            st.stop()

        probe_local_idx = np.random.choice(np.where(test_mask)[0])
        probe_global_idx = test_idx[probe_local_idx]
        probe_img = to_img(X[probe_global_idx])

        w_probe = proj_test[probe_local_idx]
        score = min_dist_to_class(proj_train, y_train, w_probe, claim_id)

        # Confidence from distance
        confidence = float(np.exp(-score / (thr + 1e-9)))
        confidence = max(0.0, min(confidence, 1.0))
        decision = score <= thr

        # FAR / FRR estimate
        genuine_scores, impostor_scores = [], []
        for i in range(len(proj_test)):
            w_i = proj_test[i]
            true_id = y_test[i]
            genuine_scores.append(min_dist_to_class(proj_train, y_train, w_i, true_id))
            impostor_scores.append(min_dist_not_class(proj_train, y_train, w_i, true_id))
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        far = float(np.mean(impostor_scores <= thr))
        frr = float(np.mean(genuine_scores > thr))

        if user_mode.startswith("👤"):
            st.markdown(
                "<div style='background:#020617; border-radius:16px; padding:16px 18px; "
                "border:1px solid #1f2937;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:1.1rem; font-weight:600; margin-bottom:4px;'>"
                f"Identity Match: {'<span style=\"color:#22c55e;\">Yes</span>' if decision else '<span style=\"color:#f97373;\">No</span>'}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.caption(f"Confidence: {confidence*100:.1f}%")
            st.caption(
                "The system compares facial patterns to decide whether the probe and enrolled face belong to the same person."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.metric("Distance score", f"{score:.4f}")
            st.metric("Decision", "✅ Accept" if decision else "❌ Reject")

        col_far, col_frr = st.columns(2)
        col_far.metric("False Accept Rate (FAR)", f"{far*100:.2f}%")
        col_frr.metric("False Reject Rate (FRR)", f"{frr*100:.2f}%")

        genuine_score = min_dist_to_class(proj_train, y_train, w_probe, claim_id)
        impostor_score = min_dist_not_class(proj_train, y_train, w_probe, claim_id)
        if not user_mode.startswith("👤"):
            st.write(f"Genuine score (claimed ID): `{genuine_score:.4f}`")
            st.write(f"Impostor score (nearest other ID): `{impostor_score:.4f}`")

        if st.button("Why this decision?"):
            st.info(
                "The probe face is projected into the same feature space as the enrolled faces. "
                "If its distance to the claimed ID is below the threshold and closer than other IDs, "
                "the system accepts; otherwise it rejects."
            )
        else:
            st.markdown(
                "<span style='font-size:0.8rem; color:#9ca3af;'>"
                "Curious? Click to see how the decision is made in plain language."
                "</span>",
                unsafe_allow_html=True,
            )

    with colB:
        st.write("Probe image (from test set)")
        st.image(probe_img, clamp=True, width=220)

        claim_train_mask = (y_train == claim_id)
        claim_train_global = train_idx[np.where(claim_train_mask)[0]]
        claim_train_W = proj_train[claim_train_mask]
        d = np.linalg.norm(claim_train_W - w_probe.reshape(1, -1), axis=1)
        best_idx = int(np.argmin(d))
        best_global = int(claim_train_global[best_idx])
        best_img = to_img(X[best_global])

        st.write("Best enrolled match (within claimed ID)")
        st.image(best_img, clamp=True, width=220)

        diff = np.abs(probe_img.astype(np.float32) - best_img.astype(np.float32))
        diff = (diff / (diff.max() + 1e-9) * 255).astype(np.uint8)
        st.write("Difference map (how the system sees the change)")
        st.image(diff, clamp=True, width=220)


# -----------------------------
# Tab 2: Similarity Search / Look-Alikes
# -----------------------------
with tabs[1]:
    st.subheader("Similarity Search (1:N)" if not user_mode.startswith("👤") else "Find Look-Alikes")
    k_top = st.slider("Top-K results", min_value=1, max_value=10, value=5, step=1)

    q_pos = st.slider("Query index (within test set)", min_value=0, max_value=len(test_idx) - 1, value=0, step=1)
    q_global = int(test_idx[q_pos])
    q_img = to_img(X[q_global])
    q_label = int(y[q_global])
    w_q = proj_test[q_pos]

    st.write(f"Query true ID: **{q_label}**")
    st.image(q_img, clamp=True, width=220)

    d = np.linalg.norm(proj_train - w_q.reshape(1, -1), axis=1)
    topk = np.argsort(d)[:k_top]

    if user_mode.startswith("👤"):
        st.caption("These are the most visually similar faces the system knows.")
    else:
        st.caption("Similarity search retrieves the closest faces in feature space, not necessarily the correct identity.")

    cols = st.columns(k_top)
    for i, idx in enumerate(topk):
        pred_id = int(y_train[idx])
        gidx = int(train_idx[idx])
        img = to_img(X[gidx])
        ok = (pred_id == q_label)
        title = f"Rank {i+1} | ID {pred_id} | d={d[idx]:.3f}"
        cols[i].image(img, clamp=True, use_container_width=True)
        if not user_mode.startswith("👤"):
            cols[i].markdown(
                f"<div style='color:{'green' if ok else 'red'}; font-weight:600'>{title}</div>",
                unsafe_allow_html=True,
            )


# -----------------------------
# Tab 3: Model Lab / How AI Sees
# -----------------------------
with tabs[2]:
    st.subheader("Model Lab (Mean face, Eigenfaces, Reconstruction)" if not user_mode.startswith("👤") else "How AI Sees a Face")

    if base_model[0] != "PCA":
        st.info("Model Lab uses PCA visualization. Switch Method to PCA in the sidebar for eigenfaces + reconstruction.")
    else:
        pca: PCAModel = base_model[1]
        mu_img = to_img(pca.mean)

        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.write("Mean Face")
            st.image(mu_img, clamp=True, width=220)

        with c2:
            st.write("Top Eigenfaces")
            n_show = min(10, pca.k)
            cols = st.columns(5)
            for i in range(n_show):
                ef = pca.eigvecs[:, i].copy()
                ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-9) * 255.0
                cols[i % 5].image(to_img(ef), clamp=True, use_container_width=True, caption=f"e{i+1}")

        st.divider()
        st.write("Reconstruction")
        k_rec = st.slider("Number of components (k)", min_value=1, max_value=pca.k, value=min(25, pca.k), step=1)
        img_pos = st.slider("Image index (global 0..399)", min_value=0, max_value=X.shape[0] - 1, value=0, step=1)

        x0 = X[img_pos]
        w0 = (x0 - pca.mean) @ pca.eigvecs[:, :k_rec]
        xhat = (w0 @ pca.eigvecs[:, :k_rec].T) + pca.mean

        err = mse(x0, xhat)
        cc1, cc2, cc3 = st.columns(3)
        cc1.image(to_img(x0), clamp=True, use_container_width=True, caption="Original")
        cc2.image(to_img(xhat), clamp=True, use_container_width=True, caption="Reconstructed")
        cc3.metric("Reconstruction MSE", f"{err:.2f}")

        compression_ratio = (pca.k * 1.0) / X.shape[1]
        if user_mode.startswith("👤"):
            st.caption(f"The system keeps about {compression_ratio*100:.1f}% of the raw information as numbers.")
        else:
            st.metric("Compression ratio", f"{compression_ratio*100:.2f}%")

# -----------------------------
# Tab 4: Metrics / System Reliability
# -----------------------------
with tabs[3]:
    st.subheader("Metrics & Evaluation" if not user_mode.startswith("👤") else "System Reliability")
    st.write(f"Current setting: Method={method}, Split={split_mode}, K={knn_k}")

    st.markdown("### Overall Accuracy")
    st.metric("Accuracy", f"{eval_out['acc']:.4f}")
    if user_mode.startswith("👤"):
        st.caption(
            "Roughly, out of 100 test faces the system correctly recognizes about "
            f"{eval_out['acc']*100:.1f}."
        )
    else:
        st.markdown("### Accuracy curves")
        left, right = st.columns(2, gap="large")

        with left:
            st.write("Accuracy vs alpha (PCA)")
            pca_accs = []
            for a in ALPHAS:
                out = run_pca_eval(X, y, train_idx, test_idx, alpha=a, knn_k=knn_k)
                pca_accs.append(out["acc"])
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.plot(ALPHAS, pca_accs, marker="o")
            ax.set_xlabel("alpha (variance threshold)")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with right:
            st.write("Accuracy vs K (current method)")
            accs_k = []
            for kk in KVALUES:
                if method == "PCA":
                    out = run_pca_eval(
                        X, y, train_idx, test_idx,
                        alpha=(alpha_ui if "alpha_ui" in globals() else 0.95),
                        knn_k=kk,
                    )
                else:
                    out = run_lda_eval(X, y, train_idx, test_idx, knn_k=kk)
                accs_k.append(out["acc"])
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.plot(KVALUES, accs_k, marker="o")
            ax.set_xlabel("K (KNN)")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.markdown("### Confusion matrix")
        st.caption("Tip: LDA typically improves separability vs PCA because it optimizes class separation in the projected space.")
        cm = eval_out["cm"]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, cmap="Blues" if method == "PCA" else "Greens", cbar=True, ax=ax)
        ax.set_xlabel("Predicted ID")
        ax.set_ylabel("True ID")
        st.pyplot(fig)
