# 🛡️ Cyberbullying Detection System from Scratch (NumPy Only)

An automated, high-performance machine learning pipeline designed to classify social media text into binary categories: **Clean (Class 0)** and **Cyberbullying (Class 1)**. 

The primary architectural objective of this project is to demonstrate the internal mathematical mechanics, tensor transformations, and gradient optimization loops of classification models **entirely from scratch using raw NumPy matrix commands**, bypassing high-level deep learning and statistical frameworks like Scikit-Learn, PyTorch, or TensorFlow.

---

## 🚀 Key Features

* **Zero Framework Dependencies:** Custom matrix implementations for all tokenizers, vectorizers, and training loops.
* **Imbalance Compensation:** An analytical inverse class frequency weight mask injected directly into the gradient updates to handle severe data asymmetry (Class 1 outnumbering Class 0).
* **Memory-Optimized Vectorization:** All vectorized feature spaces are explicitly cast into 32-bit floating-point structures (`np.float32`), reducing memory footprint by 50%.
* **Low-Level Tensor Striding:** The 1D CNN uses native memory stride tricks (`as_strided`) and Einstein Notation (`np.einsum`) to perform fast parallel matrix dot products without nested python loops.

---

## 📊 Implemented Architectures & Performance

The pipeline evaluates and benchmarks three fundamentally different geometric and probabilistic decision boundaries over an independent test partition of **9,539 unseen entries**:

| Model Architecture | Global Accuracy | Precision | Recall | F1-Score | Optimal Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Support Vector Machine (SVM)** | 85.41% | 86.91% | **97.11%** | **91.73%** | **Risk-Averse Moderation:** Captures obscure toxic signals with absolute minimum missed threats. |
| **1D Convolutional Network (CNN)** | 81.43% | **93.16%** | 83.87% | 88.27% | **Automated Precision Flagging:** Evaluates phrase context (Trigrams) to completely suppress false alarms. |
| **Logistic Regression** | **84.48%** | 90.77% | 90.58% | 90.68% | **Statistical Equilibrium:** Symmetric error profile with highly balanced thresholds. |

---

## 🛠️ Data Preprocessing & Feature Pipeline

### 1. Text Cleansing & Normalization
* **Noise Stripping:** Regular expressions parse out hypermedia links (`http/https`), user mentions (`@user`), digits, and punctuation.
* **Case Unification:** Global mapping to lowercase to stabilize vocabulary keys.
* **Character De-duplication:** Restricts repetitive letters (e.g., "loooove" $\rightarrow$ "love") to a max of two consecutive characters using a custom lookup routine.
* **Stopwords Elimination:** Filters out zero-semantic functional tokens.

### 2. Feature Vectorization Arrays
* **Manual TF-IDF Matrix:** Structured coordinates calculating Term Frequency ($TF$) and Inverse Document Frequency ($IDF$) with custom smoothing bounds for linear pipelines.
* **Sequential Token Padding:** Encodes raw text tokens into sequential integer index tensors up to a strict boundary constraint of `max_len = 200` with automated sequence padding (`<PAD>` allocated securely at index `0`).

---

## 📐 Optimization & Loss Mechanics

### Logistic Regression
Passes linear scores through a clipped vector sigmoid layer to prevent floating-point underflow/overflow (`NaN` exceptions) under numerical bounds of $\pm500$:
$$\sigma(z)=\frac{1}{1+e^{-\text{clip}(z,-500,500)}}$$
Optimized via Mini-Batch SGD utilizing a Weighted Binary Cross-Entropy loss.

### Soft-Margin Linear SVM
Operates sub-gradient optimization routines across a Weighted Hinge Loss objective function based on structural margin boundary violations:
* **Case 1 (Outside Margin):** $dw = \lambda w$
* **Case 2 (Margin-Violating):** $dw = \lambda w - (C_{wi} \times y_i \times X_i)^T$

### 1D Convolutional Neural Network
* **Feature Maps:** Employs spatial phrase filters (Trigrams) across localized textual windows.
* **Backpropagation:** Gradients are routed backwards through custom element-wise ReLU derivatives and Global Max-Pooling layers down to the continuous word embeddings via a native Adam Optimizer loop.
* **Embedding Masking:** A dedicated spatial constraint forces the `<PAD>` token's embedding gradient to zero after every optimization step to safeguard structural integrity.

---

## 📂 Project Structure

```text
├── data/
│   └── cyberbullying_tweets.csv      # Unified text corpus
├── notebooks/
│   └── cyberbullying_detection.ipynb # Complete execution pipeline and visuals
├── README.md
└── requirements.txt                  # Strictly NumPy and Matplotlib
