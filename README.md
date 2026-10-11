# Automated ECG Arrhythmia Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras%20Tuner-Optimization-red.svg)](https://keras.io/keras_tuner/)
[![Dataset](https://img.shields.io/badge/Dataset-MIT--BIH%20Arrhythmia-success.svg)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework for automated, real-time Electrocardiogram (ECG) heartbeat classification. This project investigates and benchmarks **1D Convolutional Neural Networks (1D-CNN)** against **Stacked Long Short-Term Memory (LSTM)** networks on the clinical **MIT-BIH Arrhythmia Database**, addressing extreme class imbalance via dynamic **Inverse-Frequency Loss Weighting**.

---

## 📌 Executive Summary

Manual review of continuous 24-hour Holter ECG recordings (>100,000 heartbeats/patient) creates diagnostic delays and human observer fatigue. This project establishes an algorithmic diagnostic decision-support system:
- **Severe Imbalance Resolution:** Implemented inverse-frequency cross-entropy penalty weights ($W_3=27.32$ for Fusion beats vs. $W_0=0.24$ for Normal beats), overcoming extreme 1:111 class disparity.
- **Architecture Champion:** Proved that **1D spatial template matching (1D-CNN)** is superior to temporal unrolling (LSTM) for pre-segmented single beats, achieving **95.34% Test Accuracy** and **95.85% Weighted F1-Score**.
- **Real-Time Clinical Viability:** 1D-CNN achieved **7.0× faster inference latency** (26 ms/step vs. 181 ms/step on CPU), rendering it ideal for edge devices and wearable cardiac monitors.

---

## 📊 Benchmark Results

| Model Architecture | Test Accuracy | Weighted F1 | Macro F1 | Ventricular Ectopic Recall (Class V) | Fusion Beat Recall (Class F) | CPU Latency (ms/step) |
|---|---|---|---|---|---|---|
| **Optimized 1D-CNN** | **95.34%** | **95.85%** | **80.26%** | **93.65%** | **86.42%** | **26 ms** |
| **Stacked LSTM** | 45.96% | 55.15% | 31.81% | 22.44% | 18.06% | 181 ms |

---

## 🗂️ Repository Structure

```text
├── 1_Exploratory_Data_Analysis_and_Preparation.ipynb    # Signal EDA, class distribution & Welch PSD
├── 2_Model_Building_and_Hyperparameter_Tuning.ipynb    # CNN & LSTM definition with Keras Tuner
├── 3_Model_Evaluation_ROC_PR_and_Latency_Quadrant.ipynb# ROC, PR curves, Confusion Matrix & Latency Profiling
├── 4_CNN_Feature_Maps_and_Kernel_Weights.ipynb         # Convolutional layer filter & activation visualization
├── ODL_ECG_Classification_Notebook.ipynb               # Master end-to-end execution pipeline
├── requirements.txt                                    # Python dependencies
├── .gitignore                                          # Excludes large CSV datasets & local checkpoints
└── README.md                                           # Project documentation
```

---

## 📥 Dataset Setup

This project uses the **ECG Heartbeat Categorization Dataset** from Kaggle (derived from PhysioNet MIT-BIH & PTB databases).

1. Download the dataset from [Kaggle Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).
2. Create a `data/` directory in the repository root:
```bash
mkdir data
```
3. Place `mitbih_train.csv` and `mitbih_test.csv` inside the `data/` folder:
```text
data/
├── mitbih_train.csv
└── mitbih_test.csv
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd <YOUR-REPO-NAME>
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter notebook
```
Open and run `ODL_ECG_Classification_Notebook.ipynb` for the complete end-to-end execution, or run the modular notebooks (`1_...` to `4_...`) sequentially.

---

## 🔬 Key Methodologies

- **Signal Normalization:** Zero-mean, unit-variance standardized 187-sample time series.
- **Spectral Density Analysis:** Welch Power Spectral Density (PSD) analysis identifying frequency bands between 0.5 Hz and 40 Hz distinguishing ventricular ectopic morphology.
- **Loss Formulation:** Weighted Categorical Cross-Entropy:
  $$L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^4 w_c \cdot y_{i,c} \log(\hat{y}_{i,c})$$
- **Hyperparameter Optimization:** Keras Tuner Bayesian Search optimizing kernel sizes ($k \in \{3, 5, 7\}$), filter depths ($32 \to 64 \to 128$), and dropout regularization ($0.2 - 0.5$).

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
