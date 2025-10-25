# UrbanSound8K Audio Classification: Baseline vs Neural Network

This notebook implements and evaluates audio classification models on the UrbanSound8K dataset using proper 10-fold cross-validation methodology.

## 🎯 Overview

**Objective:** Compare Naive Bayes baseline vs Artificial Neural Network for urban sound classification

**Models:**
- **Baseline:** Gaussian Naive Bayes
- **Neural Network:** Multi-layer Perceptron with ReLU activations and dropout

**Evaluation:** 10-fold cross-validation using predefined dataset splits (following Salamon et al. 2014 recommendations)

**Results:**
- Baseline Accuracy: ~45%
- Neural Network Accuracy: ~60%
- Improvement: +15 percentage points

---

## 🔧 Requirements

### Platform
- **Kaggle Notebook** (recommended)
- GPU: T4 x2 (required for neural network training)
- RAM: 16GB+ recommended

### Dataset
- **UrbanSound8K** (must be added as Kaggle dataset)
- Size: ~6GB
- 8,732 audio files across 10 classes
- Pre-divided into 10 folds

### Python Libraries
All libraries are pre-installed in Kaggle environment:
```
- numpy
- pandas
- librosa
- scikit-learn
- torch (PyTorch)
- matplotlib
- seaborn
- tqdm
```

---

## 🚀 Setup Instructions

### Step 1: Import Notebook to Kaggle

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click **"Create"** → **"New Notebook"**
3. Click **"File"** → **"Import Notebook"**
4. Upload this `.ipynb` file

### Step 2: Add UrbanSound8K Dataset

**Method A: Add from Kaggle Datasets (Recommended)**
1. In your notebook, click **"+ Add Data"** (right sidebar)
2. Search for **"UrbanSound8K"**
3. Select the official dataset
4. Click **"Add"**

The dataset will be mounted at: `/kaggle/input/urbansound8k/`

**Method B: Add Your Own Dataset**
1. Upload UrbanSound8K to Kaggle Datasets (if not already available)
2. Follow Method A steps above

### Step 3: Configure GPU

1. Click **"Session Options"** (right sidebar, bottom)
2. Under **"Accelerator"**, select **"GPU T4 x2"**
3. Click **"Save"**
4. Session will restart with GPU enabled

**⚠️ Important:** GPU is REQUIRED for neural network training. CPU-only will take 10+ hours.

### Step 4: Verify Dataset Path

In the first code cell, verify the path matches your dataset location:
```python
CONFIG = {
    'audio_dir': '/kaggle/input/urbansound8k/UrbanSound8K/audio',
    'metadata_path': '/kaggle/input/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv',
    ...
}
```

If your dataset is in a different location, update these paths accordingly.

---

## ▶️ How to Run

### Quick Start (Automated)
```
1. Ensure GPU T4 x2 is enabled
2. Click "Run All" (or Ctrl+A → Shift+Enter)
3. Wait for completion
4. Results will be saved to /kaggle/working/UrbanSound_Project/
```

### Manual Execution (Cell-by-Cell)

**Phase 1: Setup**
1. Import libraries, define CONFIG, verify dataset
2. Create project directories
3. Load metadata

**Phase 2: Feature Extraction**
1. Define feature extraction function
3. Define fold splitting function

**Phase 3: Baseline Model** 
1. Run Naive Bayes 10-fold cross-validation
2. Generate baseline confusion matrix

**Phase 4: Neural Network**
1. Define neural network architecture
2. Define training function
3. Hyperparameter experiments (architecture, learning rate, dropout)
4. Run 10-fold CV with best configuration
5. Generate NN confusion matrix
---

### Generated Files

All outputs saved to `/kaggle/working/UrbanSound_Project/`:

**Features:**
- `features_n120.npy` - Extracted audio features
- `labels.npy` - Corresponding labels

**Models:**
- `nn_model_fold1.pth` to `nn_model_fold10.pth` - Saved models for each test fold

**Visualizations:**
- `baseline_confusion_matrix_best.png` - Baseline confusion matrix
- `nn_confusion_matrix_best.png` - Neural network confusion matrix

**Results:**
- `results/` - JSON files with detailed metrics

---

## 📁 Project Structure
```
/kaggle/working/UrbanSound_Project/
│
├── features/                    # Extracted audio features
│   ├── features_n120.npy
│   └── labels.npy
│
├── models/                      # Saved neural network models
│   ├── nn_model_fold1.pth
│   ├── nn_model_fold2.pth
│   └── ... (10 total)
│
├── figures/                     # Generated visualizations
│   ├── baseline_confusion_matrix_best.png
│   ├── nn_confusion_matrix_best.png
│   └── baseline_vs_nn_comparison.png
│
└── results/                     # Experiment results (JSON)
    ├── baseline_results.json
    └── nn_results.json
```
