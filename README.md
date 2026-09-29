# 💵 Damaged Indian Banknote Detection & Severity Classification

## 📌 Project Overview

This computer-vision project is designed to recognize Indian banknotes and assess visible damage using deep learning.

The proposed end-to-end system combines:

- **Denomination recognition**
- **Spoilt / damaged banknote analysis**
- **Fine-grained damage classification**
- **Model confidence**
- **A Streamlit demo**
- **Informational RBI note-exchange guidance**

The project is structured so that model predictions and official exchange decisions remain clearly separate.

## 🎯 Project Objectives

- Recognize Indian currency denominations from images
- Build a deep-learning image-classification pipeline
- Explore damaged / spoilt Indian banknotes
- Create a fine-grained damage-severity dataset
- Classify note condition after verified labeling
- Deploy the trained model through Streamlit
- Provide non-binding guidance based on public RBI note-exchange information

## 📂 Dataset Sources

### 1. Dataset of Spoilt Banknotes of India — Recommended damaged-note source

The Mendeley dataset contains **5,125 images**:
- 2,584 old banknotes
- 2,541 new banknotes

It contains damaged ₹10, ₹20, ₹50 and ₹100 notes and includes examples described as soiled, mutated, holed, torn and crumpled, captured under varied backgrounds and lighting.

**Important:** its released eight classes are based on denomination and old/new note type. It does **not** directly label each image as `Torn`, `Crumpled`, `Missing Corner`, etc.

Source:
`https://data.mendeley.com/datasets/jh6979fg2t/4`

### 2. Indian Currency Dataset — Denomination Recognition

A separate Mendeley dataset contains **1,786 images** across seven Indian denominations:
₹10, ₹20, ₹50, ₹100, ₹200, ₹500 and ₹2000.

Source:
`https://data.mendeley.com/datasets/8ckhkssyn3/1`

### 3. Kaggle ViT Reference

Reference notebook supplied for project inspiration:

`https://www.kaggle.com/code/bryamblasrimac/classification-indiancurrency-vit-accuracy-97-80`

The Kaggle notebook is a reference only. Its reported accuracy must **not** be presented as the result of this repository unless independently reproduced.

## ⚠️ Fine-Grained Damage Labels

The proposed damage classes are:

| Class | Meaning |
|---|---|
| Good | Normal / no significant visible damage |
| Slightly Worn | Minor visible wear |
| Torn | Visible tear |
| Dirty / Soiled | Heavy dirt / soiling |
| Crumpled | Strong folding / crumpling |
| Missing Corner | A visible piece / corner is missing |
| Heavily Damaged | Severe visible deterioration |

These seven labels are **not provided directly** by the 5,125-image Mendeley dataset.

A fine-grained classifier should be trained only after images have been reliably annotated into these classes.

An annotation template is included at:

`labels/damage_annotation_template.csv`

## 🧠 Proposed Architecture

```text
Input Banknote Image
        ↓
Image Validation / Preprocessing
        ↓
Denomination Recognition Model
(EfficientNet / ViT / ResNet)
        ↓
Damage Severity Model
(MobileNetV2 / EfficientNet / CNN)
        ↓
Predicted Damage Class + Confidence
        ↓
Informational Recommendation
        ↓
Streamlit Application
```

## 📓 Repository Notebooks

### `01_Dataset_Audit.ipynb`

- Audits image folders
- Counts images per class
- Checks image dimensions
- Helps validate dataset organization before training

### `02_Denomination_Classification.ipynb`

- Transfer learning with EfficientNetB0
- Image augmentation
- Train / validation split
- Model checkpointing
- Accuracy visualization

### `03_Damage_Severity_Classification.ipynb`

- Fine-grained damage-classification template
- MobileNetV2 transfer learning
- Seven proposed damage classes
- Must be trained only after verified labeling

## 🛠 Technologies

- Python
- TensorFlow / Keras
- OpenCV
- Pandas
- NumPy
- Matplotlib
- CNN
- Transfer Learning
- EfficientNetB0
- MobileNetV2
- Streamlit
- Jupyter Notebook

## 📁 Repository Structure

```text
Indian-Banknote-Damage-Detection/
│
├── README.md
├── DATA_SOURCES.md
├── RBI_GUIDANCE.md
├── requirements.txt
├── .gitignore
│
├── 01_Dataset_Audit.ipynb
├── 02_Denomination_Classification.ipynb
├── 03_Damage_Severity_Classification.ipynb
├── app.py
│
├── data/
│   └── README.md
│
├── labels/
│   └── damage_annotation_template.csv
│
├── models/
│   └── README.md
│
└── screenshots/
    └── README.md
```

## 🏦 RBI Guidance Layer

The app does **not** determine whether a note is legally valid or the refund amount.

It only displays general informational guidance. Official assessment belongs to authorised banks / RBI procedures.

Examples of safe app messages:

| Model Output | Informational Guidance |
|---|---|
| Good | No major damage detected by the model |
| Slightly Worn | Minor wear detected; consult current bank/RBI guidance if needed |
| Torn | Bank branch assessment recommended |
| Dirty / Soiled | Soiled-note exchange guidance may apply |
| Missing Corner | Mutilated-note assessment may be required |
| Heavily Damaged | Seek official assessment; app cannot determine refund value |

## 🚀 Running the Streamlit App

Install packages:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

The app will show a warning until a trained model exists at:

```text
models/damage_severity_model.keras
```

## 📊 Evaluation

When training is completed, report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance

Do not copy accuracy values from another notebook or paper.

## 🔮 Future Scope

- Mobile application
- Live camera inference
- Better severity annotation
- Bank-counter assistance
- Currency sorting support
- Accessibility tools for visually impaired users
- Model explainability
- Larger dataset covering ₹200 and ₹500 damaged notes

## 📌 Current Project Status

- ✅ GitHub project structure prepared
- ✅ Dataset sources identified
- ✅ Dataset-audit notebook prepared
- ✅ Denomination-training notebook prepared
- ✅ Fine-grained severity-training template prepared
- ✅ Streamlit demo template prepared
- ⏳ Datasets need to be downloaded
- ⏳ Seven-class severity labels need verified annotation
- ⏳ Models need to be trained
- ⏳ Results must be generated from actual training

## 👤 Project Owner

**Ganesh T**

---

⭐ This project demonstrates computer vision, transfer learning, image classification, responsible model reporting, and practical application design.
