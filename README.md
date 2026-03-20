# 🧬 Track B — Antibiotic Resistance Prediction
### CodeCure AI Hackathon 2026

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML](https://img.shields.io/badge/Model-Random%20Forest-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Problem Statement

Antimicrobial resistance (AMR) is one of the most pressing global health
challenges. This project builds a machine learning model that predicts
**Gentamicin resistance** in bacterial isolates collected from environmental
and clinical sources in Nigeria, using co-resistance patterns from other
antibiotics as predictive features.

---

## 📊 Dataset

**Source:** [Mendeley AMR Dataset](https://data.mendeley.com/datasets/ccmrx8n7mk/1)

| Property | Detail |
|----------|--------|
| Total isolates | 274 unique bacterial samples |
| Collection sites | 12 sites across IFE, OSU, IWO, EDE |
| Sources | Butcher Table, Concrete Slab, Surrounding Soil |
| Antibiotics | Imipenem, Ceftazidime, Gentamicin, Augmentin, Ciprofloxacin |
| Values | Zone of inhibition (mm) → converted to R/S/I using CLSI breakpoints |

---

## 🔬 Methodology
```
Raw Dataset (mm values)
        ↓
Convert mm → R/S/I (CLSI breakpoints)
        ↓
Feature Engineering (City + Source extraction)
        ↓
One-hot Encoding → 19 features
        ↓
Train/Test Split (80/20, stratified)
        ↓
SMOTE Balancing (357 training samples)
        ↓
Train 4 Models → Random Forest wins (CV F1: 0.735)
        ↓
SHAP Explainability → Biological Insights
        ↓
Streamlit Prediction App
```

---

## 📈 Model Performance

| Model | CV F1 | Test F1 | Accuracy |
|-------|-------|---------|----------|
| **Random Forest** | **0.735** | **0.518** | **0.600** |
| LightGBM | 0.716 | 0.532 | 0.618 |
| XGBoost | 0.691 | 0.533 | 0.600 |
| Logistic Regression | 0.550 | 0.547 | 0.600 |

> CV F1 is the primary metric — more reliable than Test F1 for small datasets (274 isolates)

---

## 🔍 Key Biological Findings (SHAP Analysis)

| Rank | Feature | SHAP Value | Biological Meaning |
|------|---------|-----------|-------------------|
| 1 | IMIPENEM_Susceptible | 0.0644 | Strongest predictor — different resistance mechanism |
| 2 | CIPROFLOXACIN_Resistant | 0.0619 | Plasmid-mediated co-resistance (PMQR genes) |
| 3 | City_IWO | 0.0423 | Environmental selective pressure |
| 4 | City_OSU | 0.0381 | Site-specific resistance patterns |
| 5 | Source_Surrounding_Soil | 0.0355 | Agricultural antibiotic residue accumulation |

**Key insight:** Two independent methods — Pearson phi correlation (EDA) and
SHAP (ML) — both identified **CIPROFLOXACIN resistance** as the strongest
predictor of Gentamicin resistance (r=0.250), consistent with
plasmid-mediated co-resistance mechanisms.

---

## 📁 Project Structure
```
track-b-amr-prediction/
│
├── data/
│   └── Dataset.xlsx              ← Mendeley AMR dataset
│
├── notebooks/
│   ├── 01_eda.ipynb              ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   ← Cleaning, encoding, SMOTE
│   ├── 03_modelling.ipynb        ← Model training & evaluation
│   └── 04_shap.ipynb             ← SHAP feature importance
│
├── figures/
│   ├── fig1_class_distribution.png
│   ├── fig2_resistance_heatmap.png
│   ├── fig3_mdr_analysis.png
│   ├── fig4_coresistance_matrix.png
│   ├── fig5_smote_balance.png
│   ├── fig6_model_comparison.png
│   ├── fig7_confusion_matrix.png
│   ├── fig8_shap_importance.png
│   └── fig9_shap_dotplot.png
│
├── src/
│   └── app.py                    ← Streamlit prediction app
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/raghuvarshan452-commits/track-b-amr-prediction.git
cd track-b-amr-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn shap streamlit openpyxl
```

### 3. Run the notebooks in order
```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_modelling.ipynb
notebooks/04_shap.ipynb
```

### 4. Launch the Streamlit app
```bash
streamlit run src/app.py
```

---

## 📊 EDA Visualisations

### Co-resistance Patterns
Top co-resistance pairs identified from real data:
- GENTAMICIN ↔ CIPROFLOXACIN: r = 0.250 ← strongest
- GENTAMICIN ↔ IMIPENEM: r = 0.243
- AUGMENTIN ↔ CIPROFLOXACIN: r = 0.236

### Multi-Drug Resistance
- ~25% of isolates resist 3+ antibiotics simultaneously
- Butcher Table sites show highest MDR prevalence

---

## 🏥 Clinical Implications

1. **Ciprofloxacin-resistant isolates** should be screened for Gentamicin
   resistance before aminoglycoside treatment is prescribed
2. **IWO and OSU collection sites** show elevated resistance rates —
   suggesting targeted antibiotic stewardship interventions in these areas
3. **Butcher Table environments** are high-risk AMR transmission points —
   consistent with known links between animal slaughter and resistance spread

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas + NumPy | Data manipulation |
| Scikit-learn | ML pipeline |
| XGBoost + LightGBM | Gradient boosting models |
| imbalanced-learn | SMOTE balancing |
| SHAP | Model explainability |
| Matplotlib + Seaborn | Visualisations |
| Streamlit | Prediction web app |

---

## 👤 Author

**NEURAL NINJAS**
CodeCure AI Hackathon 2026 — Track B