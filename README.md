# 🏠 House Price Prediction — End-to-End Machine Learning Project

A complete machine learning pipeline for predicting house sale prices using the King County, WA housing dataset. The project demonstrates the full ML lifecycle — from raw data exploration to model comparison — with a modular, production-style codebase.

---

## 📌 Project Highlights

| Area | Details |
|---|---|
| **Goal** | Predict house sale prices and systematically improve model accuracy |
| **Dataset** | King County House Sales (~21,600 records, 21 features) |
| **Best Model** | Random Forest — **R² = 0.88** on test data with clean + scaled pipeline |
| **Key Skills** | Regression, Feature Engineering, Outlier Treatment, Scaling, Model Comparison |

---

## 🧠 Problem Statement

Given a set of house features (square footage, bedrooms, bathrooms, location, condition, etc.), build regression models to predict the sale price. Iteratively improve performance through data cleaning, outlier removal, and feature scaling.

---

## 🔬 Methodology & Pipeline

The project follows an iterative, experiment-driven approach with four progressive stages:

### 1. Baseline Model (v0.0)
- Dropped non-predictive columns (`id`, `date`)
- Trained models on raw features with an 80/20 train-test split
- Established a performance baseline across 7 regression algorithms

### 2. Baseline + Scaling (v0.1)
- Applied `StandardScaler` to normalize feature ranges
- Re-trained all models to measure the impact of feature scaling
- Observed significant improvement for distance-based models (KNN: +27% R²)

### 3. Data Cleaning (v0.2)
- Investigated and handled invalid values (e.g., 0 bedrooms, 0 bathrooms)
- Removed records with data quality issues based on domain logic
- Improved linear model R² from **0.70 → 0.79** through cleaning alone

### 4. Outlier Treatment + Cleaned & Scaled (v0.3)
- Applied **IQR-based outlier removal** on numeric features
- Combined cleaning + scaling for the best-performing pipeline
- Achieved the strongest results: **Random Forest R² = 0.88**

---

## 📊 Model Comparison Results

Seven regression models were benchmarked at each pipeline stage:

| Model | Baseline R² (Test) | Clean + Scaled R² (Test) | Overfit Gap |
|---|---|---|---|
| Linear Regression | 0.7012 | 0.7943 | 1.71% |
| Ridge Regression | 0.7011 | 0.7943 | 1.71% |
| Lasso Regression | 0.7012 | 0.7943 | 1.71% |
| KNN Regressor | 0.5037 | 0.8115 | 7.00% |
| Decision Tree | 0.7155 | 0.7928 | 20.66% |
| **Random Forest** | **0.8596** | **0.8792** | **10.37%** |
| XGBoost | 0.7540 | 0.7722 | 7.28% |

> **Key Insight:** Random Forest delivered the best accuracy, while linear models showed the most stable generalization (lowest overfit gap). KNN benefited the most from scaling.

---

## 🗂️ Project Structure

```
├── notebook/
│   ├── main.ipynb                  # Primary analysis & experimentation notebook
│   ├── baseline_scaled.ipynb       # Baseline vs. scaled comparison
│   └── utils/                      # Notebook-specific helper functions
│       ├── model_evaluation.py     #   Evaluation metrics (R², MAE, RMSE)
│       ├── remove_outlier.py       #   IQR-based outlier removal
│       └── split_dataset.py        #   Train-test splitting
│
├── src/                            # Core source modules
│   ├── data_loader.py              #   DataLoader class (CSV/Excel/JSON)
│   └── preprocess.py               #   Preprocessor class (StandardScaler, RobustScaler)
│
├── utils/                          # Shared utility library
│   ├── common_util.py              #   Column descriptions, dataset helpers
│   ├── compare_models.py           #   Multi-model training & comparison
│   ├── outliers_iqr.py             #   OutlierIQR class
│   ├── plot_results.py             #   Visualization functions
│   └── split_dataset.py            #   Dataset splitting utility
│
├── results/                        # Saved experiment results (CSV)
│   ├── baseline_results.csv
│   ├── baseline_scaled_results.csv
│   ├── cleaned_results.csv
│   └── clean_scaled_results.csv
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Environment** | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/harmandeep2993/ml-house-price-predictions-project.git
cd ml-house-price-predictions-project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebook/main.ipynb
```

---

## 📈 Key Features Explored

| Feature | Description |
|---|---|
| `sqft_living` | Interior living area in square feet |
| `grade` | Construction quality grade (1–13) |
| `bathrooms` | Total number of bathrooms |
| `bedrooms` | Total number of bedrooms |
| `floors` | Number of floors |
| `waterfront` | Waterfront property (0/1) |
| `view` | View quality rating (0–4) |
| `condition` | Overall condition rating (1–5) |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year of last renovation |
| `zipcode` | ZIP code (location proxy) |
| `lat` / `long` | Geographic coordinates |
| `sqft_living15` | Avg. living area of 15 nearest neighbors |
| `sqft_lot15` | Avg. lot size of 15 nearest neighbors |

---

## 💡 Key Takeaways

- **Data quality matters:** Cleaning alone boosted Linear Regression R² by ~10 percentage points.
- **Scaling is essential for distance-based models:** KNN improved from 0.50 → 0.81 R² after scaling.
- **Ensemble methods generalize best:** Random Forest consistently outperformed other models.
- **Overfitting awareness:** Decision Tree achieved near-perfect training R² (0.999) but poor test R² — a textbook overfitting example tracked via the overfit gap metric.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Harmandeep Singh**

Built as a portfolio project demonstrating end-to-end machine learning skills — from data wrangling and feature engineering to model selection and performance analysis.
