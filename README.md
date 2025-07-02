# 📊 Scientific Paper Citation Prediction

This repository contains my solution to a Machine Learning challenge where the goal was to **predict the number of citations** a scientific paper would receive, using only paper metadata (e.g. authors, abstract length, title, references).

---

## 🧠 Task

> Predict `n_citation` for each paper in the test set, using a model trained on paper metadata from the training set.

Evaluation metric: **Mean Absolute Error (MAE)**  
🎯 My final score: **30.2456 MAE**

---

## 🔍 Dataset

The original dataset is too large to upload here. It includes:
- `train.json`: list of papers with metadata + citation counts
- `test.json`: list of papers with metadata only

Each record is a dictionary.  
To reproduce results, you must provide the same dataset locally.

---

## 🧪 Models Implemented

| File                  | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `My_baseline.py`      | DummyRegressor and Ridge baseline                           |
| `Run_Model.py`        | RandomForest & GradientBoosting with GridSearchCV tuning     |
| `Best_Model.py`       | Final tuned ensemble models (used 1% data for experimentation)|
| `ML_Assignment_Angelos_Moulas.py` | Final LightGBM model with log-transform & feature engineering |
| `TEMP_Model.py`       | Variant snapshot of baseline logic                           |

---

## 🛠 Features Used

- Paper year (numeric)
- Title length (word count)
- Abstract length (word count)
- Number of references (list length)
- TF-IDF of authors' names

---

## 📁 Project Files

- 📄 `Machine_Learning_Assignment_INSTR.pdf`: Full assignment description
- 📦 `predicted.zip`: Final prediction results for the test set (Codalab submission format)

---

## 🚀 Results

The final model was trained with LightGBM using a custom feature engineering pipeline and log-transformation on the citation counts.  
It achieved strong performance compared to the baseline.

---

