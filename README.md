# Fake Job Posting Detection

A machine learning system that classifies job postings as **fraudulent** or **legitimate** by analyzing textual (job description, requirements, company profile) and structured job-related data.

## Overview

Fraudulent job postings are used for scams, resume harvesting, and phishing. This project builds a text classification pipeline that flags suspicious postings using NLP feature extraction and multiple classification algorithms, evaluated beyond simple accuracy due to significant class imbalance in the data.

## Dataset

- **Source:** [EMSCAD – Employment Scam Aegean Dataset](http://emscad.samos.aegean.gr/), redistributed via Kaggle as ["Real or Fake] Fake Jobposting Prediction"](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size:** ~17,880 job postings, of which ~800 (≈5%) are fraudulent
- **File:** `data/fake_job_postings.csv`
- **Key columns:** `title`, `location`, `department`, `salary_range`, `company_profile`, `description`, `requirements`, `benefits`, `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `fraudulent` (target)

## Project Structure

```
fake-job-posting-detection/
├── data/
│   └── fake_job_postings.csv
├── notebook.ipynb              # Full analysis & modeling pipeline
├── README.md
└── requirements.txt
```

## Methodology

1. **EDA** — missing value analysis, class distribution, text length patterns by class
2. **Preprocessing** — null handling, text cleaning (URL/number/punctuation removal), combining title + company profile + description + requirements + benefits into a single text field
3. **Feature extraction** — TF-IDF and CountVectorizer (unigrams + bigrams, top 5,000 features)
4. **Class imbalance handling** — SMOTE oversampling on the training set
5. **Models trained** — Logistic Regression, Multinomial Naive Bayes, Random Forest, Linear SVM
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices (accuracy alone is misleading given the ~95/5 class split)
7. **Hyperparameter tuning** — GridSearchCV on Logistic Regression (`C`, penalty, solver)
8. **Interpretability** — top TF-IDF terms driving fraud vs. legitimate predictions
9. **Prediction system** — a `predict_job_posting()` function that takes raw job fields and returns a label + confidence score

## Results

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|----------|-----------|--------|------|---------|
| Logistic Regression | –        | –         | –      | –    | –       |
| Naive Bayes         | –        | –         | –      | –    | –       |
| Random Forest       | –        | –         | –      | –    | –       |
| Linear SVM          | –        | –         | –      | –    | –       |

*(Fill in after running the notebook — results vary slightly by random seed / vectorizer settings.)*

## How to Run

1. Clone the repo:
```bash
   git clone https://github.com/<your-username>/fake-job-posting-detection.git
```
2. Open `notebook.ipynb` in Google Colab or Jupyter
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Run all cells — the notebook loads `data/fake_job_postings.csv` directly

## Limitations

- Severe class imbalance (~5% fraud) means the model can still miss novel or well-written scam listings
- TF-IDF is bag-of-words based and doesn't capture deeper context, sarcasm, or paraphrased scam language
- Model is trained on historical scam patterns and may not generalize to new fraud tactics
- Structured features (missing logo, no salary listed, no company profile) are strong signals but only lightly incorporated here — a hybrid text + metadata model could improve results further

## Acknowledgments

Dataset originally published by the University of the Aegean, Laboratory of Information & Communication Systems Security (EMSCAD), and made publicly available via Kaggle.
