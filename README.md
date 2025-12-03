# shap-explainability

A collection of Python notebooks demonstrating how to interpret Machine Learning models using [SHAP (SHapley Additive exPlanations)](https://github.com/shap/shap).

## Overview

This repository contains two practical examples of "Glass Box AI"—moving beyond black-box predictions to understand the *why* behind model decisions.

---
![Shapley](./img/Shapley.png)
---

### 1. Finance & Loan Approval (`finance_loan_shap.ipynb`)
**Domain:** Tabular Data (Structured)  
**Model:** XGBoost Classifier  
**Use Case:** - Predicting loan defaults based on income, credit score, and debt.
- Generating regulatory "Reason Codes" (e.g., "Why was this specific applicant rejected?").
- Visualizing global risk factors vs. individual applicant risk.

### 2. LLM Sentiment Analysis (`llm_sentiment_shap.ipynb`)
**Domain:** Natural Language Processing (Unstructured)  
**Model:** Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`)  
**Use Case:**
- Explaining sentiment analysis predictions.
- Visualizing token-level attribution (which specific words drove the Positive/Negative score).
- Debugging model bias in text.

## Installation

To run these notebooks locally, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/roguetrainer/shap-explainability.git](https://github.com/roguetrainer/shap-explainability.git)
   ```
2. Navigate to the folder:
   ```bash
   cd shap-explainability
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
