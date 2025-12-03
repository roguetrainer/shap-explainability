import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 1. Generate Synthetic "Loan Approval" Data
np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    "Income_Annual": np.random.normal(50000, 15000, n_samples).astype(int),
    "Credit_Score": np.random.normal(650, 100, n_samples).astype(int),
    "Debt_to_Income": np.random.uniform(0.1, 0.6, n_samples),
    "Loan_Amount": np.random.normal(15000, 5000, n_samples).astype(int),
    "Years_Employed": np.random.randint(0, 20, n_samples),
    "Has_Prior_Default": np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
})

risk_score = (
    (data["Debt_to_Income"] * 100) - 
    (data["Credit_Score"] / 10) - 
    (data["Income_Annual"] / 2000) + 
    (data["Has_Prior_Default"] * 50)
)
risk_score += np.random.normal(0, 10, n_samples)
data["Defaulted"] = (risk_score > -20).astype(int)

# 2. Train Model
X = data.drop("Defaulted", axis=1)
y = data["Defaulted"]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)
print(f"Model Accuracy: {model.score(X, y):.2f}")

# 3. Compute Shapley Values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# 4. Global Interpretability
plt.figure(figsize=(10, 6))
plt.title("Global Feature Importance (Beeswarm)")
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.show()

# 5. Local Interpretability
high_risk_idx = np.where(y == 1)[0][0]
print(f"\n--- Analysis for Applicant #{high_risk_idx} ---")
print(X.iloc[high_risk_idx])

plt.figure(figsize=(10, 6))
plt.title(f"Why Applicant #{high_risk_idx} was predicted to Default")
shap.plots.waterfall(shap_values[high_risk_idx], show=False)
plt.tight_layout()
plt.show()

# 6. Reason Codes
applicant_shap = pd.Series(shap_values[high_risk_idx].values, index=X.columns)
print("\n--- Reason Codes (Top Factors increasing Risk) ---")
risk_factors = applicant_shap.sort_values(ascending=False).head(3)
for feature, impact in risk_factors.items():
    if impact > 0:
        val = X.iloc[high_risk_idx][feature]
        print(f"Factor: {feature} (Value: {val:.2f}) increased risk score by {impact:.2f}")
