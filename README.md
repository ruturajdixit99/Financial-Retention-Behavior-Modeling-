# Financial Retention Behavior Modeling

This project focuses on predicting customer churn for a financial services company using simulated behavioral data. The objective is to understand **why customers churn**, **which attributes influence their decision**, and **how businesses can intervene early** using actionable strategies.

---

## 📂 Project Structure
📁 Financial_Retention_Behavior_Modeling
├── data/
│ ├── customer_retention_simulated.csv
│ ├── feature_importance.csv
│ ├── churn_model.joblib
├── src/
│ ├── simulate_customers.py
│ ├── train_churn_model.py
│ ├── plot_feature_importance.py
└── README.md

yaml
Copy code

---

## 🔄 Workflow Overview

1. **simulate_customers.py**  
   Generates synthetic customer churn dataset with behavioral and financial features.

2. **train_churn_model.py**  
   - Preprocesses data (scaling + one-hot encoding)  
   - Trains a Random Forest Classifier  
   - Evaluates using ROC-AUC, Precision, Recall  
   - Extracts feature importance  

3. **plot_feature_importance.py**  
   Visualizes top churn drivers.

---

## 📊 Model Evaluation

| Metric | Score |
|--------|--------|
| ROC-AUC | 0.6069 |
| Accuracy | 59% |
| Recall (Churn Class) | 0.47 |
| Precision (Churn Class) | 0.45 |

🔎 Interpretation:
- Model identifies churners better than random guessing.
- Indicates behavioral patterns influence churn.
- Still improvable with additional features (complaints, sentiment, demographics).

---


<img width="2133" height="1012" alt="newplot (2)" src="https://github.com/user-attachments/assets/dee3147d-ea93-4e2b-a820-54b5b9034c41" />


## 🏆 Top Churn Predictors

| Feature | Business Interpretation |
|---------|--------------------------|
| max_gap_days | High inactivity = major churn risk |
| avg_monthly_spend | Higher spenders are loyal |
| tenure_months | Longer relationships reduce churn |
| transactions_90d | More transactions = higher engagement |
| is_high_risk_segment | Behavioral risk indicator |
| card_type | Premium card users churn less |

---

## 🎯 Business Insights & Retention Strategies

| Insight | Strategy |
|---------|----------|
| Customers with long inactivity are most likely to leave | Reactivation emails, targeted discounts |
| Low-spend and low-transaction users show early churn signs | Upselling and cashback incentives |
| First-year customers churn more frequently | Onboarding programs and loyalty rewards |
| Premium card users are more loyal | Promote card upgrades to basic users |

---

## 🛠 Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn (RandomForestClassifier)**
- **Plotly**
- **Joblib**

---

## 🚀 Future Enhancements
- Add customer complaint logs & sentiment analysis  
- Use Gradient Boosting / XGBoost for better performance  
- Build dashboard for churn risk scoring  
- Deploy with Flask/Streamlit

---

📧 For any improvements or enhancements, feel free to collaborate!
