import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib


if __name__ == "__main__":
    df = pd.read_csv(r"D:\Projects\Finance\Financial_Retention_Behavior_Modeling\data\customer_retention_simulated.csv")

    target = "churned"
    y = df[target]

    numeric_features = [
        "tenure_months", "avg_monthly_spend", "transactions_30d",
        "transactions_90d", "max_gap_days"
    ]
    categorical_features = ["card_type", "is_high_risk_segment"]

    X = df[numeric_features + categorical_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(clf, "D:\Projects\Finance\Financial_Retention_Behavior_Modeling\data\churn_model.joblib")

    # Extract feature importance back into original feature space
    rf = clf.named_steps["model"]
    ohe = clf.named_steps["preprocessor"].named_transformers_["cat"]

    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)

    importances = rf.feature_importances_

    feat_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    feat_imp.to_csv(r"D:\Projects\Finance\Financial_Retention_Behavior_Modeling\data\feature_importance.csv", index=False)
    print("\nTop features:")
    print(feat_imp.head(10))
