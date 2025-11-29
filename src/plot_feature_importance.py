import pandas as pd
import plotly.express as px


if __name__ == "__main__":
    feat_imp = pd.read_csv(r"D:\Projects\Finance\Financial_Retention_Behavior_Modeling\data\feature_importance.csv")
    topk = feat_imp.head(10)

    fig = px.bar(
        topk,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Churn Feature Importances",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.show()
