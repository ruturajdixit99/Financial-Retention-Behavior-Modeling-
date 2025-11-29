import numpy as np
import pandas as pd


def simulate_customers(n_customers: int = 3000, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)

    customer_id = np.arange(1, n_customers + 1)

    tenure_months = np.random.randint(1, 60, size=n_customers)
    avg_monthly_spend = np.random.gamma(shape=2.0, scale=200.0, size=n_customers)
    tx_30d = np.random.poisson(lam=2, size=n_customers)
    tx_90d = tx_30d + np.random.poisson(lam=3, size=n_customers)
    max_gap_days = np.random.randint(1, 90, size=n_customers)
    is_high_risk_segment = np.random.binomial(1, 0.3, size=n_customers)

    card_types = np.random.choice(["basic", "gold", "platinum"], size=n_customers, p=[0.6, 0.3, 0.1])

    # True churn probability (latent function)
    base_churn = 0.20
    p_churn = (
        base_churn
        + 0.002 * (30 - tx_90d)    # fewer transactions → higher churn
        + 0.003 * (max_gap_days)   # bigger gaps → higher churn
        - 0.001 * tenure_months    # longer tenure → lower churn
        - 0.0004 * avg_monthly_spend / 10
        + 0.10 * is_high_risk_segment
    )

    p_churn = np.clip(p_churn, 0.01, 0.85)
    churned = np.random.binomial(1, p_churn)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "avg_monthly_spend": avg_monthly_spend,
        "transactions_30d": tx_30d,
        "transactions_90d": tx_90d,
        "max_gap_days": max_gap_days,
        "is_high_risk_segment": is_high_risk_segment,
        "card_type": card_types,
        "churned": churned
    })

    return df


if __name__ == "__main__":
    df = simulate_customers()
    df.to_csv("D:\Projects\Finance\Financial_Retention_Behavior_Modeling\data\customer_retention_simulated.csv", index=False)
    print(df.head())
