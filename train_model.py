import os
import joblib
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]

FEATURE_QUERY = """
SELECT
    stop_id,
    eva_no,
    SPLIT_PART(train_name, ' ', 1) AS train_category, -- e.g., RE, RB, S, ICE
    train_name,
    EXTRACT(HOUR FROM planned_dep) AS hour_of_day,
    EXTRACT(DOW FROM planned_dep) AS day_of_week,     -- 0=Sun, 6=Sat
    CASE WHEN planned_platform IS DISTINCT FROM actual_platform THEN 1 ELSE 0 END AS platform_changed,
    -- Rolling 1-hour station congestion prior to this train's scheduled departure
    COALESCE(
        AVG(delay_min) OVER (
            PARTITION BY eva_no
            ORDER BY planned_dep
            RANGE BETWEEN INTERVAL '60 minutes' PRECEDING AND INTERVAL '1 minute' PRECEDING
        ), 0
    ) AS rolling_1h_station_delay,
    -- Binary Target: 1 if severely delayed (>= 6 min) or cancelled, else 0
    CASE WHEN is_cancelled OR delay_min >= 6 THEN 1 ELSE 0 END AS target_disrupted
FROM fact_train_stops
WHERE planned_dep < NOW() - INTERVAL '15 minutes'; -- Only train on completed departures
"""


def train():
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql(FEATURE_QUERY, conn)
    conn.close()

    print(f"Loaded {len(df)} completed stop records for training.")
    if len(df) < 100:
        print(
            "Warning: Under 100 rows so far. Let GitHub Actions collect more data for stronger accuracy!"
        )

    features = [
        "eva_no",
        "train_category",
        "train_name",
        "hour_of_day",
        "day_of_week",
        "platform_changed",
        "rolling_1h_station_delay",
    ]
    X = df[features]
    y = df["target_disrupted"]

    # Chronological 80/20 split (never randomly shuffle time-series data!)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    cat_cols = ["eva_no", "train_category", "train_name"]
    num_cols = [
        "hour_of_day",
        "day_of_week",
        "platform_changed",
        "rolling_1h_station_delay",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # Evaluate on unseen future rows
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("\n--- Validation Report ---")
    print(classification_report(y_test, preds))
    if len(set(y_test)) > 1:
        print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.3f}")

    # Save trained artifact
    joblib.dump(model, "rail_delay_model.joblib")
    print("Saved trained model to rail_delay_model.joblib")


if __name__ == "__main__":
    train()