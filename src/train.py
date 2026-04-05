import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib
import argparse
import os
import json

# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Arguments for experiment variation
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="rf")
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--features", type=str, default="all")
args = parser.parse_args()

# Load data
df = pd.read_csv("data/data.csv")

# Basic preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Feature selection
if args.features == "reduced":
    df = df[['Survived', 'Pclass', 'Sex']]

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model selection
if args.model == "rf":
    model = RandomForestClassifier(max_depth=args.max_depth)
else:
    model = LogisticRegression()

# MLflow experiment
mlflow.set_experiment("2022bcs0162_experiment")

with mlflow.start_run():

    # Train model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Log parameters
    mlflow.log_param("model", args.model)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("features", args.features)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Save model
    joblib.dump(model, "models/model.pkl")
    mlflow.log_artifact("models/model.pkl")

    # Print output (for console)
    print({
        "accuracy": acc,
        "f1_score": f1,
        "mse": mse,
        "r2": r2,
        "Name": "Santosh Hrushith Yelamanchili",
        "Roll": "2022bcs0162"
    })

    # Save metrics JSON (for CI/CD)
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "mse": mse,
        "r2": r2,
        "Name": "Santosh Hrushith Yelamanchili",
        "Roll": "2022bcs0162"
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f)