import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib
df = pd.read_csv("data/winequality-red.csv")
print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(df.head())

print("\nShape of Dataset:")
print(df.shape)
print("\nMissing Values")
print(df.isnull().sum())
print("\nDuplicate Rows")

print(df.duplicated().sum())
df = df.drop_duplicates()

print("\nShape After Removing Duplicates")

print(df.shape)
X = df.drop("quality", axis=1)

y = df["quality"]

print("\nFeature Matrix Shape:", X.shape)

print("Target Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape")

print(X_train.shape)

print("\nTesting Data Shape")

print(X_test.shape)
# ==========================
# Train Decision Tree Model
# ==========================

print("\n" + "=" * 50)
print("Training Decision Tree Model")
print("=" * 50)

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)
# ==========================
# Decision Tree Prediction
# ==========================

dt_predictions = dt_model.predict(X_test)
# ==========================
# Decision Tree Accuracy
# ==========================

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:")

print(dt_accuracy)
# ==========================
# Decision Tree Report
# ==========================

print("\nClassification Report")

print(classification_report(y_test, dt_predictions))
# ==========================
# Create Confusion Matrix
# ==========================

cm = confusion_matrix(y_test, dt_predictions)

print(cm)
# ==========================
# Plot Confusion Matrix
# ==========================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d"
)

plt.title("Decision Tree Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
# ==========================
# Random Forest
# ==========================

print("\n" + "=" * 50)
print("Training Random Forest Model")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
# ==========================
# Random Forest Prediction
# ==========================

rf_predictions = rf_model.predict(X_test)

#Accuracy

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Accuracy")

print(rf_accuracy)

#Classification Report
print("\nRandom Forest Classification Report")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

#Confusion Matrix
rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

print("\nRandom Forest Confusion Matrix")

print(rf_cm)

#Plot Confusion Matrix
plt.figure(figsize=(8,6))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

#Compare Both Models
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(f"Decision Tree Accuracy : {dt_accuracy:.4f}")

print(f"Random Forest Accuracy : {rf_accuracy:.4f}")
# ==========================
# XGBoost
# ==========================

print("\n" + "=" * 50)
print("Training XGBoost Model")
print("=" * 50)

# Create the XGBoost model
xgb_model = XGBClassifier(
    objective="multi:softmax",
    num_class=6,
    eval_metric="mlogloss",
    random_state=42
)

# Train the model
# Convert labels from 3-8 to 0-5
xgb_model.fit(X_train, y_train - 3)

# ==========================
# Prediction
# ==========================

# Predict on the test data
xgb_predictions = xgb_model.predict(X_test)

# Convert predictions back from 0-5 to 3-8
xgb_predictions = xgb_predictions + 3

# ==========================
# Accuracy
# ==========================

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print("\nXGBoost Accuracy:")
print(xgb_accuracy)

# ==========================
# Classification Report
# ==========================

print("\nXGBoost Classification Report")

print(
    classification_report(
        y_test,
        xgb_predictions
    )
)

# ==========================
# Confusion Matrix
# ==========================

xgb_cm = confusion_matrix(
    y_test,
    xgb_predictions
)

print("\nXGBoost Confusion Matrix")

print(xgb_cm)

# ==========================
# Plot Confusion Matrix
# ==========================

plt.figure(figsize=(8,6))

sns.heatmap(
    xgb_cm,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("XGBoost Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==========================
# Compare All Models
# ==========================

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

print(f"Decision Tree : {dt_accuracy:.4f}")
print(f"Random Forest : {rf_accuracy:.4f}")
print(f"XGBoost       : {xgb_accuracy:.4f}")
# ==========================
# Save Best Model
# ==========================

best_model = rf_model

joblib.dump(
    best_model,
    "models/wine_quality_model.pkl"
)

print("\nRandom Forest model saved successfully!")