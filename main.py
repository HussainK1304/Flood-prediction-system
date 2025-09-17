import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
import joblib
import json

warnings.filterwarnings('ignore')

# === 1. Load the data ===
csv_path = r"C:\Users\nisha\OneDrive\Desktop\flood_risk_project\rainfall in india 1901-2015.csv"
x = pd.read_csv(csv_path)

print("Original data shape:", x.shape)
print("Columns in dataset:", x.columns.tolist())

print("\nMissing values per column:")
print(x.isnull().sum())

print("\nBasic statistics:")
print(x[['Jun-Sep', 'JUN', 'MAY', 'Mar-May']].describe())

# === 2. Feature engineering ===
y1 = x["YEAR"].tolist()
x1 = x["Jun-Sep"].tolist()
z1 = x["JUN"].tolist()
w1 = x["MAY"].tolist()

plt.figure(figsize=(12, 6))
plt.plot(y1, x1, '*', alpha=0.7)
plt.title('Jun-Sep Rainfall Over Years (1901-2015)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

flood, june, sub = [], [], []
for i in range(len(x1)):
    if pd.notna(x1[i]) and x1[i] > 2400:
        flood.append(1)
    elif pd.notna(x1[i]):
        flood.append(0)
    else:
        flood.append(np.nan)

for k in range(len(z1)):
    june.append(z1[k]/3 if pd.notna(z1[k]) else np.nan)

for k in range(len(z1)):
    if pd.notna(w1[k]) and pd.notna(z1[k]):
        sub.append(abs(w1[k]-z1[k]))
    else:
        sub.append(np.nan)

x["flood"] = flood
x["avgjune"] = june
x["sub"] = sub

x.to_csv("out1.csv", index=False)
print("Enhanced dataset saved as out1.csv")

# === 3. Machine-learning model ===
feature_cols = ['Mar-May', 'avgjune', 'sub']
target_col = 'flood'

X_df = x[feature_cols]
y_df = x[target_col]

mask = y_df.notna()
X_df = X_df[mask]
y_df = y_df[mask]

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X_df)
y = y_df.values.astype(int)

print(f"Dataset size after cleaning: {X.shape}, floods: {y.sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Logistic Regression ---
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

train_accuracy = lr.score(X_train, y_train)
test_accuracy = lr.score(X_test, y_test)
cv_scores = cross_val_score(lr, X, y, cv=5)

print("\nLogistic Regression:")
print(f"Train acc: {train_accuracy:.3f}")
print(f"Test  acc: {test_accuracy:.3f}")
print("Cross-val acc:", cv_scores.mean())

print("\nClassification report:")
print(classification_report(y_test, lr.predict(X_test)))
print("Confusion matrix:")
print(confusion_matrix(y_test, lr.predict(X_test)))

joblib.dump(lr, "flood_model.pkl")
print("Model saved to flood_model.pkl")

# === Save metrics for the Streamlit app ===
metrics = {
    "train_accuracy": float(train_accuracy),
    "test_accuracy": float(test_accuracy),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std())
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

np.save("confusion_matrix.npy", confusion_matrix(y_test, lr.predict(X_test)))
print("Metrics and artifacts saved.")
