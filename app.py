import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import plotly.express as px

# =======================
# Load Data & Model
# =======================
@st.cache_data
def load_data():
    return pd.read_csv("out1.csv")

@st.cache_resource
def load_model():
    return joblib.load("flood_model.pkl")

df = load_data()
model = load_model()

# Load saved evaluation artifacts
metrics = json.load(open("metrics.json"))
cm = np.load("confusion_matrix.npy")

# =======================
# UI: Title & Inputs
# =======================
st.title("India Rainfall – Flood Risk Prediction")

subdivisions = sorted(df["SUBDIVISION"].dropna().unique())
selected_sub = st.selectbox("Select Subdivision", subdivisions)

months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
selected_month = st.selectbox("Select Month", months)

rainfall_amount = st.number_input(
    f"Rainfall in {selected_month} (mm)", min_value=0.0, step=1.0
)

if rainfall_amount == 0:
    st.warning("Please enter a positive rainfall value for meaningful prediction.")

# =======================
# Prepare Features
# =======================
hist = df[df["SUBDIVISION"] == selected_sub]
mar_may = hist["Mar-May"].median()

avgjune = (
    rainfall_amount / 3
    if selected_month.upper() == "JUN"
    else hist["avgjune"].median()
)

if selected_month.upper() == "MAY":
    sub_val = abs(rainfall_amount - hist["JUN"].median())
elif selected_month.upper() == "JUN":
    sub_val = abs(hist["MAY"].median() - rainfall_amount)
else:
    sub_val = hist["sub"].median()

X_input = np.array([[mar_may, avgjune, sub_val]])

# =======================
# Prediction
# =======================
if st.button("Predict Flood Risk") and rainfall_amount > 0:
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Flood Risk:** {'Flood' if pred==1 else 'No Flood'}")
    st.write(f"**Probability of Flood:** {prob:.1%}")

    st.subheader("Feature Contribution (Logistic Regression)")
    features = ['Mar-May', 'avgjune', 'sub']
    coef = model.coef_[0]
    contrib = X_input.flatten() * coef
    contrib_df = pd.DataFrame({
        "Feature": features,
        "Value": X_input.flatten(),
        "Coefficient": coef,
        "Contribution": contrib
    })
    st.table(contrib_df)

    fig_contrib, ax_contrib = plt.subplots()
    ax_contrib.barh(features, np.abs(contrib), color="teal")
    ax_contrib.set_xlabel("Absolute Influence on Flood Risk")
    st.pyplot(fig_contrib)

# =======================
# Model Evaluation
# =======================
st.header("Model Evaluation")

st.subheader("Model Accuracy")
acc_df = pd.DataFrame({
    "Metric": ["Train", "Test", "Cross-val mean"],
    "Accuracy": [
        metrics["train_accuracy"],
        metrics["test_accuracy"],
        metrics["cv_mean"]
    ]
})
fig_acc, ax_acc = plt.subplots()
ax_acc.bar(acc_df["Metric"], acc_df["Accuracy"], color="skyblue")
ax_acc.set_ylim(0, 1)
ax_acc.set_ylabel("Accuracy")
for i, v in enumerate(acc_df["Accuracy"]):
    ax_acc.text(i, v + 0.01, f"{v:.2f}", ha='center')
st.pyplot(fig_acc)

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, cmap="Blues")
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
st.pyplot(fig_cm)



# Matplotlib line chart with labels
fig_hist, ax_hist = plt.subplots()
ax_hist.plot(hist["YEAR"], hist["Jun-Sep"], color="deepskyblue")
ax_hist.set_xlabel("Year")
ax_hist.set_ylabel("Jun–Sep Rainfall (mm)")
st.pyplot(fig_hist)

flood_years = hist[hist["flood"] == 1]["YEAR"].tolist()
st.write("Flood Years:", flood_years)

pivot = df.pivot(index="YEAR", columns="SUBDIVISION", values="Jun-Sep")
pivot = pivot.sort_index()  # ensure correct year order




