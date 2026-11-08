import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import compute_metrics, encode_target

st.set_page_config(page_title="Classifier Lab", page_icon="🧪", layout="wide")

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}


@st.cache_resource
def load_model(slug):
    return joblib.load(f"model/{slug}.joblib")


meta = json.load(open("model/meta.json"))
st.title("🧪 Classifier Lab")
st.caption("Pick a saved model, upload test data, and inspect how it performs. No retraining happens here.")

with st.sidebar:
    st.header("Controls")
    choice = st.selectbox("Model", list(MODELS))
    up = st.file_uploader("Test data (CSV)", type="csv")
    st.markdown(f"**Required columns**: {len(meta['features'])} features + `{meta['target']}`")
    st.download_button("Download sample test CSV", open("test_data.csv", "rb"),
                       file_name="test_data.csv")

if up is None:
    st.info("Upload a test CSV from the sidebar to begin. You can grab the sample file there.")
    st.subheader("Stored comparison (from training run)")
    st.dataframe(pd.read_csv("model/metrics.csv"), use_container_width=True)
    st.stop()

df = pd.read_csv(up)
missing = [c for c in meta["features"] + [meta["target"]] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()

X = df[meta["features"]]
y, unknown = encode_target(df[meta["target"]], meta["classes"])
if unknown:
    st.error(f"Target column contains labels the models were never trained on: {unknown}")
    st.stop()
y = y.astype(int)
n_classes = len(meta["classes"])
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(), use_container_width=True)


def evaluate(name):
    m = load_model(MODELS[name])
    pred = m.predict(X)
    return pred, compute_metrics(y, pred, m.predict_proba(X), n_classes)


pred, scores = evaluate(choice)
st.subheader(f"Results: {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, scores.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    labels = list(range(n_classes))
    names = meta["classes"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, pred, labels=labels), annot=True, fmt="d",
                cmap="crest", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame(classification_report(y, pred, labels=labels, target_names=names, output_dict=True, zero_division=0)).T.round(3),
                 use_container_width=True)

st.subheader("All models on this upload")
table = pd.DataFrame({n: evaluate(n)[1] for n in MODELS}).T.round(4)
st.dataframe(table.style.highlight_max(axis=0, color="#2e7d5b55"), use_container_width=True)
