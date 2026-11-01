"""
Heart Disease Risk Prediction - Streamlit dashboard
Run:  streamlit run dashboard.py
Educational use only - not a diagnostic tool.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from heart_disease_project import COLUMNS, evaluate, load_data

DATA_PATH = "data/processed.cleveland.data"
PREDICTORS = COLUMNS[:-1]

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="wide")


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_resource
def train(seed, test_size, depth):
    df = get_data()
    X, y = df[PREDICTORS], df["heart_disease"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    models = {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed))]),
        "Decision Tree": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=depth, random_state=seed))]),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)
    return models, X_tr, X_te, y_tr, y_te


df = get_data()

# ---------------- Sidebar ----------------
st.sidebar.title("❤️ Heart Disease PBL")
page = st.sidebar.radio("Page", ["Overview", "EDA", "Model comparison", "Try a prediction"])
st.sidebar.markdown("---")
st.sidebar.subheader("Model settings")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.25, 0.05)
depth = st.sidebar.slider("Decision Tree max depth", 2, 10, 4)
st.sidebar.caption("Educational project. Not for diagnosis or treatment decisions.")

# ---------------- Overview ----------------
if page == "Overview":
    st.title("Heart Disease Risk Prediction")
    st.caption("Processed Cleveland dataset • Logistic Regression vs Decision Tree")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df))
    c2.metric("Predictors", len(PREDICTORS))
    c3.metric("No disease (0)", int((df.heart_disease == 0).sum()))
    c4.metric("Disease recorded (1)", int((df.heart_disease == 1).sum()))

    st.subheader("Missing values")
    miss = pd.DataFrame({"Valid N": df.notna().sum(), "Missing N": df.isna().sum()})
    miss["Missing %"] = (miss["Missing N"] / len(df) * 100).round(1)
    st.dataframe(miss[miss["Missing N"] > 0])

    st.subheader("Descriptive statistics")
    num = ["AGE", "TREST", "CHOL", "THALACH", "OLDPEAK", "heart_disease"]
    st.dataframe(df[num].agg(["count", "min", "max", "mean", "std"]).T.round(4))

    with st.expander("Preview data"):
        st.dataframe(df.head(20))
    st.info("Target: NUM = 0 → heart_disease = 0; NUM 1–4 → heart_disease = 1.")

# ---------------- EDA ----------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    left, right = st.columns(2)

    with left:
        st.subheader("Target distribution")
        fig, ax = plt.subplots()
        counts = df.heart_disease.value_counts().sort_index()
        ax.bar(["0", "1"], counts.values, color=["#4C9F70", "#C8553D"])
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        ax.set_xlabel("heart_disease"); ax.set_ylabel("Count")
        st.pyplot(fig)

    with right:
        st.subheader("Histogram")
        col = st.selectbox("Variable", ["AGE", "CHOL", "TREST", "THALACH", "OLDPEAK"])
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20, color="#3D6FA3", edgecolor="white")
        ax.set_xlabel(col); ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.subheader("Boxplot by class")
    bcol = st.selectbox("Variable ", ["AGE", "CHOL", "THALACH", "OLDPEAK", "TREST"], key="box")
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column=bcol, by="heart_disease", ax=ax, grid=False)
    plt.suptitle(""); ax.set_title(f"{bcol} by heart_disease")
    st.pyplot(fig)

    st.subheader("Crosstab with target")
    ccol = st.selectbox("Categorical variable",
                        ["SEX", "CP", "FBS", "RESTECG", "EXANG", "SLOPE", "CA", "THAL"])
    st.dataframe(pd.crosstab(df[ccol], df["heart_disease"]))

# ---------------- Model comparison ----------------
elif page == "Model comparison":
    st.title("Model comparison")
    models, X_tr, X_te, y_tr, y_te = train(seed, test_size, depth)
    st.write(f"Training records: **{len(X_tr)}** | Test records: **{len(X_te)}**")

    res = pd.DataFrame({n: evaluate(n, m, X_te, y_te) for n, m in models.items()})
    st.dataframe(res.round(3))
    st.download_button("Download metrics CSV", res.round(3).to_csv().encode(),
                       "test_metrics.csv", "text/csv")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC curves")
        fig, ax = plt.subplots()
        for n, m in models.items():
            fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te)[:, 1])
            ax.plot(fpr, tpr, label=f"{n} (AUC {res.loc['ROC-AUC', n]:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(); st.pyplot(fig)
    with c2:
        st.subheader("Confusion matrix")
        pick = st.radio("Model", list(models), horizontal=True)
        cm = confusion_matrix(y_te, models[pick].predict(X_te))
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in pd.DataFrame(cm).stack().items():
            ax.text(j, i, v, ha="center", va="center", fontsize=16)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.caption("Results depend on the split and settings; the test set is small. "
               "Not evidence of clinical usefulness.")

# ---------------- Prediction ----------------
else:
    st.title("Try a prediction (classroom demo)")
    st.warning("Demo of how the trained model behaves. It is NOT medical advice "
               "and must not be used for any real screening or treatment decision.")
    models, *_ = train(seed, test_size, depth)
    model_name = st.radio("Model", list(models), horizontal=True)

    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(PREDICTORS):
        lo, hi = float(df[f].min()), float(df[f].max())
        default = float(df[f].median())
        with cols[i % 3]:
            vals[f] = st.number_input(f, lo, hi, default,
                                      step=0.1 if f == "OLDPEAK" else 1.0)

    if st.button("Predict"):
        p = models[model_name].predict_proba(pd.DataFrame([vals]))[0, 1]
        st.metric("Model-estimated probability of the positive class", f"{p:.1%}")
        st.progress(float(p))
        st.caption("Positive class = disease recorded in the dataset (NUM 1–4).")
