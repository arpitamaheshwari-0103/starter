import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="SkillSync AI", layout="wide")

st.title("🚀 SkillSync AI – Career Intelligence Platform")
st.markdown("### Bridging Skill Gaps using Data & AI")
st.caption("Note: All salary values represent Annual Income in INR (₹ per year).")

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("SkillSync_AI_Candidates.csv")

# -----------------------
# AUTO DETECT COLUMNS
# -----------------------
columns = df.columns

salary_col = [col for col in columns if "salary" in col.lower()][0]
skills_col = [col for col in columns if "skill" in col.lower()][0]
exp_col = [col for col in columns if "exp" in col.lower()][0]
emp_col = [col for col in columns if "employ" in col.lower()][0]

# -----------------------
# FIX EMPLOYABILITY
# -----------------------
emp_numeric = pd.to_numeric(df[emp_col], errors='coerce')

if emp_numeric.isnull().all():
    emp_numeric = df[emp_col].map({
        "Yes": 1, "No": 0,
        "High": 1, "Low": 0,
        "Employable": 1, "Not Employable": 0
    })

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Descriptive Analytics",
    "Diagnostic Analytics",
    "Predictive Analytics",
    "Prescriptive Analytics"
])

# -----------------------
# OVERVIEW
# -----------------------
if page == "Overview":
    st.header("📊 Overview Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Avg Annual Salary", f"₹{int(df[salary_col].mean())}")
    col2.metric("🧠 Avg Skills", round(df[skills_col].mean(), 1))
    col3.metric("📈 Employability %", f"{round(emp_numeric.mean()*100,1)}%")

    st.markdown("---")

    st.markdown("### 📌 Key Insights")
    st.success("Higher skill levels strongly influence higher annual salary.")
    st.info("Experience contributes to growth but is less impactful than skills.")
    st.warning("Mid-skill candidates indicate a visible skill gap in the dataset.")

    st.plotly_chart(
        px.scatter(
            df,
            x=skills_col,
            y=salary_col,
            title="Skills vs Annual Salary (₹ per year)",
            labels={
                skills_col: "Number of Skills",
                salary_col: "Annual Salary (₹)"
            }
        ),
        use_container_width=True
    )

# -----------------------
# DESCRIPTIVE
# -----------------------
elif page == "Descriptive Analytics":
    st.header("📊 Descriptive Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.histogram(
                df,
                x=salary_col,
                nbins=20,
                title="Annual Salary Distribution (₹ per year)",
                labels={
                    salary_col: "Annual Salary (₹)",
                    "count": "Number of Candidates"
                }
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.histogram(
                df,
                x=skills_col,
                title="Skills Distribution",
                labels={
                    skills_col: "Number of Skills",
                    "count": "Number of Candidates"
                }
            ),
            use_container_width=True
        )

    emp_counts = emp_numeric.value_counts().rename(index={0: "Not Employable", 1: "Employable"})

    st.plotly_chart(
        px.bar(
            x=emp_counts.index,
            y=emp_counts.values,
            title="Employability Distribution",
            labels={"x": "Category", "y": "Number of Candidates"}
        ),
        use_container_width=True
    )

    st.markdown("### 📌 Insight")
    st.info("Distribution shows concentration in mid-skill levels, highlighting potential for targeted upskilling.")

# -----------------------
# DIAGNOSTIC
# -----------------------
elif page == "Diagnostic Analytics":
    st.header("🔍 Diagnostic Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.scatter(
                df,
                x=exp_col,
                y=salary_col,
                title="Experience vs Annual Salary",
                labels={
                    exp_col: "Years of Experience",
                    salary_col: "Annual Salary (₹)"
                }
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.scatter(
                df,
                x=skills_col,
                y=salary_col,
                title="Skills vs Annual Salary",
                labels={
                    skills_col: "Number of Skills",
                    salary_col: "Annual Salary (₹)"
                }
            ),
            use_container_width=True
        )

    st.markdown("### 📌 Insight")
    st.info("Skills demonstrate stronger correlation with salary than experience, indicating skill-driven growth.")

# -----------------------
# PREDICTIVE (ML)
# -----------------------
elif page == "Predictive Analytics":
    st.header("🤖 Predictive Analytics")

    X = df[[exp_col, skills_col]].copy()
    y_class = emp_numeric.copy()
    y_reg = df[salary_col].copy()

    X[exp_col] = pd.to_numeric(X[exp_col], errors='coerce')
    X[skills_col] = pd.to_numeric(X[skills_col], errors='coerce')
    y_reg = pd.to_numeric(y_reg, errors='coerce')

    ml_df = pd.concat([X, y_class, y_reg], axis=1).dropna()

    X = ml_df[[exp_col, skills_col]]
    y_class = ml_df[emp_col]
    y_reg = ml_df[salary_col]

    X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train_c)
    y_pred = clf.predict(X_test)

    st.subheader("📊 Classification Metrics")
    st.write("Accuracy:", round(accuracy_score(y_test_c, y_pred), 2))
    st.write("Precision:", round(precision_score(y_test_c, y_pred, zero_division=0), 2))
    st.write("Recall:", round(recall_score(y_test_c, y_pred, zero_division=0), 2))
    st.write("F1 Score:", round(f1_score(y_test_c, y_pred, zero_division=0), 2))

    reg = RandomForestRegressor()
    reg.fit(X_train, y_train_r)
    y_pred_r = reg.predict(X_test)

    st.subheader("📊 Regression Metrics")
    st.write("R² Score:", round(r2_score(y_test_r, y_pred_r), 2))

    st.markdown("### 📌 Insight")
    st.success("Predictive models confirm that increasing skills and experience improves employability and salary outcomes.")

    st.subheader("🔮 Try Prediction")

    exp = st.slider("Experience (Years)", 0, 10, 2)
    skills = st.slider("Number of Skills", 1, 10, 5)

    input_data = np.array([[exp, skills]])

    st.success(f"Predicted Employability: {'Yes' if clf.predict(input_data)[0]==1 else 'No'}")
    st.success(f"Predicted Annual Salary: ₹{int(reg.predict(input_data)[0])}")

# -----------------------
# PRESCRIPTIVE
# -----------------------
elif page == "Prescriptive Analytics":
    st.header("🎯 Prescriptive Analytics")

    skills = st.slider("Your Current Skills", 1, 10, 3)

    if skills < 4:
        st.warning("Recommendation: Learn Python, SQL")
    elif skills < 7:
        st.info("Recommendation: Learn Machine Learning, APIs")
    else:
        st.success("You are highly skilled! Focus on specialization.")

    if st.checkbox("Add Skill"):
        st.success("Projected salary increases with additional skills 🚀")

    st.markdown("### 📌 Insight")
    st.success("Upskilling directly improves employability and earning potential.")
