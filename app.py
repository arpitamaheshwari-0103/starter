import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

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
# FIX EMPLOYABILITY COLUMN
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
st.sidebar.title("SkillSync AI")
page = st.sidebar.radio("Navigate", [
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
    st.title("⚡ SkillSync AI Dashboard")

    st.markdown("### Business Objective")
    st.write("Analyze skills, predict employability, and recommend career improvements.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Salary", int(df[salary_col].mean()))
    col2.metric("Avg Skills", round(df[skills_col].mean(), 1))
    col3.metric("Employability %", round(emp_numeric.mean()*100, 1))


# -----------------------
# DESCRIPTIVE
# -----------------------
elif page == "Descriptive Analytics":
    st.title("📊 Descriptive Analytics")

    st.subheader("Salary Distribution")
    fig = px.histogram(df, x=salary_col, nbins=20)
    st.plotly_chart(fig)

    st.subheader("Skills Distribution")
    fig2 = px.histogram(df, x=skills_col)
    st.plotly_chart(fig2)

    st.subheader("Employability Count")
    fig3 = px.bar(emp_numeric.value_counts())
    st.plotly_chart(fig3)


# -----------------------
# DIAGNOSTIC
# -----------------------
elif page == "Diagnostic Analytics":
    st.title("🔍 Diagnostic Analytics")

    st.subheader("Salary vs Experience")
    fig = px.scatter(df, x=exp_col, y=salary_col)
    st.plotly_chart(fig)

    st.subheader("Skills vs Salary")
    fig2 = px.scatter(df, x=skills_col, y=salary_col)
    st.plotly_chart(fig2)

    st.info("Insight: Higher skills and experience → higher salary.")


# -----------------------
# PREDICTIVE
# -----------------------
elif page == "Predictive Analytics":
    st.title("🤖 Predictive Analytics")

    # Prepare data
  # Prepare data safely
X = df[[exp_col, skills_col]].copy()
y_class = emp_numeric.copy()
y_reg = df[salary_col].copy()

# Convert to numeric
X[exp_col] = pd.to_numeric(X[exp_col], errors='coerce')
X[skills_col] = pd.to_numeric(X[skills_col], errors='coerce')
y_reg = pd.to_numeric(y_reg, errors='coerce')

# Combine into one dataframe for cleaning
ml_df = pd.concat([X, y_class, y_reg], axis=1)

# Drop missing values
ml_df = ml_df.dropna()

# Re-split clean data
X = ml_df[[exp_col, skills_col]]
y_class = ml_df[emp_col]
y_reg = ml_df[salary_col]

    # Train-test split
    X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Classification
    st.subheader("Employability Prediction (Classification)")

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train_c)

    y_pred = clf.predict(X_test)

    st.write("Accuracy:", round(accuracy_score(y_test_c, y_pred),2))
    st.write("Precision:", round(precision_score(y_test_c, y_pred, zero_division=0),2))
    st.write("Recall:", round(recall_score(y_test_c, y_pred, zero_division=0),2))
    st.write("F1 Score:", round(f1_score(y_test_c, y_pred, zero_division=0),2))

    # Regression
    st.subheader("Salary Prediction (Regression)")

    reg = RandomForestRegressor()
    reg.fit(X_train, y_train_r)

    y_pred_r = reg.predict(X_test)

    st.write("R² Score:", round(r2_score(y_test_r, y_pred_r),2))

    # User input
    st.subheader("Try Prediction")

    exp = st.slider("Experience", 0, 10, 2)
    skills = st.slider("Skills Count", 1, 10, 5)

    input_data = np.array([[exp, skills]])

    pred_emp = clf.predict(input_data)[0]
    pred_sal = reg.predict(input_data)[0]

    st.success(f"Predicted Employability: {'Yes' if pred_emp==1 else 'No'}")
    st.success(f"Predicted Salary: ₹{int(pred_sal)}")


# -----------------------
# PRESCRIPTIVE
# -----------------------
elif page == "Prescriptive Analytics":
    st.title("🎯 Prescriptive Analytics")

    skills = st.slider("Your Current Skills", 1, 10, 3)

    if skills < 4:
        st.warning("Recommendation: Learn Python, SQL")
    elif skills < 7:
        st.info("Recommendation: Learn Machine Learning, APIs")
    else:
        st.success("You are highly skilled! Focus on specialization.")

    add_skill = st.checkbox("Add 1 Skill")

    base_salary = int(df[salary_col].mean() + skills * 2000)

    if add_skill:
        new_salary = base_salary + 3000
        st.write(f"New Salary: ₹{new_salary}")
    else:
        st.write(f"Current Salary: ₹{base_salary}")
