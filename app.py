import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("SkillSync_AI_Candidates.csv")

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

    col1.metric("Avg Salary", int(df["Salary"].mean()))
    col2.metric("Avg Skills", round(df["Skills_Count"].mean(), 1))
    col3.metric("Employability %", round(df["Employable"].mean()*100, 1))


# -----------------------
# DESCRIPTIVE
# -----------------------
elif page == "Descriptive Analytics":
    st.title("📊 Descriptive Analytics")

    st.subheader("Salary Distribution")
    fig = px.histogram(df, x="Salary", nbins=20)
    st.plotly_chart(fig)

    st.subheader("Skills Distribution")
    fig2 = px.histogram(df, x="Skills_Count")
    st.plotly_chart(fig2)

    st.subheader("Employability Count")
    fig3 = px.bar(df["Employable"].value_counts())
    st.plotly_chart(fig3)


# -----------------------
# DIAGNOSTIC
# -----------------------
elif page == "Diagnostic Analytics":
    st.title("🔍 Diagnostic Analytics")

    st.subheader("Salary vs Experience")
    fig = px.scatter(df, x="Experience", y="Salary")
    st.plotly_chart(fig)

    st.subheader("Skills vs Salary")
    fig2 = px.scatter(df, x="Skills_Count", y="Salary")
    st.plotly_chart(fig2)

    st.info("Insight: Higher skills and experience → higher salary.")


# -----------------------
# PREDICTIVE
# -----------------------
elif page == "Predictive Analytics":
    st.title("🤖 Predictive Analytics")

    st.subheader("Simple Salary Prediction")

    exp = st.slider("Experience", 0, 10, 2)
    skills = st.slider("Skills Count", 1, 10, 5)

    # simple model logic (safe)
    predicted_salary = 30000 + (exp * 5000) + (skills * 4000)

    st.success(f"Predicted Salary: ₹{predicted_salary}")


# -----------------------
# PRESCRIPTIVE
# -----------------------
elif page == "Prescriptive Analytics":
    st.title("🎯 Prescriptive Analytics")

    st.subheader("Skill Recommendation")

    skills = st.slider("Your Current Skills", 1, 10, 3)

    if skills < 4:
        st.warning("Recommendation: Learn Python, SQL")
    elif skills < 7:
        st.info("Recommendation: Learn Machine Learning, APIs")
    else:
        st.success("You are highly skilled! Focus on specialization.")

    st.subheader("What-if Analysis")

    add_skill = st.checkbox("Add 1 Skill")

    base_salary = 40000 + skills * 5000

    if add_skill:
        new_salary = base_salary + 5000
        st.write(f"New Salary: ₹{new_salary}")
    else:
        st.write(f"Current Salary: ₹{base_salary}")
