import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Student Performance & Career Predictor",
    layout="wide"
)

st.title("🎓 Student Performance & Career Recommendation System")

st.write(
    "Predict student performance and explore suitable career paths using Machine Learning."
)

performance_model = joblib.load(
    "models/performance_model.pkl"
)

career_model = joblib.load(
    "models/career_classifier.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

st.sidebar.header("Student Inputs")

st.sidebar.write(
    "Adjust the student traits and explore predictions."
)

study_hours = st.sidebar.slider(
    "Study Hours",
    1,
    10,
    5
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    4,
    10,
    7
)

attendance = st.sidebar.slider(
    "Attendance Percentage",
    50,
    100,
    75
)

assignments = st.sidebar.slider(
    "Assignments Completed",
    0,
    10,
    5
)

screen_time = st.sidebar.slider(
    "Screen Time",
    1,
    12,
    5
)

coding_interest = st.sidebar.slider(
    "Coding Interest",
    1,
    10,
    5
)

logical_reasoning = st.sidebar.slider(
    "Logical Reasoning",
    1,
    10,
    5
)

creativity = st.sidebar.slider(
    "Creativity Score",
    1,
    10,
    5
)

communication = st.sidebar.slider(
    "Communication Skill",
    1,
    10,
    5
)

stress = st.sidebar.slider(
    "Stress Level",
    1,
    10,
    5
)

input_data = pd.DataFrame({
    "study_hours": [study_hours],
    "sleep_hours": [sleep_hours],
    "attendance_percentage": [attendance],
    "assignments_completed": [assignments],
    "screen_time": [screen_time],
    "coding_interest": [coding_interest],
    "logical_reasoning": [logical_reasoning],
    "creativity_score": [creativity],
    "communication_skill": [communication],
    "stress_level": [stress]
})

# Performance Prediction
predicted_performance = performance_model.predict(
    input_data
)[0]

input_data["performance_score"] = (
    predicted_performance
)


# Performance Category
if predicted_performance >= 85:
    performance_category = "Excellent"

elif predicted_performance >= 70:
    performance_category = "Good"

elif predicted_performance >= 50:
    performance_category = "Average"

else:
    performance_category = "Needs Improvement"


# Career Prediction
career_prediction = career_model.predict(
    input_data
)

predicted_career = label_encoder.inverse_transform(
    career_prediction
)[0]


# Affinity Scores
career_scores = {
    "Data Science": (
        coding_interest * 0.35
        + logical_reasoning * 0.4
        + predicted_performance * 0.15
        + study_hours * 0.1
    ),

    "Backend Development": (
        coding_interest * 0.4
        + logical_reasoning * 0.35
        + assignments * 0.15
        + attendance * 0.1
    ),

    "Cybersecurity": (
        logical_reasoning * 0.4
        - stress * 0.2
        + attendance * 0.2
        + predicted_performance * 0.2
    ),

    "UI/UX Design": (
        creativity * 0.45
        + communication * 0.3
        + screen_time * 0.15
        + coding_interest * 0.1
    ),

    "Web Development": (
        coding_interest * 0.35
        + creativity * 0.3
        + communication * 0.15
        + screen_time * 0.2
    ),

    "Cloud Computing": (
        logical_reasoning * 0.35
        + attendance * 0.3
        + predicted_performance * 0.2
        - stress * 0.15
    ),

    "Mobile Development": (
        coding_interest * 0.35
        + creativity * 0.25
        + screen_time * 0.25
        + communication * 0.15
    ),

    "Game Development": (
        creativity * 0.4
        + coding_interest * 0.3
        + screen_time * 0.2
        + logical_reasoning * 0.1
    ),

    "DevOps Engineering": (
        logical_reasoning * 0.35
        + attendance * 0.25
        - stress * 0.15
        + assignments * 0.25
    ),

    "Software Testing": (
        logical_reasoning * 0.35
        + communication * 0.25
        + assignments * 0.2
        - stress * 0.1
        + coding_interest * 0.1
    )
}

# Normalise the Scores
max_score = max(career_scores.values())
min_score = min(career_scores.values())

for key in career_scores:

    career_scores[key] = (
        (
            career_scores[key] - min_score
        )
        /
        (
            max_score - min_score
        )
    ) * 100

career_df = pd.DataFrame({
    "Career": career_scores.keys(),
    "Affinity": career_scores.values()
})

career_df = career_df.sort_values(
    by="Affinity",
    ascending=False
)

# Display Results
st.header("📊 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Predicted Performance Score",
        f"{predicted_performance:.2f}"
    )

with col2:
    st.metric(
        "Performance Category",
        performance_category
    )

st.markdown("---")

if predicted_performance >= 85:

    st.success(
        "Excellent academic profile detected."
    )

elif predicted_performance >= 70:

    st.info(
        "Good academic performance with strong potential."
    )

elif predicted_performance >= 50:

    st.warning(
        "Average performance. Improvement possible with consistency."
    )

else:

    st.error(
        "Performance needs improvement."
    )

st.markdown("---")

st.subheader("💼 Recommended Career Path")

st.success(
    f"Recommended Career: {predicted_career}"
)

st.markdown("---")

st.subheader("🏆 Top Career Matches")

top_3 = career_df.head(3)

for i, row in top_3.iterrows():

    st.write(
        f"### {row['Career']} — {row['Affinity']:.1f}% Match"
    )
st.markdown("---")

st.subheader("📈 Career Affinity Scores")

st.bar_chart(
    career_df.set_index("Career")
)

st.markdown("---")

st.caption(
    "Built using Machine Learning, Streamlit, and Scikit-learn"
)