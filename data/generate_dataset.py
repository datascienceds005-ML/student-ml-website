"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset of student academic records,
modeled after common real-world "student performance" datasets
(e.g. UCI Student Performance, Kaggle Student Study Habits datasets).

Run this once to produce data/student_data.csv. If you have your own
real-world CSV, just replace student_data.csv with the same column
names (or update src/data_preprocessing.py to match your columns).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

# ----- Core features -----
study_hours = np.round(np.random.normal(4.5, 2.2, N).clip(0, 12), 1)
attendance = np.round(np.random.normal(80, 12, N).clip(40, 100), 1)
previous_score = np.round(np.random.normal(65, 15, N).clip(20, 100), 1)
sleep_hours = np.round(np.random.normal(6.8, 1.4, N).clip(3, 10), 1)
extracurricular = np.random.choice(["Yes", "No"], N, p=[0.45, 0.55])
parental_education = np.random.choice(
    ["High School", "Bachelors", "Masters", "PhD"], N, p=[0.35, 0.35, 0.22, 0.08]
)
internet_access = np.random.choice(["Yes", "No"], N, p=[0.82, 0.18])
study_group = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
screen_time = np.round(np.random.normal(4.0, 1.8, N).clip(0, 10), 1)
tutoring = np.random.choice(["Yes", "No"], N, p=[0.25, 0.75])

# gender / age just for realism / demographic EDA
gender = np.random.choice(["Male", "Female"], N, p=[0.51, 0.49])
age = np.random.randint(15, 19, N)

# ----- Target: final exam score -----
# Built as a weighted combination + noise, so relationships are learnable
# but not perfectly deterministic (mirrors real-world messiness).
parent_edu_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
extracurricular_bonus = np.where(extracurricular == "Yes", 2.0, 0.0)
tutoring_bonus = np.where(tutoring == "Yes", 4.0, 0.0)
study_group_bonus = np.where(study_group == "Yes", 1.5, 0.0)
internet_bonus = np.where(internet_access == "Yes", 2.0, 0.0)

final_score = (
    0.55 * previous_score
    + 3.1 * study_hours
    + 0.28 * attendance
    + 1.1 * sleep_hours
    - 1.3 * screen_time
    + 1.8 * np.array([parent_edu_map[p] for p in parental_education])
    + extracurricular_bonus
    + tutoring_bonus
    + study_group_bonus
    + internet_bonus
    + np.random.normal(0, 6, N)  # random noise
)

final_score = np.round(final_score.clip(0, 100), 1)

df = pd.DataFrame({
    "student_id": [f"STU{1000+i}" for i in range(N)],
    "gender": gender,
    "age": age,
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "previous_exam_score": previous_score,
    "sleep_hours": sleep_hours,
    "screen_time_hours": screen_time,
    "extracurricular_activities": extracurricular,
    "parental_education": parental_education,
    "internet_access": internet_access,
    "study_group": study_group,
    "tutoring": tutoring,
    "final_exam_score": final_score,
})

# Inject a small amount of realistic missingness (real datasets are messy)
for col in ["attendance_percentage", "sleep_hours", "parental_education"]:
    idx = np.random.choice(df.index, size=int(0.03 * N), replace=False)
    df.loc[idx, col] = np.nan

out_path = "data/student_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.head())
