import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train_and_save_model():
    """Generate synthetic salary data and train a Random Forest model."""

    np.random.seed(42)
    n = 1000

    job_titles = ["Data Scientist", "Data Analyst", "ML Engineer",
                  "Software Engineer", "Data Engineer", "AI Researcher"]
    educations = ["Bachelor's", "Master's", "PhD"]
    locations  = ["Bangalore", "Mumbai", "Hyderabad",
                  "Delhi", "Pune", "Chennai", "Remote"]
    sizes      = ["Startup (< 50)", "Mid-size (50-500)", "Large (500+)"]

    data = pd.DataFrame({
        "job_title"    : np.random.choice(job_titles, n),
        "experience"   : np.random.randint(0, 21, n),
        "education"    : np.random.choice(educations, n),
        "location"     : np.random.choice(locations, n),
        "skill_count"  : np.random.randint(1, 10, n),
        "company_size" : np.random.choice(sizes, n),
    })

    # Base salaries per role (INR)
    base = {
        "Data Scientist"  : 900000,
        "Data Analyst"    : 650000,
        "ML Engineer"     : 1100000,
        "Software Engineer": 800000,
        "Data Engineer"   : 950000,
        "AI Researcher"   : 1300000,
    }
    edu_bonus  = {"Bachelor's": 0, "Master's": 150000, "PhD": 350000}
    loc_bonus  = {"Bangalore": 120000, "Mumbai": 100000, "Hyderabad": 90000,
                  "Delhi": 80000, "Pune": 70000, "Chennai": 75000, "Remote": 50000}
    size_bonus = {"Startup (< 50)": 50000, "Mid-size (50-500)": 100000, "Large (500+)": 200000}

    data["salary"] = (
        data["job_title"].map(base)
        + data["experience"] * 55000
        + data["education"].map(edu_bonus)
        + data["location"].map(loc_bonus)
        + data["skill_count"] * 30000
        + data["company_size"].map(size_bonus)
        + np.random.normal(0, 50000, n)   # noise
    ).clip(300000, 5000000).round(-3)

    # Encode categorical columns
    encoders = {}
    for col, key in [("job_title","job"), ("education","edu"),
                     ("location","loc"), ("company_size","size")]:
        le = LabelEncoder()
        data[col + "_enc"] = le.fit_transform(data[col])
        encoders[key] = le

    features = ["job_title_enc", "experience", "education_enc",
                "location_enc", "skill_count", "company_size_enc"]
    X = data[features]
    y = data["salary"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train_and_save_model()
