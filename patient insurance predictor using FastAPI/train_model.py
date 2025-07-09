import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

# Sample data
data = pd.DataFrame({
    "bmi": [22.5, 30.1, 28.3, 35.0, 24.2, 31.5],
    "age_group": ["young", "adult", "middle_aged", "senior", "adult", "middle_aged"],
    "lifestyle_risk": ["low", "medium", "medium", "high", "low", "high"],
    "city_tier": [1, 2, 2, 1, 3, 3],
    "income_lpa": [6.5, 10.0, 8.2, 12.0, 7.5, 11.0],
    "occupation": ["student", "private_job", "business_owner", "retired", "unemployed", "government_job"],
    "premium_category": ["Low", "Medium", "Medium", "High", "Low", "High"]
})

X = data.drop("premium_category", axis=1)
y = data["premium_category"]

# Preprocessing
numeric_features = ["bmi", "income_lpa", "city_tier"]
categorical_features = ["age_group", "lifestyle_risk", "occupation"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ model.pkl saved successfully")
