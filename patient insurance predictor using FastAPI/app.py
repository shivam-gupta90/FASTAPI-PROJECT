from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal, Annotated, Dict
import pickle
import pandas as pd

# ----------------- City Tier Info -----------------
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

# ----------------- Input Schema -----------------
class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120)]
    weight: Annotated[float, Field(..., gt=0)]
    height: Annotated[float, Field(..., gt=0, lt=6.5)]
    income_lpa: Annotated[float, Field(..., gt=0)]
    smoker: Annotated[bool, Field(...)]
    city: Annotated[str, Field(...)]
    occupation: Annotated[Literal[
        'retired', 'freelancer', 'student', 'government_job',
        'business_owner', 'unemployed', 'private_job'
    ], Field(...)]

    @field_validator('city')
    @classmethod
    def normalize_city(cls, v: str) -> str:
        return v.strip().title()

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        return 3

# ----------------- Response Schema -----------------
class PredictionResponse(BaseModel):
    predicted_category: str = Field(..., example="High")
    confidence: float = Field(..., example=0.8432)
    class_probabilities: Dict[str, float] = Field(..., example={"Low": 0.01, "Medium": 0.15, "High": 0.84})

# ----------------- Load ML Model -----------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

MODEL_VERSION = "1.0.0"
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):
    df = pd.DataFrame([user_input])
    predicted_class = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))
    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs
    }

# ----------------- FastAPI Setup -----------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API"}

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version": MODEL_VERSION,
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):
    try:
        user_input = {
            "bmi": data.bmi,
            "age_group": data.age_group,
            "lifestyle_risk": data.lifestyle_risk,
            "city_tier": data.city_tier,
            "income_lpa": data.income_lpa,
            "occupation": data.occupation
        }
        prediction = predict_output(user_input)
        return JSONResponse(status_code=200, content={"response": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




# run : python train_model.py  first and then run : uvicorn app:app --reload
#this is final code
