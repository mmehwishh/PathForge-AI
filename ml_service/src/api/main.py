from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserInput(BaseModel):
    experience_level: str
    preferred_topic: str
    study_hours: int

@app.post("/predict")
async def get_path(data: UserInput):
    return {
        "status": "success",
        "path": [
            {"step": 1, "title": "Introduction to " + data.preferred_topic, "duration": "2 hours"},
            {"step": 2, "title": "Advanced " + data.preferred_topic, "duration": "5 hours"}
        ]
    }
