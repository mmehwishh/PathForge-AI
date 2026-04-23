import os

# Updated structure with safe path handling
project_structure = {
    # ML SERVICE
    "ml_service/requirements.txt": "fastapi\nuvicorn\npandas\nscikit-learn\nsentence-transformers\nnumpy\npython-dotenv",
    "ml_service/src/engine/preprocessing.py": """import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

    def encode_experience_level(self, df, column='experience_level'):
        # ORDINAL ENCODING: Order matters
        mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
        df[column] = df[column].map(mapping)
        return df

    def encode_categories(self, df, column='preferred_topic'):
        # ONE-HOT ENCODING: Nominal data (no order)
        encoded_data = self.one_hot_encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded_data, columns=self.one_hot_encoder.get_feature_names_out([column]))
        return pd.concat([df.drop(column, axis=1), encoded_df], axis=1)

    def scale_numerical_features(self, df, columns=['study_hours']):
        # SCALING: Normalize data ranges
        df[columns] = self.scaler.fit_transform(df[columns])
        return df
""",
    "ml_service/src/api/main.py": """from fastapi import FastAPI
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
""",

    # WEB API (ASP.NET CORE)
    "web_api/PathGenerator.Infrastructure/MLServiceClient.cs": """using System.Net.Http.Json;

namespace PathGenerator.Infrastructure {
    public class MLServiceClient {
        private readonly HttpClient _http;
        public MLServiceClient(HttpClient http) { _http = http; }

        public async Task<object> GetLearningPath(object userPrefs) {
            var response = await _http.PostAsJsonAsync("http://localhost:8000/predict", userPrefs);
            return await response.Content.ReadFromJsonAsync<object>();
        }
    }
}""",

    # FRONTEND (REACT)
    "frontend/src/components/PathVisualizer.jsx": """import React from 'react';

export default function PathVisualizer({ steps }) {
  if (!steps) return null;
  return (
    <div className="p-6 bg-white shadow-xl rounded-lg">
      <h2 className="text-xl font-bold mb-4 text-indigo-600">Your Learning Journey</h2>
      <div className="space-y-4">
        {steps.map((s, i) => (
          <div key={i} className="border-l-4 border-indigo-500 pl-4 py-2">
            <p className="font-semibold text-gray-800">{s.title}</p>
            <p className="text-sm text-gray-500">{s.duration}</p>
          </div>
        ))}
      </div>
    </div>
  );
}""",

    # ROOT FILES
    "README.md": """# Smart Personalized Learning Path Generator
1. Start ML Service: `uvicorn src.api.main:app --reload` (port 8000)
2. Start .NET API: `dotnet run` (port 5000)
3. Start Frontend: `npm run dev` (port 5173)
""",
    
    # Placeholder folders
    "ml_service/data/raw/.gitkeep": "",
    "ml_service/data/processed/.gitkeep": "",
    "ml_service/models/.gitkeep": "",
    "ml_service/notebooks/exploration.ipynb": "",
}

def create_project():
    print("🏗️ Creating Smart Learning Path Project Structure...")
    for path, content in project_structure.items():
        # Get the directory part of the path
        directory = os.path.dirname(path)
        
        # Only try to create the directory if the path actually contains one
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Write the file
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Created: {path}")
        except Exception as e:
            print(f"❌ Failed to create {path}: {e}")

    print("\n🚀 Project successfully generated!")

if __name__ == "__main__":
    create_project()