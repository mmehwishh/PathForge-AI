# AI Learning Path Recommender

An intelligent multi-service learning roadmap generator that creates personalized study plans using **semantic search**, **vector embeddings**, **ChromaDB**, and **AI-powered topic matching**.

The system recommends structured weekly learning roadmaps based on a user's:

- Topic / Subject
- Skill Level
- Weekly Study Hours
- Learning Duration

Instead of returning random courses, the system uses **semantic similarity search** and **strict topic validation** to generate more reliable and professional learning paths.

---

# Project Architecture

This project follows a **3-tier architecture**:

```text
Frontend (React + Vite)
        ↓
ASP.NET Core Web API
        ↓
Python ML Service (FastAPI + ChromaDB)
```

---

# Technologies Used

## Frontend
- React.js
- Vite
- Axios
- CSS

## Backend API
- ASP.NET Core Web API (.NET)

## Machine Learning Service
- Python
- FastAPI
- SentenceTransformers
- ChromaDB
- Pandas
- NumPy
- Scikit-learn

## AI / NLP
- Semantic Search
- Vector Embeddings
- Topic Resolution
- AI-assisted Subject Classification

---

# Core Features

## Personalized Learning Roadmaps
Generates weekly learning plans based on:
- User topic
- Skill level
- Weekly study hours
- Duration

---

## Semantic Course Search
Uses SentenceTransformer embeddings to:
- Understand meaning instead of exact keywords
- Find relevant learning resources semantically

---

## ChromaDB Vector Database
Stores vector embeddings of courses for:
- Fast retrieval
- Semantic similarity search
- Intelligent recommendations

---

## Topic Validation System
Professional fallback handling:
- Unsupported topics are blocked
- No fake/random roadmap generation
- Users receive supported topic suggestions

Example:

```json
{
  "status": "unsupported_topic",
  "message": "We do not have enough courses for this topic.",
  "available_topics": [
    "Machine Learning",
    "Web Development",
    "Data Science"
  ]
}
```

---

## Available Topics Section
Frontend automatically loads supported roadmap topics from backend and displays them as clickable chips/tabs.

---

## Multi-Service Communication
The application demonstrates real-world microservice communication:
- React → .NET API
- .NET API → Python ML Service

---

# Project Structure

```text
ai_assistant/
│
├── frontend/                # React + Vite Frontend
│
├── web_api/                 # ASP.NET Core API
│
├── ml_service/              # Python ML Service
│   ├── data/
│   ├── src/
│   ├── chroma_db/
│   └── requirements.txt
│
└── README.md
```

---

# How the System Works

## Step 1 — User Input
User enters:
- Topic
- Skill level
- Weekly hours
- Duration

Example:

```text
Topic: Machine Learning
Level: Beginner
Hours/Week: 10
Duration: 6 Weeks
```

---

## Step 2 — Frontend Sends Request
React frontend sends request to ASP.NET Core API.

---

## Step 3 — .NET API Processing
The API:
- Validates request
- Forwards request to Python ML service

---

## Step 4 — ML Service Processing
Python service:
- Resolves topic aliases
- Searches ChromaDB
- Retrieves semantically similar courses
- Builds structured roadmap

---

## Step 5 — Response Returned
Frontend visualizes:
- Weekly roadmap
- Course sequence
- Estimated study hours
- Progress status

---

# Semantic Search Flow

```text
User Query
   ↓
SentenceTransformer Embedding
   ↓
ChromaDB Similarity Search
   ↓
Relevant Courses Retrieved
   ↓
Roadmap Generation
```

---

# Machine Learning Concepts Used

## Sentence Embeddings

Courses and user queries are converted into vectors using:

```python
SentenceTransformer
```

This allows:
- Meaning-based retrieval
- Better recommendations
- NLP-powered matching

---

## Vector Similarity Search

ChromaDB compares vector similarity to find the most relevant courses.

---

## Topic Resolution

The system maps:

```text
AI → Artificial Intelligence
ML → Machine Learning
```

and validates supported topics before generating a roadmap.

---

# API Endpoints

## Python ML Service

### Health Check

```http
GET /health
```

### Available Topics

```http
GET /topics
```

### Generate Roadmap

```http
POST /generate-roadmap
```

---

## ASP.NET Core API

### Health Check

```http
GET /health
```

### Available Topics

```http
GET /api/learning-path/topics
```

### Generate Learning Path

```http
POST /api/learning-path/generate
```

---

# Installation Guide

# 1. Clone Repository

```bash
git clone https://github.com/your-username/ai-learning-path-recommender.git
cd ai-learning-path-recommender
```

---

# 2. Setup Python ML Service

## Navigate to ML Service

```bash
cd ml_service
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run ML Service

```bash
uvicorn src.api.main:app --reload
```

Runs on:

```text
http://127.0.0.1:8000
```

---

# 3. Setup ASP.NET Core API

## Navigate to API

```bash
cd web_api
```

## Restore Packages

```bash
dotnet restore
```

## Run API

```bash
dotnet run --urls http://localhost:5000
```

Runs on:

```text
http://localhost:5000
```

---

# 4. Setup Frontend

## Navigate to Frontend

```bash
cd frontend
```

## Install Dependencies

```bash
npm install
```

## Run Frontend

```bash
npm run dev
```

OR

```bash
npm run build
npm run preview
```

Frontend URL:

```text
http://127.0.0.1:5173
```

---

# Running Full Application

Start services in this order:

## Terminal 1 — ML Service

```bash
cd ml_service
uvicorn src.api.main:app --reload
```

---

## Terminal 2 — ASP.NET API

```bash
cd web_api
dotnet run --urls http://localhost:5000
```

---

## Terminal 3 — Frontend

```bash
cd frontend
npm run build
npm run preview
```

---

# Example Workflow

```text
User selects "Machine Learning"
        ↓
Frontend sends request
        ↓
.NET API receives request
        ↓
Python ML service processes query
        ↓
ChromaDB finds relevant courses
        ↓
Roadmap generated
        ↓
Frontend visualizes roadmap
```

---

# Professional Design Decisions

## Strict Unsupported Topic Handling

The system does NOT generate fake/random roadmaps.

If insufficient topic data exists:
- Roadmap generation stops
- User receives supported topic suggestions

This prevents low-quality recommendations.

---

## Semantic Recommendation System

Instead of keyword matching:
- Uses embeddings
- Understands context and meaning
- Produces more relevant recommendations

---

## Microservice Architecture

Separation of concerns:
- Frontend handles UI
- .NET handles API orchestration
- Python handles ML logic

This architecture is scalable and production-oriented.

---

# Current Limitations

- Roadmap quality depends on dataset quality
- Some topics may have limited course coverage
- ChromaDB is currently local storage
- No authentication system yet

---

# Future Improvements

- User authentication
- Progress tracking
- Roadmap export (PDF)
- AI chatbot mentor
- Course difficulty prediction
- Recommendation feedback loop
- Docker support
- Real-time analytics

---

# Python Requirements (`requirements.txt`)

```txt
fastapi
uvicorn
pandas
numpy
scikit-learn
sentence-transformers
chromadb
pydantic
python-dotenv
anthropic
requests
```

---

# Frontend Dependencies

```bash
npm install react react-dom axios
npm install -D vite
```

---

# Learning Outcomes

This project demonstrates:
- NLP applications
- Semantic search systems
- Vector databases
- Recommendation systems
- Microservice communication
- API integration
- React frontend development
- ML service architecture

---

# Author

Developed as an AI-powered educational roadmap generation system using:
- React
- ASP.NET Core
- FastAPI
- ChromaDB
- SentenceTransformers