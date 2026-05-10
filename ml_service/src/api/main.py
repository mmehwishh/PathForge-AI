import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src.engine.embedding_recommender import (
        EmbeddingRecommender,
        InsufficientCoursesError,
        UnsupportedTopicError,
    )
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    from ml_service.src.engine.embedding_recommender import (
        EmbeddingRecommender,
        InsufficientCoursesError,
        UnsupportedTopicError,
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Singleton recommender — initialized once at startup, reused for all requests.
# Using a module-level variable instead of @lru_cache avoids subtle issues with
# FastAPI's async context and hot-reloads.
# ──────────────────────────────────────────────────────────────────────────────
_recommender: Optional[EmbeddingRecommender] = None


def get_recommender() -> EmbeddingRecommender:
    global _recommender
    if _recommender is None:
        raise RuntimeError("Recommender not initialized. Did startup fail?")
    return _recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the recommender (and rebuild ChromaDB if needed) before accepting requests."""
    global _recommender
    logger.info("Starting up — initializing EmbeddingRecommender …")
    try:
        _recommender = EmbeddingRecommender()
        logger.info("EmbeddingRecommender ready.")
    except Exception:
        logger.exception("FATAL: Could not initialize EmbeddingRecommender.")
        raise
    yield
    # Shutdown — nothing to clean up for ChromaDB PersistentClient
    logger.info("Shutting down ML service.")


app = FastAPI(
    title="AirForge ML Service",
    description="Generates personalized learning roadmaps from course embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────────────

class UserInput(BaseModel):
    preferred_topic:  str = Field(..., min_length=2, examples=["Web Development"])
    experience_level: str = Field(..., min_length=2, examples=["Beginner"])
    study_hours:      int = Field(..., ge=1, le=80,  examples=[10])


class RecommendationInput(UserInput):
    top_k: int = Field(default=5, ge=1, le=20, examples=[5])


class CourseStep(BaseModel):
    week:             int
    start_week:       int
    end_week:         int
    phase:            str
    title:            str
    description:      str
    estimated_hours:  float
    estimated_weeks:  int
    status:           str


class LearningPath(BaseModel):
    topic:                  str
    level:                  str
    total_weeks:            int
    weekly_hours:           int
    phases:                 List[str]
    courses:                List[CourseStep]
    total_estimated_hours:  float


class LearningPathResponse(BaseModel):
    status:        str
    learning_path: LearningPath


class RecommendationResponse(BaseModel):
    status:          str
    recommendations: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Quick liveness check — also confirms the recommender loaded correctly."""
    recommender = get_recommender()
    course_count = len(recommender.courses_df)
    return {
        "status":       "healthy",
        "service":      "ml_service",
        "courses_loaded": str(course_count),
    }


@app.post("/predict", response_model=LearningPathResponse)
async def get_learning_path(data: UserInput) -> LearningPathResponse:
    """Generate a week-by-week learning path for a given topic and level."""
    try:
        recommender   = get_recommender()
        learning_path = recommender.get_learning_path(
            topic=data.preferred_topic.strip(),
            level=data.experience_level.strip(),
            hours_per_week=data.study_hours,
        )
    except UnsupportedTopicError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_topic",
                "message": (
                    f"We do not have enough data to build a reliable "
                    f"{exc.topic} roadmap yet."
                ),
                "available_topics": exc.available_topics,
            },
        ) from exc
    except InsufficientCoursesError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "insufficient_courses",
                "message": (
                    f"We found only {exc.available_count} courses for {exc.topic}. "
                    f"At least {exc.minimum_required} are required for a reliable roadmap."
                ),
                "available_topics": recommender.available_topics(),
            },
        ) from exc
    except Exception as exc:
        logger.exception("Failed to generate learning path")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate learning path: {exc}",
        ) from exc

    if not learning_path.get("courses"):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No courses found for topic='{data.preferred_topic}' "
                f"level='{data.experience_level}'. "
                "Check that the topic matches a known subject (e.g. 'Web Development')."
            ),
        )

    return LearningPathResponse(status="success", learning_path=learning_path)


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(data: RecommendationInput) -> RecommendationResponse:
    """Return a ranked list of individual course recommendations."""
    try:
        recommender     = get_recommender()
        recommendations = recommender.recommend(
            topic=data.preferred_topic.strip(),
            level=data.experience_level.strip(),
            hours_per_week=data.study_hours,
            top_k=data.top_k,
        )
    except UnsupportedTopicError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_topic",
                "message": (
                    f"We do not have enough data to recommend {exc.topic} courses yet."
                ),
                "available_topics": exc.available_topics,
            },
        ) from exc
    except InsufficientCoursesError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "insufficient_courses",
                "message": (
                    f"We found only {exc.available_count} courses for {exc.topic}. "
                    f"At least {exc.minimum_required} are required."
                ),
                "available_topics": recommender.available_topics(),
            },
        ) from exc
    except Exception as exc:
        logger.exception("Failed to generate recommendations")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {exc}",
        ) from exc

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No recommendations found for topic='{data.preferred_topic}' "
                f"level='{data.experience_level}'."
            ),
        )

    return RecommendationResponse(status="success", recommendations=recommendations)
