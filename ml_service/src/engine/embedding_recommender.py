import logging
import re
from math import ceil
from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


LEVEL_ORDER = {
    "beginner": 0,
    "beginner level": 0,
    "all levels": 0,
    "intermediate": 1,
    "intermediate level": 1,
    "advanced": 2,
    "advanced level": 2,
}

ROADMAP_STAGES = ("Foundations", "Core", "Building", "Advanced", "Deployment")

FOUNDATION_KEYWORDS = (
    "introduction", "intro", "foundation", "foundations", "basic", "basics",
    "beginner", "getting started", "part 1", "fundamentals",
    "html", "css", "javascript basics",
)

ADVANCED_KEYWORDS = (
    "advanced", "professional", "expert", "master", "capstone",
    "specialization", "deployment", "deploy", "portfolio", "production",
)

# Maps user-facing topic names → exact subject values in the CSV.
# These must match what preprocessing.py writes into the subject column.
TOPIC_ALIASES = {
    "web dev":                  "Web Development",
    "web development":          "Web Development",
    "data science":             "Data Science",
    "machine learning":         "Machine Learning",
    "ai":                       "Generative AI & LLMs",
    "generative ai":            "Generative AI & LLMs",
    "llms":                     "Generative AI & LLMs",
    "cyber security":           "Cybersecurity",
    "cybersecurity":            "Cybersecurity",
    "deep learning":            "Deep Learning",
    "nlp":                      "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "computer vision":          "Computer Vision",
    "databases":                "Databases & SQL",
    "sql":                      "Databases & SQL",
    "devops":                   "DevOps & Cloud Engineering",
    "cloud":                    "Cloud Infrastructure",
    "mobile":                   "Mobile Development",
    "mobile development":       "Mobile Development",
    "software engineering":     "Software Engineering",
    "blockchain":               "Blockchain & Web3",
    "game development":         "Game Development",
    "big data":                 "Big Data & Data Engineering",
    "data engineering":         "Big Data & Data Engineering",
    "mlops":                    "MLOps & AI Engineering",
    "data analytics":           "Data Analytics & Business Intelligence",
    "business intelligence":    "Data Analytics & Business Intelligence",
    "digital marketing":        "Digital Marketing",
    "finance":                  "Finance & Accounting",
    "project management":       "Project Management",
    "graphic design":           "Graphic Design",
    "photography":              "Photography",
    "healthcare":               "Healthcare & Medicine",
}

PHASE_KEYWORDS = {
    "Web Development": {
        "Foundations": (
            "html", "css", "javascript", "basic", "beginner",
            "foundation", "introduction", "getting started",
        ),
        "Building": (
            "react", "node", "express", "api", "project", "app",
            "frontend", "backend", "typescript", "framework",
        ),
        "Deployment": (
            "deploy", "hosting", "portfolio", "production",
            "cloud", "devops", "docker", "ci cd",
        ),
    },
    "Data Science": {
        "Foundations": (
            "introduction", "basic", "math", "statistics",
            "python", "foundation", "beginner",
        ),
        "Building": (
            "analysis", "pandas", "machine learning",
            "visualization", "project", "numpy",
        ),
        "Advanced": (
            "advanced", "capstone", "professional",
            "specialization", "deep learning",
        ),
    },
    "Machine Learning": {
        "Foundations": (
            "introduction", "basic", "foundation", "beginner", "supervised",
        ),
        "Building": (
            "scikit", "sklearn", "model", "classification",
            "regression", "clustering", "pipeline",
        ),
        "Advanced": (
            "advanced", "deep learning", "neural", "production", "mlops",
        ),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Set FORCE_REBUILD = True once after regenerating your CSV, then set back to
# False. This guarantees ChromaDB is rebuilt with fresh subject labels.
# ──────────────────────────────────────────────────────────────────────────────
FORCE_REBUILD = False


class UnsupportedTopicError(ValueError):
    """Raised when the dataset cannot support a requested roadmap topic."""

    def __init__(self, topic: str, available_topics: List[str]):
        self.topic = topic
        self.available_topics = available_topics
        super().__init__(
            f"Unsupported topic '{topic}'. Choose one of: {', '.join(available_topics)}"
        )


class InsufficientCoursesError(ValueError):
    """Raised when a topic exists but has too few courses for a reliable roadmap."""

    def __init__(self, topic: str, minimum_required: int, available_count: int):
        self.topic = topic
        self.minimum_required = minimum_required
        self.available_count = available_count
        super().__init__(
            f"Not enough courses for '{topic}'. Found {available_count}; "
            f"minimum required is {minimum_required}."
        )


class EmbeddingRecommender:
    """
    Embedding-based course recommender using ChromaDB and Sentence Transformers.
    """

    def __init__(
        self,
        courses_csv_path: str = "ml_service/data/processed/clean_courses.csv",
        db_path: str = "ml_service/models/chroma_db",
    ):
        # Resolve paths relative to the project root (3 levels up from this file)
        BASE_DIR         = Path(__file__).resolve().parents[3]
        courses_csv_path = BASE_DIR / courses_csv_path
        db_path          = BASE_DIR / db_path

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load course data
        self.courses_df = pd.read_csv(courses_csv_path)
        self.logger.info("Loaded %d courses from %s", len(self.courses_df), courses_csv_path)

        # ── DEBUG: print actual subject values so you can verify the filter works ──
        subjects_in_csv = sorted(self.courses_df["subject"].dropna().unique().tolist())
        self.logger.info("Subjects found in CSV:\n  %s", "\n  ".join(subjects_in_csv))

        # Precompute median duration for fallback
        self._median_duration = float(self.courses_df["content_duration"].median() or 5.0)

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # ChromaDB
        self.client     = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection("courses")

        if FORCE_REBUILD or self.collection.count() == 0 or self._collection_needs_rebuild():
            reason = (
                "FORCE_REBUILD=True"      if FORCE_REBUILD
                else "empty collection"   if self.collection.count() == 0
                else "metadata schema changed"
            )
            self.logger.info("Rebuilding ChromaDB collection (%s) …", reason)
            self.client.delete_collection("courses")
            self.collection = self.client.get_or_create_collection("courses")
            self._create_collection()
        else:
            self.logger.info(
                "Using existing ChromaDB collection (%d docs).", self.collection.count()
            )

    # ──────────────────────────────────────────────────────────────────────────
    # CHROMADB SETUP
    # ──────────────────────────────────────────────────────────────────────────

    def _create_collection(self) -> None:
        """Embed every course and store in ChromaDB."""
        self.logger.info("Creating embeddings for %d courses …", len(self.courses_df))

        for idx, row in self.courses_df.iterrows():
            search_text = (
                f"Title: {row['title']}\n"
                f"Subject: {row['subject']}\n"
                f"Level: {row['difficulty_level']}\n"
                f"Description: {row['description'] if pd.notna(row.get('description')) else ''}"
            )

            self.collection.add(
                documents=[search_text],
                metadatas=[{
                    "course_id":        int(idx),
                    "title":            str(row["title"]),
                    "subject":          str(row["subject"]),
                    "difficulty":       str(row["difficulty_level"]),
                    "rating":           self._to_float(row.get("rating")),
                    "students":         int(self._to_float(row.get("no_students"))),
                    "content_duration": self._to_float(row.get("content_duration")),
                }],
                ids=[f"course_{idx}"],
            )

            if idx % 500 == 0:
                self.logger.info("  Embedded %d / %d courses …", idx, len(self.courses_df))

        self.logger.info("ChromaDB collection ready (%d docs).", self.collection.count())

    def _collection_needs_rebuild(self) -> bool:
        sample    = self.collection.get(limit=1, include=["metadatas"])
        metadatas = sample.get("metadatas") or []
        if not metadatas:
            return True
        required = {"course_id", "subject", "difficulty", "content_duration"}
        return not required.issubset(metadatas[0].keys())

    # ──────────────────────────────────────────────────────────────────────────
    # STATIC HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_level(level: Optional[str]) -> str:
        return str(level or "").strip().lower()

    @staticmethod
    def _level_rank(level: Optional[str]) -> int:
        return LEVEL_ORDER.get(EmbeddingRecommender._normalize_level(level), 1)

    @staticmethod
    def _stage_for_course(user_level: str, course_level: str) -> str:
        course_rank = EmbeddingRecommender._level_rank(course_level)
        user_rank   = EmbeddingRecommender._level_rank(user_level)
        if course_rank < user_rank:
            return "Foundations"
        if course_rank == user_rank:
            return "Core"
        return "Advanced"

    @staticmethod
    def _metadata_duration(metadata: Dict[str, Any]) -> float:
        try:
            return float(metadata.get("content_duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            if pd.isna(value):
                return 0.0
        except Exception:
            pass
        match = re.search(r"\d+(\.\d+)?", str(value).replace(",", ""))
        return float(match.group()) if match else 0.0

    @staticmethod
    def _sequence_rank(title: str) -> int:
        t = str(title or "").lower()
        if any(kw in t for kw in FOUNDATION_KEYWORDS):
            return 0
        if any(kw in t for kw in ADVANCED_KEYWORDS):
            return 2
        return 1

    # ──────────────────────────────────────────────────────────────────────────
    # INSTANCE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _course_hours(self, metadata: Dict[str, Any]) -> float:
        hours = self._metadata_duration(metadata)
        if hours > 0:
            return hours
        # Fallback: look up from DataFrame by course_id
        try:
            idx = int(metadata.get("course_id", -1))
            if idx >= 0 and idx in self.courses_df.index:
                hours = self._to_float(self.courses_df.loc[idx].get("content_duration", 0))
                if hours > 0:
                    return hours
        except (TypeError, ValueError):
            pass
        return self._median_duration

    def _resolve_subject(self, topic: str) -> Optional[str]:
        """
        Map a user-supplied topic string to the exact subject label used in
        the CSV (and therefore in ChromaDB metadata).

        Priority:
        1. TOPIC_ALIASES lookup (fast, handles aliases like "web dev")
        2. Case-insensitive exact match against subjects present in the CSV
        3. Returns None if nothing matches → no where-filter applied
        """
        normalized = str(topic or "").strip().lower()

        # 1. Alias table
        if normalized in TOPIC_ALIASES:
            resolved = TOPIC_ALIASES[normalized]
            self.logger.debug("_resolve_subject: alias '%s' → '%s'", topic, resolved)
            return resolved

        # 2. Direct match against CSV subjects (case-insensitive)
        subjects = self.courses_df["subject"].dropna().unique()
        for subject in subjects:
            if str(subject).strip().lower() == normalized:
                self.logger.debug("_resolve_subject: CSV match '%s' → '%s'", topic, subject)
                return str(subject)

        # 3. No match — warn so the problem is visible in logs
        self.logger.warning(
            "_resolve_subject: no match for topic='%s'. "
            "Available subjects: %s",
            topic,
            sorted(str(s) for s in subjects),
        )
        return None

    def available_topics(self) -> List[str]:
        """Return user-selectable subjects available in the processed dataset."""
        return sorted(str(s) for s in self.courses_df["subject"].dropna().unique())

    def validate_topic_support(self, topic: str, minimum_courses: int = 3) -> str:
        """
        Resolve a topic and ensure enough matching courses exist before building a roadmap.
        This prevents ChromaDB from returning unrelated courses when the topic is unsupported.
        """
        subject = self._resolve_subject(topic)
        if subject is None:
            raise UnsupportedTopicError(topic, self.available_topics())

        matching_count = int((self.courses_df["subject"] == subject).sum())
        if matching_count < minimum_courses:
            raise InsufficientCoursesError(subject, minimum_courses, matching_count)

        return subject

    def _phase_for_course(self, topic: str, title: str, fallback_stage: str) -> str:
        subject        = self._resolve_subject(topic) or topic
        normalized     = str(title or "").lower()
        phase_rules    = PHASE_KEYWORDS.get(subject, {})

        for phase, keywords in phase_rules.items():
            if any(kw in normalized for kw in keywords):
                return phase

        # Sensible fallback when no keyword matches
        return "Building" if fallback_stage == "Core" else fallback_stage

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        topic: str,
        level: str,
        hours_per_week: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return top_k course recommendations for the given topic / level.
        """
        safe_hours     = max(int(hours_per_week or 1), 1)
        query          = f"{topic} course tutorial learning path"
        subject_filter = self.validate_topic_support(topic)

        # ── DEBUG: log what filter is being applied ───────────────────────────
        self.logger.info(
            "recommend() → topic='%s', subject_filter='%s', level='%s'",
            topic, subject_filter, level,
        )

        where_clause = {"subject": subject_filter}

        # Fetch a large candidate pool so post-filtering has room to work
        n_candidates = min(top_k * 10, self.collection.count())
        results = self.collection.query(
            query_texts=[query],
            n_results=n_candidates,
            where=where_clause,
        )

        # ── DEBUG: show raw hits ──────────────────────────────────────────────
        raw_titles = [m["title"] for m in results["metadatas"][0]]
        self.logger.info(
            "ChromaDB returned %d raw results. First 5: %s",
            len(raw_titles), raw_titles[:5],
        )

        recommendations = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):

            # Cosine distance in ChromaDB is in [0, 2]; convert to [0, 1] similarity
            similarity     = max(0.0, 1.0 - (distance / 2.0))
            content_hours  = self._course_hours(metadata)
            weekly_hours   = min(content_hours, safe_hours)

            subject_match  = (
                subject_filter is not None
                and str(metadata.get("subject", "")).strip() == subject_filter
            )
            level_gap      = abs(
                self._level_rank(metadata.get("difficulty"))
                - self._level_rank(level)
            )
            stage          = self._stage_for_course(level, metadata.get("difficulty"))
            sequence_rank  = self._sequence_rank(metadata.get("title", ""))

            match_score = (
                similarity
                + (0.3 if subject_match else -0.5)   # heavy penalty for wrong subject
                - (0.08 * level_gap)
                - (0.03 * sequence_rank)
            )

            recommendations.append({
                "title":                  metadata["title"],
                "subject":                metadata["subject"],
                "difficulty":             metadata["difficulty"],
                "rating":                 metadata.get("rating", 0.0),
                "similarity_score":       round(similarity, 3),
                "stage":                  stage,
                "sequence_rank":          sequence_rank,
                "estimated_total_hours":  round(content_hours, 1),
                "estimated_weekly_hours": round(weekly_hours, 1),
                "match_score":            round(match_score, 3),
            })

        # Sort: foundation-first, then by stage order, then by score & rating
        stage_order = {s: i for i, s in enumerate(ROADMAP_STAGES)}
        recommendations.sort(
            key=lambda x: (
                x["sequence_rank"],
                stage_order.get(x["stage"], 99),
                -x["match_score"],
                -float(x.get("rating", 0)),
            )
        )

        top = recommendations[:top_k]

        # ── DEBUG: show final picks ───────────────────────────────────────────
        self.logger.info(
            "Final %d recommendations: %s",
            len(top), [r["title"] for r in top],
        )

        return top

    def get_learning_path(
        self,
        topic: str,
        level: str,
        hours_per_week: int,
    ) -> Dict[str, Any]:
        """
        Build a week-by-week learning path from course recommendations.
        """
        safe_hours = max(int(hours_per_week or 1), 1)
        courses    = self.recommend(topic, level, safe_hours)

        learning_path: Dict[str, Any] = {
            "topic":                 topic,
            "level":                 level,
            "total_weeks":           0,
            "weekly_hours":          safe_hours,
            "phases":                [],
            "courses":               [],
            "total_estimated_hours": 0,
        }

        current_week = 1
        phase_names: List[str] = []

        for course in courses:
            estimated_hours = course["estimated_total_hours"]
            course_weeks    = max(ceil(estimated_hours / safe_hours), 1)
            start_week      = current_week
            end_week        = current_week + course_weeks - 1
            phase           = self._phase_for_course(topic, course["title"], course["stage"])

            learning_path["courses"].append({
                "week":            start_week,
                "start_week":      start_week,
                "end_week":        end_week,
                "phase":           phase,
                "title":           course["title"],
                "description":     f"{phase}: {course['title']}",
                "estimated_hours": estimated_hours,
                "estimated_weeks": course_weeks,
                "status":          "pending",
            })

            if phase not in phase_names:
                phase_names.append(phase)

            learning_path["total_estimated_hours"] += estimated_hours
            current_week = end_week + 1

        learning_path["total_weeks"] = max(current_week - 1, 1)
        learning_path["phases"]      = phase_names

        return learning_path


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    recommender = EmbeddingRecommender()

    print("\n" + "=" * 50)
    print("TESTING RECOMMENDER")
    print("=" * 50)

    print("\nUser: Data Science, Beginner, 10 hours/week")
    recommendations = recommender.recommend(
        topic="Data Science",
        level="Beginner",
        hours_per_week=10,
    )
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Subject: {rec['subject']} | Difficulty: {rec['difficulty']}")
        print(f"   Rating: {rec['rating']} | Match: {rec['match_score']}")

    print("\n" + "=" * 50)
    print("LEARNING PATH GENERATION")
    print("=" * 50)

    learning_path = recommender.get_learning_path(
        topic="Web Development",
        level="Beginner",
        hours_per_week=15,
    )

    print(f"\nYour {learning_path['level']} {learning_path['topic']} Learning Path")
    print(f"{learning_path['weekly_hours']} hours/week for {learning_path['total_weeks']} weeks")
    print(f"Total estimated hours: {learning_path['total_estimated_hours']}")

    for course in learning_path["courses"]:
        print(f"\n  Weeks {course['start_week']}-{course['end_week']}: {course['title']}")
        print(f"    Phase: {course['phase']} | Hours: {course['estimated_hours']}")
