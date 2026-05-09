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
    "all levels": 1,
    "intermediate": 1,
    "intermediate level": 1,
    "advanced": 2,
    "advanced level": 2,
}


ROADMAP_STAGES = ("Foundations", "Core", "Advanced")

FOUNDATION_KEYWORDS = (
    "introduction",
    "intro",
    "foundation",
    "foundations",
    "basic",
    "basics",
    "beginner",
    "getting started",
    "part 1",
    "fundamentals",
    "html",
    "css",
    "javascript basics",
)

ADVANCED_KEYWORDS = (
    "advanced",
    "professional",
    "expert",
    "master",
    "capstone",
    "specialization",
    "deployment",
    "deploy",
    "portfolio",
    "production",
)

TOPIC_ALIASES = {
    "web dev": "Web Development",
    "web development": "Web Development",
    "data science": "Data Science",
    "machine learning": "Machine Learning",
    "ai": "Generative AI & LLMs",
    "generative ai": "Generative AI & LLMs",
    "cyber security": "Cybersecurity",
    "cybersecurity": "Cybersecurity",
}

PHASE_KEYWORDS = {
    "Web Development": {
        "Foundations": ("html", "css", "javascript", "basic", "beginner", "foundation", "introduction"),
        "Building": ("react", "node", "express", "api", "project", "app", "frontend", "backend"),
        "Deployment": ("deploy", "hosting", "portfolio", "production", "cloud", "devops"),
    },
    "Data Science": {
        "Foundations": ("introduction", "basic", "math", "statistics", "python", "foundation"),
        "Building": ("analysis", "pandas", "machine learning", "visualization", "project"),
        "Advanced": ("advanced", "capstone", "professional", "specialization", "deep learning"),
    },
}


class EmbeddingRecommender:
    """
    A pure embedding-based course recommender system using ChromaDB and Sentence Transformers.

    This class provides course recommendations based on semantic similarity without requiring
    traditional machine learning training. It uses pre-trained embeddings to match user
    preferences with course content.

    Attributes:
        courses_df (pd.DataFrame): DataFrame containing course data.
        model (SentenceTransformer): Pre-trained sentence transformer model.
        client (chromadb.PersistentClient): ChromaDB client for vector storage.
        collection (chromadb.Collection): ChromaDB collection for course embeddings.
    """

    def __init__(self, courses_csv_path: str = 'ml_service/data/processed/clean_courses.csv', db_path: str = 'ml_service/models/chroma_db'):
        """
        Initialize the EmbeddingRecommender.

        Args:
            courses_csv_path (str): Path to the cleaned courses CSV file.
            db_path (str): Path to the ChromaDB persistent storage directory.
        """
        #resolve path
        BASE_DIR = Path(__file__).resolve().parents[3]
        courses_csv_path = BASE_DIR / courses_csv_path
        db_path = BASE_DIR / db_path

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load course data
        self.courses_df = pd.read_csv(courses_csv_path)
        self.logger.info(f"Loaded {len(self.courses_df)} courses from {courses_csv_path}")

        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Setup ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection("courses")
        if self.collection.count() == 0:
            self._create_collection()
        elif self._collection_needs_rebuild():
            self.logger.info("Rebuilding ChromaDB collection with updated metadata")
            self.client.delete_collection("courses")
            self.collection = self.client.get_or_create_collection("courses")
            self._create_collection()
        else:
            self.logger.info("Loaded existing ChromaDB collection")
    
    def _create_collection(self) -> None:
        """
        Create embeddings for all courses and store them in ChromaDB.

        This method should be called once to initialize the vector database.
        """
        self.logger.info("Creating embeddings for all courses...")

        for idx, row in self.courses_df.iterrows():
            # Create rich text for embedding
            search_text = f"""
            Title: {row['title']}
            Subject: {row['subject']}
            Level: {row['difficulty_level']}
            Description: {row['description'] if pd.notna(row['description']) else ''}
            """

            # Add to ChromaDB (auto-creates embedding)
            self.collection.add(
                documents=[search_text],
                metadatas=[{
                    'course_id': idx,
                    'title': row['title'],
                    'subject': row['subject'],
                    'difficulty': row['difficulty_level'],
                    'rating': self._to_float(row.get('rating')),
                    'students': int(self._to_float(row.get('no_students'))),
                    'content_duration': self._to_float(row.get('content_duration')),
                }],
                ids=[f"course_{idx}"]
            )

            if idx % 500 == 0:
                self.logger.info(f"Processed {idx}/{len(self.courses_df)} courses")

        self.logger.info(f"Created embeddings for {len(self.courses_df)} courses")

    def _collection_needs_rebuild(self) -> bool:
        sample = self.collection.get(limit=1, include=["metadatas"])
        metadatas = sample.get("metadatas") or []
        if not metadatas:
            return True

        required_fields = {"course_id", "subject", "difficulty", "content_duration"}
        return not required_fields.issubset(metadatas[0].keys())

    @staticmethod
    def _normalize_level(level: Optional[str]) -> str:
        return str(level or "").strip().lower()

    @staticmethod
    def _level_rank(level: Optional[str]) -> int:
        return LEVEL_ORDER.get(EmbeddingRecommender._normalize_level(level), 1)

    @staticmethod
    def _stage_for_course(user_level: str, course_level: str) -> str:
        course_rank = EmbeddingRecommender._level_rank(course_level)
        user_rank = EmbeddingRecommender._level_rank(user_level)

        if course_rank < user_rank:
            return "Foundations"
        if course_rank == user_rank:
            return "Core"
        return "Advanced"

    @staticmethod
    def _metadata_duration(metadata: Dict[str, Any]) -> float:
        duration = metadata.get("content_duration", 0.0)
        try:
            return float(duration)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_float(value: Any) -> float:
        if pd.isna(value):
            return 0.0

        match = re.search(r"\d+(\.\d+)?", str(value).replace(",", ""))
        return float(match.group()) if match else 0.0

    def _course_hours(self, metadata: Dict[str, Any]) -> float:
        metadata_hours = self._metadata_duration(metadata)
        if metadata_hours > 0:
            return metadata_hours

        course_id = metadata.get("course_id")
        try:
            course_idx = int(course_id)
        except (TypeError, ValueError):
            return 0.0

        if course_idx in self.courses_df.index:
            return self._to_float(self.courses_df.loc[course_idx].get("content_duration", 0))

        return 0.0

    @staticmethod
    def _sequence_rank(title: str) -> int:
        normalized_title = str(title or "").lower()
        if any(keyword in normalized_title for keyword in FOUNDATION_KEYWORDS):
            return 0
        if any(keyword in normalized_title for keyword in ADVANCED_KEYWORDS):
            return 2
        return 1

    def _phase_for_course(self, topic: str, title: str, fallback_stage: str) -> str:
        subject = self._resolve_subject(topic) or topic
        normalized_title = str(title or "").lower()
        phase_rules = PHASE_KEYWORDS.get(subject, {})

        for phase, keywords in phase_rules.items():
            if any(keyword in normalized_title for keyword in keywords):
                return phase

        if fallback_stage == "Core":
            return "Building"
        return fallback_stage

    def _resolve_subject(self, topic: str) -> Optional[str]:
        normalized_topic = str(topic or "").strip().lower()
        if normalized_topic in TOPIC_ALIASES:
            return TOPIC_ALIASES[normalized_topic]

        subjects = self.courses_df["subject"].dropna().unique()
        for subject in subjects:
            if str(subject).strip().lower() == normalized_topic:
                return str(subject)

        return None
    
    def recommend(self, topic: str, level: str, hours_per_week: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Generate course recommendations based on user preferences.

        Args:
            topic (str): Desired topic, e.g., "Data Science", "Web Development".
            level (str): User's skill level, e.g., "Beginner", "Intermediate", "Advanced".
            hours_per_week (int): Available study hours per week.
            top_k (int): Number of recommendations to return.

        Returns:
            List[Dict[str, Any]]: List of recommended courses with metadata and scores.
        """
        safe_hours = max(int(hours_per_week or 1), 1)

        # Keep availability out of the semantic query. Embeddings should focus on the topic,
        # not on matching words like "10" or "hours" inside course descriptions.
        query = f"{topic} course tutorial learning path"
        subject_filter = self._resolve_subject(topic)

        # Search ChromaDB using embeddings
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 8,
            where={"subject": subject_filter} if subject_filter else None,
        )

        # Process and rank results
        recommendations = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Convert distance to similarity score (lower distance = more similar)
            similarity = 1 - distance
            content_hours = self._course_hours(metadata) or safe_hours
            weekly_hours = min(content_hours, safe_hours)

            subject_match = subject_filter and str(metadata.get("subject", "")).lower() == subject_filter.lower()
            level_gap = abs(self._level_rank(metadata.get("difficulty")) - self._level_rank(level))
            stage = self._stage_for_course(level, metadata.get("difficulty"))
            sequence_rank = self._sequence_rank(metadata.get("title"))
            match_score = similarity + (0.2 if subject_match else 0) - (0.08 * level_gap) - (0.03 * sequence_rank)

            recommendations.append({
                'title': metadata['title'],
                'subject': metadata['subject'],
                'difficulty': metadata['difficulty'],
                'rating': metadata['rating'],
                'similarity_score': round(similarity, 3),
                'stage': stage,
                'sequence_rank': sequence_rank,
                'estimated_total_hours': round(content_hours, 1),
                'estimated_weekly_hours': round(weekly_hours, 1),
                'match_score': round(match_score, 3)
            })

        recommendations.sort(
            key=lambda x: (
                x['sequence_rank'],
                ROADMAP_STAGES.index(x['stage']),
                -x['match_score'],
                -float(x.get('rating', 0)),
            )
        )

        return recommendations[:top_k]
    
    def get_learning_path(self, topic: str, level: str, hours_per_week: int) -> Dict[str, Any]:
        """
        Create a structured learning path based on recommendations.

        Args:
            topic (str): Desired topic.
            level (str): User's skill level.
            hours_per_week (int): Available study hours per week.

        Returns:
            Dict[str, Any]: Learning path structure with courses organized by weeks.
        """
        safe_hours = max(int(hours_per_week or 1), 1)
        courses = self.recommend(topic, level, safe_hours)

        # Create learning path structure
        learning_path = {
            'topic': topic,
            'level': level,
            'total_weeks': 0,
            'weekly_hours': safe_hours,
            'phases': [],
            'courses': [],
            'total_estimated_hours': 0
        }

        current_week = 1
        phase_names = []

        for i, course in enumerate(courses):
            estimated_hours = course['estimated_total_hours']
            course_weeks = max(ceil(estimated_hours / safe_hours), 1)
            start_week = current_week
            end_week = current_week + course_weeks - 1
            phase = self._phase_for_course(topic, course['title'], course['stage'])

            learning_path['courses'].append({
                'week': start_week,
                'start_week': start_week,
                'end_week': end_week,
                'phase': phase,
                'title': course['title'],
                'description': f"{phase}: {course['title']}",
                'estimated_hours': estimated_hours,
                'estimated_weeks': course_weeks,
                'status': 'pending'
            })

            if phase not in phase_names:
                phase_names.append(phase)

            learning_path['total_estimated_hours'] += estimated_hours
            current_week = end_week + 1

        learning_path['total_weeks'] = max(current_week - 1, 1)
        learning_path['phases'] = phase_names

        return learning_path

# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Initialize recommender (creates embeddings on first run)
    recommender = EmbeddingRecommender()

    # Test recommendation
    print("\n" + "="*50)
    print("TESTING RECOMMENDER")
    print("="*50)

    # Example 1: Data Science for Beginners
    print("\nUser: Data Science, Beginner, 10 hours/week")
    recommendations = recommender.recommend(
        topic="Data Science",
        level="Beginner",
        hours_per_week=10
    )

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Subject: {rec['subject']} | Difficulty: {rec['difficulty']}")
        print(f"   Rating: {rec['rating']} | Match: {rec['match_score']}")

    # Example 2: Get full learning path
    print("\n" + "="*50)
    print("LEARNING PATH GENERATION")
    print("="*50)

    learning_path = recommender.get_learning_path(
        topic="Web Development",
        level="Beginner",
        hours_per_week=15
    )

    print(f"\nYour {learning_path['level']} {learning_path['topic']} Learning Path")
    print(f"{learning_path['weekly_hours']} hours/week for {learning_path['total_weeks']} weeks")
    print(f"Total estimated hours: {learning_path['total_estimated_hours']}")

    for course in learning_path['courses']:
        print(f"\n  Weeks {course['start_week']}-{course['end_week']}: {course['title']}")
        print(f"    Phase: {course['phase']} | Hours: {course['estimated_hours']}")
