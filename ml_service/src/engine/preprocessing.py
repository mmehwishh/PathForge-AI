"""
preprocessing.py
================
AirForge ML Service – Data Preprocessing Pipeline

Loads raw course data from Coursera, EdX, and Udemy, cleans and
standardises it into a single DataFrame, detects each course's subject,
fills missing values, and saves the result as a CSV ready for model training.

Author : Mehwish
Project: AirForge Learning-Path Recommender

CHANGES (v2)
------------
- _get_subject() replaced with _score_subject(): scores ALL subjects by keyword
  hit-count and picks the highest, eliminating first-match bias.
- Description is now used as a secondary signal when the title score is low.
- Low-confidence rows (score < MIN_CONFIDENCE_SCORE) are sent to the Claude API
  for accurate labelling in a single batch call.
- Difficulty levels are normalised to a consistent set of values.
- Duplicate courses (same title + source) are dropped after merging.
- Added _validate_output() to log a quality report before saving.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import logging
import time

import numpy as np
import pandas as pd
import anthropic

# ──────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR            = "ml_service/data/raw"
BASE_DIR_PROCESSED  = "ml_service/data/processed"
INPUT_FILES = {
    "coursera": os.path.join(BASE_DIR, "coursea_data.csv"),
    "edx":      os.path.join(BASE_DIR, "EdX.csv"),
    "udemy":    os.path.join(BASE_DIR, "udemy_courses.csv"),
}
OUTPUT_FILE = os.path.join(BASE_DIR_PROCESSED, "clean_courses.csv")

# Columns to drop per source (before renaming)
COLUMNS_TO_DROP = {
    "coursera": ["Unnamed: 0", "course_organization"],
    "edx":      ["University"],
    "udemy":    ["course_id", "is_paid", "price", "num_reviews",
                 "num_lectures", "published_timestamp"],
}

# Numeric columns on which we impute missing values
NUMERIC_COLS     = ["rating", "no_students", "content_duration"]
CATEGORICAL_COLS = ["certificate_type"]

# Subject classification tuning
# A course whose best keyword score is below this threshold is sent to Claude.
MIN_CONFIDENCE_SCORE = 2

# Batch size for Claude API calls (stay well within token limits)
CLAUDE_BATCH_SIZE = 50

# Valid difficulty levels after normalisation
DIFFICULTY_MAP = {
    # Beginner variants
    "beginner":           "Beginner",
    "beginner level":     "Beginner",
    "introductory":       "Beginner",
    "introduction":       "Beginner",
    "easy":               "Beginner",
    "all levels":         "Beginner",
    "all":                "Beginner",
    # Intermediate variants
    "intermediate":       "Intermediate",
    "intermediate level": "Intermediate",
    "medium":             "Intermediate",
    # Advanced variants
    "advanced":           "Advanced",
    "advanced level":     "Advanced",
    "expert":             "Advanced",
    "professional":       "Advanced",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 3. SUBJECT-KEYWORD TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────
# Each key is a human-readable subject label.
# Each value is a list of lowercase keywords that signal that subject.
# _score_subject() counts hits across ALL subjects and picks the winner.

SUBJECT_KEYWORDS: dict[str, list[str]] = {

    # ── Technology & Development ──────────────────────────────────────────────

    "Web Development": [
        "frontend", "backend", "fullstack", "full-stack", "react", "angular",
        "vue", "svelte", "html", "html5", "css", "css3", "javascript", "js",
        "typescript", "ts", "node", "nodejs", "django", "flask", "fastapi",
        "php", "laravel", "symfony", "ruby", "rails", "ruby on rails",
        "express", "expressjs", "next.js", "nextjs", "nuxt", "gatsby",
        "remix", "astro", "webpack", "vite", "babel", "sass", "scss",
        "tailwind", "bootstrap", "materialui", "rest api", "graphql",
        "websocket", "oauth", "jwt", "http", "https", "web api",
        "responsive design", "pwa", "progressive web app",
        "single page application", "spa", "ssr", "server side rendering",
        "static site", "jamstack", "cms", "wordpress", "web scraping",
        "selenium", "puppeteer", "playwright", "cheerio",
        "web development", "web app", "web application",
    ],

    "Mobile Development": [
        "android", "ios", "swift", "swiftui", "objective-c", "kotlin",
        "java android", "flutter", "dart", "react native", "ionic",
        "xamarin", "maui", "cordova", "mobile app", "mobile development",
        "app development", "cross-platform", "xcode", "android studio",
        "gradle", "cocoapods", "firebase mobile", "push notifications",
        "in-app purchase", "app store", "google play", "mobile ui",
        "jetpack compose", "material design", "human interface guidelines",
        "mobile security", "offline first", "mobile testing", "appium",
    ],

    "Software Engineering": [
        "c#", "c++", "c programming", ".net", "dotnet", "asp.net", "java",
        "python", "oop", "object oriented", "design patterns", "dsa",
        "data structures", "algorithms", "software architecture",
        "solid principles", "clean code", "refactoring", "microservices",
        "monolith", "event driven", "domain driven design", "ddd", "tdd",
        "test driven", "bdd", "unit testing", "integration testing", "jest",
        "junit", "version control", "git", "github", "gitlab", "bitbucket",
        "ci cd", "devops pipeline", "code review", "technical debt",
        "scalability", "concurrency", "multithreading", "memory management",
        "compilers", "interpreters", "operating systems", "linux", "bash",
        "shell scripting", "makefile", "cmake", "rust", "go", "golang",
        "scala", "software engineering", "software development",
    ],

    "Cybersecurity": [
        "security", "hacking", "penetration testing", "pentest", "cyber",
        "network security", "cryptography", "ethical hacking", "white hat",
        "bug bounty", "ctf", "capture the flag", "kali linux", "metasploit",
        "nmap", "wireshark", "burp suite", "owasp", "sql injection", "xss",
        "cross site scripting", "csrf", "buffer overflow", "malware",
        "ransomware", "phishing", "social engineering", "reverse engineering",
        "forensics", "digital forensics", "incident response", "soc", "siem",
        "firewall", "vpn", "zero trust", "iam", "pki", "ssl", "tls",
        "certificate", "compliance", "gdpr", "hipaa", "iso 27001", "nist",
        "risk management", "devsecops", "cloud security", "application security",
        "appsec", "threat modeling", "osint", "cybersecurity", "sscp", "cissp",
        "ceh", "comptia security",
    ],

    "DevOps & Cloud Engineering": [
        "devops", "docker", "kubernetes", "k8s", "container",
        "containerization", "ci cd", "continuous integration",
        "continuous delivery", "continuous deployment", "jenkins",
        "github actions", "gitlab ci", "circleci", "travis", "terraform",
        "ansible", "puppet", "chef", "infrastructure as code", "iac", "aws",
        "amazon web services", "azure", "gcp", "google cloud", "cloud platform",
        "helm", "istio", "service mesh", "microservices", "serverless",
        "lambda", "functions", "monitoring", "observability", "prometheus",
        "grafana", "elk stack", "splunk", "nginx", "apache", "load balancing",
        "auto scaling", "high availability", "site reliability", "sre",
        "platform engineering",
    ],

    "Cloud Infrastructure": [
        "aws", "azure", "gcp", "google cloud", "cloud computing",
        "cloud architecture", "ec2", "s3", "rds", "cloudformation",
        "vpc", "iam roles", "azure devops", "azure functions", "azure blob",
        "azure ad", "bigquery", "cloud run", "gke", "cloud storage", "pub sub",
        "multi cloud", "hybrid cloud", "private cloud", "saas", "paas", "iaas",
        "cloud migration", "cloud native", "cloud cost optimization", "finops",
        "cdn", "cloudfront", "route 53", "dns", "edge computing",
        "fog computing",
    ],

    "Game Development": [
        "game development", "unity", "unreal engine", "godot", "gamemaker",
        "c# unity", "blueprint", "game design", "game mechanics", "2d game",
        "3d game", "physics engine", "collision detection", "game ai",
        "pathfinding", "shader", "glsl", "hlsl", "opengl", "directx",
        "vulkan", "webgl", "multiplayer", "netcode", "game server",
        "matchmaking", "steam", "mobile game", "hyper casual", "rpg", "fps",
        "platformer", "level design", "procedural generation", "vfx",
        "particle system", "audio programming", "animation rigging",
        "inverse kinematics",
    ],

    "Embedded Systems & IoT": [
        "embedded", "iot", "internet of things", "arduino", "raspberry pi",
        "esp32", "rtos", "real time operating system", "firmware",
        "microcontroller", "microprocessor", "uart", "spi", "i2c", "gpio",
        "pwm", "adc", "dac", "fpga", "vhdl", "verilog", "arm cortex", "stm32",
        "mqtt", "zigbee", "lora", "bluetooth", "ble", "wifi module",
        "sensor fusion", "actuator", "industrial iot", "scada",
        "home automation", "smart devices", "wearables",
    ],

    "Blockchain & Web3": [
        "blockchain", "ethereum", "solidity", "smart contracts", "web3",
        "defi", "nft", "cryptocurrency", "bitcoin", "consensus mechanism",
        "proof of work", "proof of stake", "dao", "dapp", "decentralized",
        "ipfs", "filecoin", "hardhat", "truffle", "foundry", "remix ide",
        "metamask", "wallet", "token", "erc20", "erc721", "layer 2",
        "polygon", "solana", "rust blockchain", "zero knowledge", "zk proof",
        "cross chain", "bridge", "oracle", "chainlink",
    ],

    # ── Data Science & AI ─────────────────────────────────────────────────────

    "Data Science": [
        "data science", "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "data cleaning", "data wrangling", "eda", "exploratory data analysis",
        "data visualization", "statistics", "statistical analysis",
        "hypothesis testing", "probability", "bayesian", "scipy", "jupyter",
        "notebook", "anaconda", "feature engineering", "feature selection",
        "dimensionality reduction", "pca", "principal component analysis",
        "correlation", "regression analysis", "time series", "forecasting",
        "anomaly detection", "a b testing",
    ],

    "Machine Learning": [
        "machine learning", "scikit-learn", "sklearn",
        "supervised learning", "unsupervised learning",
        "reinforcement learning", "deep learning", "neural network",
        "tensorflow", "keras", "pytorch", "regression", "classification",
        "clustering", "random forest", "gradient boosting", "xgboost",
        "lightgbm", "catboost", "svm", "support vector machine",
        "naive bayes", "knn", "decision tree", "cross validation",
        "hyperparameter tuning", "grid search", "model evaluation",
        "confusion matrix", "roc curve", "precision recall", "overfitting",
        "regularization", "ensemble methods", "bagging", "boosting",
        "stacking",
    ],

    "Deep Learning": [
        "deep learning", "neural network", "cnn", "convolutional neural network",
        "rnn", "lstm", "gru", "transformer", "attention mechanism",
        "self attention", "bert", "gpt", "vit", "vision transformer", "resnet",
        "vgg", "efficientnet", "gan", "generative adversarial network", "vae",
        "variational autoencoder", "diffusion model", "stable diffusion",
        "backpropagation", "gradient descent", "batch normalization",
        "dropout", "activation function", "relu", "sigmoid",
        "transfer learning", "fine tuning", "pretrained model",
        "model compression", "quantization", "pruning",
        "knowledge distillation", "cuda", "gpu training",
    ],

    "Natural Language Processing": [
        "nlp", "natural language processing", "text classification",
        "sentiment analysis", "named entity recognition", "ner",
        "part of speech", "pos tagging", "tokenization", "stemming",
        "lemmatization", "stopwords", "tfidf", "bag of words", "word2vec",
        "glove", "fasttext", "embeddings", "language model", "llm", "bert",
        "gpt", "transformers", "hugging face", "spacy", "nltk", "gensim",
        "machine translation", "text summarization", "question answering",
        "chatbot", "speech recognition", "text to speech",
        "information retrieval", "semantic search", "topic modeling", "lda",
        "coreference resolution", "dependency parsing",
    ],

    "Computer Vision": [
        "computer vision", "image processing", "opencv", "object detection",
        "yolo", "image segmentation", "semantic segmentation",
        "instance segmentation", "face recognition", "facial detection",
        "pose estimation", "optical flow", "image classification",
        "feature extraction", "sift", "orb", "hog", "depth estimation",
        "3d reconstruction", "point cloud", "lidar", "video analysis",
        "action recognition", "tracking", "kalman filter",
        "augmented reality", "mixed reality", "image generation",
        "medical imaging", "dicom", "pathology", "radiology ai",
    ],

    "Generative AI & LLMs": [
        "generative ai", "llm", "large language model", "gpt", "chatgpt",
        "claude", "gemini", "llama", "mistral", "prompt engineering",
        "prompt design", "rag", "retrieval augmented generation",
        "vector database", "embedding", "langchain", "llamaindex",
        "fine tuning", "lora", "qlora", "instruction tuning",
        "reinforcement learning from human feedback", "rlhf",
        "constitutional ai", "ai agents", "autonomous agents", "tool use",
        "function calling", "multimodal", "vision language model", "vlm",
        "text to image", "text to video", "ai hallucination", "ai safety",
        "responsible ai", "model alignment",
    ],

    "MLOps & AI Engineering": [
        "mlops", "ml engineering", "model deployment", "model serving",
        "inference", "mlflow", "kubeflow", "airflow", "feature store",
        "data pipeline", "model monitoring", "data drift", "concept drift",
        "model versioning", "canary deployment", "shadow deployment",
        "bentoml", "torchserve", "triton inference", "onnx", "tensorrt",
        "model registry", "experiment tracking", "weights and biases",
        "wandb", "dvc", "data version control", "model explainability",
        "shap", "lime",
    ],

    "Big Data & Data Engineering": [
        "big data", "hadoop", "spark", "apache spark", "pyspark", "hive",
        "hbase", "kafka", "apache kafka", "flink", "apache flink", "airflow",
        "luigi", "etl", "elt", "data pipeline", "data warehouse", "data lake",
        "data lakehouse", "snowflake", "databricks", "redshift", "bigquery",
        "azure synapse", "dbt", "data build tool", "data modeling",
        "star schema", "data vault", "streaming data", "batch processing",
        "real time analytics", "lambda architecture", "delta lake", "iceberg",
        "parquet", "avro", "orc",
    ],

    "Databases & SQL": [
        "sql", "database", "mysql", "postgresql", "sqlite", "oracle",
        "sql server", "mongodb", "cassandra", "redis", "elasticsearch",
        "dynamodb", "couchdb", "nosql", "relational database",
        "document database", "graph database", "neo4j", "database design",
        "normalization", "indexing", "query optimization",
        "stored procedures", "triggers", "transactions", "acid",
        "cap theorem", "orm", "sqlalchemy", "hibernate", "prisma",
        "sequelize", "database administration", "backup", "replication",
        "sharding",
    ],

    "Data Analytics & Business Intelligence": [
        "business intelligence", "bi", "tableau", "power bi",
        "looker", "google data studio", "qlik",
        "metabase", "superset", "kpi", "dashboard",
        "data storytelling", "excel analytics", "pivot table", "vlookup",
        "power query", "dax", "web analytics", "funnel analysis",
        "cohort analysis", "retention analysis", "customer analytics",
        "market basket analysis", "rfm analysis",
        # NOTE: "analytics" and "report" removed — too generic and cause
        # false positives (e.g. "A Crash Course in Causality" was hitting
        # this bucket). Use only compound phrases for BI.
    ],

    # ── Business & Management ─────────────────────────────────────────────────

    "Digital Marketing": [
        "seo", "search engine optimization", "sem", "ppc", "google ads",
        "facebook ads", "social media marketing", "content marketing",
        "email marketing", "copywriting", "branding", "brand strategy",
        "influencer marketing", "affiliate marketing",
        "conversion rate optimization", "cro", "landing page",
        "lead generation", "marketing funnel", "customer journey", "personas",
        "target audience", "google analytics", "growth hacking",
        "viral marketing", "youtube marketing", "instagram", "tiktok marketing",
        "linkedin marketing", "marketing automation", "hubspot", "mailchimp",
        "crm marketing", "digital marketing",
    ],

    "Finance & Accounting": [
        "finance", "accounting", "financial analysis", "financial modeling",
        "stock", "trading", "investment", "portfolio", "equity", "bonds",
        "derivatives", "options", "futures", "forex",
        "cryptocurrency trading", "algorithmic trading", "excel finance",
        "financial excel", "tax", "taxation", "corporate finance",
        "valuation", "dcf", "discounted cash flow", "financial statements",
        "balance sheet", "income statement", "cash flow", "ratio analysis",
        "hedge", "quantitative finance", "fintech",
        "bookkeeping", "quickbooks", "xero", "gaap", "ifrs", "audit",
    ],

    "Financial Analysis": [
        "financial analysis", "quantitative analysis", "quant",
        "financial modeling", "excel modeling", "scenario analysis",
        "sensitivity analysis", "monte carlo", "bloomberg terminal",
        "capital markets", "asset management", "wealth management",
        "private equity", "venture capital", "mergers acquisitions", "m&a",
        "credit analysis", "credit risk", "market risk", "operational risk",
        "cfa", "frm", "cpa", "chartered accountant", "financial planning",
    ],

    "Project Management": [
        "project management", "agile", "scrum", "kanban", "waterfall", "pmp",
        "prince2", "lean", "six sigma", "project planning", "gantt chart",
        "leadership", "team management", "stakeholder management",
        "jira", "trello", "asana", "monday", "confluence",
        "notion", "entrepreneurship", "startup", "product management",
        "roadmap", "sprint planning", "retrospective", "velocity",
        "backlog grooming", "okr", "change management", "strategic planning",
    ],

    "Human Resources": [
        "hr", "human resources", "talent acquisition", "recruiting", "hiring",
        "onboarding", "employee engagement", "performance management",
        "payroll", "compensation", "benefits", "labor law",
        "organizational development", "training development",
        "learning development", "diversity inclusion", "dei",
        "employee relations", "hr analytics", "succession planning",
        "workforce planning", "hris", "workday", "sap hr",
    ],

    "Sales & Customer Success": [
        "sales", "b2b sales", "b2c sales", "salesforce", "crm",
        "hubspot crm", "cold calling", "prospecting",
        "pipeline management", "closing deals",
        "objection handling", "sales funnel", "account management",
        "customer success", "customer retention", "upselling",
        "cross selling", "revenue operations", "revops", "customer service",
        "zendesk", "intercom",
    ],

    "Supply Chain & Operations": [
        "supply chain", "logistics", "operations management",
        "inventory management", "procurement", "vendor management", "erp",
        "sap", "oracle erp", "demand forecasting", "warehouse management",
        "last mile delivery", "lean manufacturing",
        "total quality management", "tqm", "import export", "customs",
        "freight", "incoterms", "operations research", "linear programming",
        "optimization",
    ],

    "Entrepreneurship & Strategy": [
        "entrepreneurship", "startup", "business model", "lean startup", "mvp",
        "business strategy", "competitive analysis", "porter five forces",
        "swot", "market research", "market analysis", "go to market",
        "product launch", "fundraising", "pitch deck", "investor",
        "angel investor", "bootstrapping", "growth strategy", "scaling",
        "business development", "innovation", "design thinking",
        "business plan", "feasibility study",
    ],

    # ── Arts, Design & Media ──────────────────────────────────────────────────

    "Graphic Design": [
        "graphic design", "photoshop", "illustrator", "indesign",
        "affinity designer", "figma", "sketch", "adobe xd", "ui design",
        "ux design", "user experience", "typography", "color theory",
        "layout design", "logo design", "branding design", "print design",
        "packaging design", "poster design", "infographic",
        "vector illustration", "raster", "digital illustration", "procreate",
        "motion graphics", "after effects", "animation", "lottie",
        "user research", "wireframing", "prototyping", "usability testing",
    ],

    "Video Production & Editing": [
        "video editing", "premiere pro", "final cut pro", "davinci resolve",
        "motion graphics", "color grading", "color correction",
        "videography", "cinematography", "camera work",
        "lighting", "audio production", "youtube channel", "content creation",
        "vlog", "podcast production", "screencast", "screen recording",
        "obs studio", "streaming", "twitch", "visual effects", "vfx",
        "compositing", "green screen", "chroma key", "3d animation",
        "blender", "maya", "cinema 4d",
    ],

    "3D Modeling & Animation": [
        "3d modeling", "blender", "maya", "cinema 4d", "3ds max", "zbrush",
        "character modeling", "hard surface modeling", "sculpting",
        "retopology", "uv mapping", "texturing", "pbr materials",
        "substance painter", "rigging", "skinning", "animation", "keyframe",
        "motion capture", "3d printing", "cad", "autocad", "solidworks",
        "fusion 360", "rendering", "cycles", "eevee", "arnold", "vray",
        "redshift", "vr ar content", "game ready assets",
    ],

    "Music & Audio Production": [
        "music production", "daw", "ableton", "fl studio", "logic pro",
        "pro tools", "mixing", "mastering", "sound design", "synthesis",
        "sampling", "music theory", "beat making", "hip hop production",
        "electronic music", "audio engineering", "recording",
        "studio recording", "foley", "sound effects",
        "film scoring", "composition", "midi", "vst plugins", "audio fx",
        "eq", "compression",
    ],

    "Photography": [
        "photography", "lightroom", "photoshop retouching", "camera settings",
        "exposure", "aperture", "shutter speed", "iso", "composition",
        "portrait photography", "landscape photography",
        "product photography", "street photography", "wildlife photography",
        "drone photography", "photo editing", "photo retouching",
        "color grading photography", "studio lighting", "natural light",
        "flash photography",
    ],

    # ── Health, Science & Education ───────────────────────────────────────────

    "Healthcare & Medicine": [
        "medical", "healthcare", "clinical", "nursing", "pharmacology",
        "anatomy", "physiology", "pathology", "diagnosis", "treatment",
        "ehr", "electronic health records", "telemedicine",
        "health informatics", "medical imaging", "radiology", "mri",
        "ct scan", "ultrasound", "public health", "epidemiology",
        "biostatistics", "clinical trials", "mental health", "psychiatry",
        "psychology", "therapy", "counseling", "nutrition", "dietetics",
        "fitness", "rehabilitation",
    ],

    "Bioinformatics & Computational Biology": [
        "bioinformatics", "genomics", "proteomics", "transcriptomics",
        "sequence alignment", "blast", "bwa", "bowtie", "samtools",
        "variant calling", "snp", "gwas", "gene expression", "rna seq",
        "scrnaseq", "single cell", "metagenomics", "phylogenetics",
        "r bioconductor", "biopython", "structural biology",
        "molecular docking", "drug discovery", "cheminformatics", "rdkit",
        "protein structure", "alphafold", "cryo em", "flow cytometry",
    ],

    "Data Science in Healthcare": [
        "health data", "clinical data", "ehr data", "health analytics",
        "predictive modeling healthcare", "survival analysis", "clinical nlp",
        "medical image analysis", "radiology ai", "pathology ai",
        "drug discovery ai", "genomic data science", "real world evidence",
        "population health", "risk stratification", "readmission prediction",
    ],

    "Physics & Engineering Sciences": [
        "physics", "mechanics", "thermodynamics", "electromagnetism",
        "optics", "quantum mechanics", "quantum computing", "qiskit",
        "quantum algorithms", "signal processing", "dsp", "fourier transform",
        "filter design", "control systems", "pid controller", "matlab",
        "simulink", "finite element analysis", "fea",
        "computational fluid dynamics", "cfd", "materials science",
        "nanotechnology", "photonics",
    ],

    "Environmental Science & Sustainability": [
        "sustainability", "climate change", "carbon footprint",
        "renewable energy", "solar energy", "wind energy", "energy storage",
        "battery technology", "environmental monitoring", "remote sensing",
        "gis", "geospatial", "green building", "leed", "circular economy",
        "life cycle assessment", "esg", "environmental reporting",
        "carbon market",
    ],

    # ── Soft Skills & Professional Development ────────────────────────────────

    "Soft Skills & Communication": [
        "communication", "public speaking", "presentation skills",
        "negotiation", "conflict resolution", "active listening",
        "interpersonal skills", "emotional intelligence", "eq", "empathy",
        "teamwork", "collaboration", "time management", "productivity",
        "goal setting", "habit building", "critical thinking",
        "problem solving", "creativity", "innovation mindset",
        "business writing", "technical writing",
        "storytelling", "interview preparation", "resume writing",
        "linkedin profile", "job search", "networking", "personal branding",
        "career development", "mentoring",
    ],

    "Language Learning": [
        "english", "ielts", "toefl", "grammar", "vocabulary", "pronunciation",
        "spanish", "french", "german", "arabic", "chinese", "japanese",
        "korean", "business english", "academic writing", "esl",
        "language fluency", "translation", "interpretation", "linguistics",
        "language learning",
    ],

    "Teaching & Education": [
        "curriculum design", "instructional design", "elearning", "lms",
        "moodle", "canvas", "google classroom", "online teaching",
        "blended learning", "assessment design", "rubric",
        "differentiated instruction", "stem education", "coding for kids",
        "educational technology", "edtech", "higher education", "k12",
        "corporate training",
    ],

    # ── Emerging & Specialised Fields ─────────────────────────────────────────

    "Robotics & Automation": [
        "robotics", "ros", "robot operating system", "autonomous systems",
        "computer vision robotics", "slam", "localization", "mapping",
        "motion planning", "robot arm", "collaborative robot",
        "industrial automation", "plc", "process automation", "rpa",
        "robotic process automation", "uipath", "automation anywhere",
        "blue prism", "drone", "uav", "autonomous vehicle",
        "self driving", "lidar",
    ],

    "AR & VR Development": [
        "augmented reality", "virtual reality", "mixed reality", "xr",
        "unity xr", "unreal vr", "arkit", "arcore", "hololens",
        "meta quest", "openxr", "webxr", "spatial computing",
        "immersive experience", "360 video", "vr training", "ar marketing",
    ],

    "Legal & Compliance": [
        "legal", "contract law", "intellectual property", "copyright",
        "trademark", "patent", "privacy law", "data protection",
        "regulatory", "corporate law", "employment law",
        "business law", "legal writing", "legal research", "paralegal",
        "legaltech",
    ],

    "Real Estate": [
        "real estate", "property investment", "real estate investing",
        "rental property", "house flipping", "commercial real estate",
        "property management", "real estate finance", "mortgage",
        "property valuation", "appraisal", "real estate development",
        "urban planning", "zoning",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _parse_enrollment(value: str) -> float:
    """Convert enrollment strings like '12K' or '1.5M' to a plain float."""
    text = str(value).strip().upper()
    try:
        if "K" in text:
            return float(text.replace("K", "")) * 1_000
        if "M" in text:
            return float(text.replace("M", "")) * 1_000_000
        return float(text)
    except ValueError:
        return np.nan


def _score_subject(text: str) -> tuple[str, int]:
    """
    Score every subject by counting keyword hits in *text*.
    Returns (best_subject, hit_count).

    This replaces the old first-match approach which caused false positives
    (e.g. generic words like 'analytics' matching the wrong subject).
    """
    normalised = str(text).lower()
    best_subject = "Other"
    best_score   = 0

    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in normalised)
        if score > best_score:
            best_score   = score
            best_subject = subject

    return best_subject, best_score


def _get_subject_from_row(row: pd.Series) -> tuple[str, int]:
    """
    Try title first; if score is low, combine title + description.
    Returns (subject, score).
    """
    title       = str(row.get("title", "") or "")
    description = str(row.get("description", "") or "")

    subject, score = _score_subject(title)

    if score < MIN_CONFIDENCE_SCORE and description.strip():
        # Use combined text for a better signal
        combined_subject, combined_score = _score_subject(title + " " + description)
        if combined_score > score:
            return combined_subject, combined_score

    return subject, score


def _normalise_difficulty(value: str) -> str:
    """Map raw difficulty strings to a consistent controlled vocabulary."""
    normalised = str(value or "").strip().lower()
    return DIFFICULTY_MAP.get(normalised, "Beginner")  # safe default


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLAUDE API – LOW-CONFIDENCE LABELLING
# ──────────────────────────────────────────────────────────────────────────────

VALID_SUBJECTS = sorted(SUBJECT_KEYWORDS.keys()) + ["Other"]


def _label_with_claude(titles_and_descriptions: list[dict]) -> list[str]:
    """
    Send a batch of low-confidence courses to the Claude API for accurate
    subject classification.

    Args:
        titles_and_descriptions: list of {"title": ..., "description": ...}

    Returns:
        list of subject strings, same length and order as input.
    """
    client = anthropic.Anthropic()

    courses_text = "\n".join(
        f'{i+1}. Title: "{item["title"]}" | Description: "{str(item.get("description",""))[:200]}"'
        for i, item in enumerate(titles_and_descriptions)
    )

    valid_list = "\n".join(f"- {s}" for s in VALID_SUBJECTS)

    prompt = f"""You are a course taxonomy expert. Classify each online course below into exactly one subject from the valid list.

VALID SUBJECTS:
{valid_list}

COURSES TO CLASSIFY:
{courses_text}

Rules:
- Choose the single most specific subject that fits.
- If none fit, use "Other".
- Reply ONLY with a JSON array of strings, one per course, in the same order.
- No explanations, no markdown, no extra text. Example: ["Data Science", "Web Development"]"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        labels = json.loads(raw)

        # Validate — fall back to "Other" for any unrecognised label
        return [
            lbl if lbl in VALID_SUBJECTS else "Other"
            for lbl in labels
        ]
    except Exception as e:
        log.warning("Claude API labelling failed: %s. Using 'Other' for batch.", e)
        return ["Other"] * len(titles_and_descriptions)


def _apply_claude_labelling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find rows where keyword confidence is low, send them to Claude in batches,
    and update the subject column in-place.
    """
    low_conf_mask = df["_subject_score"] < MIN_CONFIDENCE_SCORE
    low_conf_idx  = df.index[low_conf_mask].tolist()

    if not low_conf_idx:
        log.info("All courses classified with high confidence — skipping Claude API.")
        return df

    log.info(
        "%d / %d courses have low keyword confidence — sending to Claude API …",
        len(low_conf_idx), len(df),
    )

    updated = 0
    for start in range(0, len(low_conf_idx), CLAUDE_BATCH_SIZE):
        batch_idx = low_conf_idx[start : start + CLAUDE_BATCH_SIZE]
        batch_rows = df.loc[batch_idx, ["title", "description"]].to_dict("records")

        labels = _label_with_claude(batch_rows)

        for idx, label in zip(batch_idx, labels):
            df.at[idx, "subject"] = label

        updated += len(batch_idx)
        log.info("  Labelled %d / %d low-confidence courses …", updated, len(low_conf_idx))

        # Avoid hammering the API
        if start + CLAUDE_BATCH_SIZE < len(low_conf_idx):
            time.sleep(0.5)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. LOAD
# ──────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three raw CSVs and return them as DataFrames."""
    log.info("Loading raw data …")
    coursera = pd.read_csv(INPUT_FILES["coursera"])
    edx      = pd.read_csv(INPUT_FILES["edx"])
    udemy    = pd.read_csv(INPUT_FILES["udemy"])
    log.info(
        "Loaded %d Coursera | %d EdX | %d Udemy rows.",
        len(coursera), len(edx), len(udemy),
    )
    return coursera, edx, udemy


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLEAN (per-source transformations)
# ──────────────────────────────────────────────────────────────────────────────

def _fix_dtypes(
    udemy: pd.DataFrame,
    coursera: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fix known type issues before any merging."""
    udemy["published_timestamp"] = pd.to_datetime(
        udemy["published_timestamp"], errors="coerce"
    )
    coursera["course_students_enrolled"] = (
        coursera["course_students_enrolled"].apply(_parse_enrollment)
    )
    return udemy, coursera


def _drop_columns(
    coursera: pd.DataFrame,
    edx: pd.DataFrame,
    udemy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop irrelevant columns from each source."""
    coursera = coursera.drop(columns=COLUMNS_TO_DROP["coursera"], errors="ignore")
    edx      = edx.drop(columns=COLUMNS_TO_DROP["edx"],      errors="ignore")
    udemy    = udemy.drop(columns=COLUMNS_TO_DROP["udemy"],   errors="ignore")
    return coursera, edx, udemy


def _rename_and_merge(
    coursera: pd.DataFrame,
    edx: pd.DataFrame,
    udemy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rename columns to a shared schema and concatenate all three sources.
    Shared schema: title | difficulty_level | description | url |
                   rating | no_students | certificate_type
    """
    coursera = coursera.rename(columns={
        "course_title":            "title",
        "course_Certificate_type": "certificate_type",
        "course_rating":           "rating",
        "course_difficulty":       "difficulty_level",
        "course_students_enrolled":"no_students",
        "course_overview":         "description",
    })

    edx = edx.rename(columns={
        "Name":               "title",
        "Difficulty Level":   "difficulty_level",
        "Course Description": "description",
        "Link":               "url",
    })

    udemy = udemy.rename(columns={
        "course_title":       "title",
        "level":              "difficulty_level",
        "Course Description": "description",
        "num_subscribers":    "no_students",
    })

    merged = pd.concat([coursera, udemy, edx], axis=0, ignore_index=True)
    log.info("Merged dataset: %d rows.", len(merged))
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 8. ENRICH
# ──────────────────────────────────────────────────────────────────────────────

def _add_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and attach a subject label + confidence score to every course.
    Low-confidence rows are later refined via the Claude API.
    """
    results = df.apply(_get_subject_from_row, axis=1)
    df["subject"]        = results.apply(lambda x: x[0])
    df["_subject_score"] = results.apply(lambda x: x[1])

    subject_counts = df["subject"].value_counts()
    low_conf_count = (df["_subject_score"] < MIN_CONFIDENCE_SCORE).sum()
    log.info("Top 10 detected subjects:\n%s", subject_counts.head(10))
    log.info("Low-confidence rows (score < %d): %d", MIN_CONFIDENCE_SCORE, low_conf_count)
    return df


def _normalise_difficulty_column(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise all difficulty_level values to Beginner / Intermediate / Advanced."""
    df["difficulty_level"] = df["difficulty_level"].apply(_normalise_difficulty)
    log.info("Difficulty distribution:\n%s", df["difficulty_level"].value_counts())
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate courses (same normalised title)."""
    before = len(df)
    df["_title_norm"] = df["title"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["_title_norm"]).drop(columns=["_title_norm"])
    log.info("Dropped %d duplicate titles. Remaining: %d", before - len(df), len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 9. IMPUTE MISSING VALUES
# ──────────────────────────────────────────────────────────────────────────────

def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using group-aware medians / modes.

    Strategy
    --------
    Numeric  → median of the (subject, difficulty_level) group;
               fall back to the column-wide median if the group is all-NaN.
    Categorical → mode of the group; fall back to the column-wide mode.
    """
    group_keys = ["subject", "difficulty_level"]

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        global_median = df[col].median()
        df[col] = df.groupby(group_keys)[col].transform(
            lambda x: x.fillna(x.median() if x.notna().any() else global_median)
        )

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        global_mode     = df[col].mode()
        global_fallback = global_mode.iloc[0] if not global_mode.empty else "Unknown"
        df[col] = df.groupby(group_keys)[col].transform(
            lambda x: x.fillna(
                x.mode().iloc[0] if not x.mode().empty else global_fallback
            )
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 10. VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def _validate_output(df: pd.DataFrame) -> None:
    """Log a quality report so problems are visible before saving."""
    log.info("── Output validation ─────────────────────────────────────────")
    log.info("Shape            : %s", df.shape)
    log.info("Columns          : %s", list(df.columns))
    log.info("Subject 'Other'  : %d rows", (df["subject"] == "Other").sum())
    log.info("Missing rating   : %d rows", df["rating"].isna().sum())
    log.info("Missing duration : %d rows", df["content_duration"].isna().sum())
    log.info("Difficulty dist  :\n%s", df["difficulty_level"].value_counts())
    log.info("Top subjects     :\n%s", df["subject"].value_counts().head(15))
    log.info("─────────────────────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline() -> pd.DataFrame:
    """
    Orchestrate every step of the preprocessing pipeline and return
    the cleaned DataFrame.

    Steps
    -----
    1.  Load raw CSVs
    2.  Fix data types
    3.  Drop irrelevant columns
    4.  Rename to a shared schema and merge
    5.  Drop duplicate titles
    6.  Drop url column
    7.  Normalise difficulty levels
    8.  Detect subject (scoring-based, title + description)
    9.  Refine low-confidence labels via Claude API
    10. Drop internal score column
    11. Impute missing values
    12. Validate output
    """
    log.info("── AirForge Preprocessing Pipeline (v2) ─────────────────────")

    coursera, edx, udemy = load_data()

    udemy, coursera = _fix_dtypes(udemy, coursera)
    coursera, edx, udemy = _drop_columns(coursera, edx, udemy)

    df = _rename_and_merge(coursera, edx, udemy)
    df = _drop_duplicates(df)
    df = df.drop(columns=["url"], errors="ignore")

    df = _normalise_difficulty_column(df)
    df = _add_subject_column(df)
    df = _apply_claude_labelling(df)     # ← refines low-confidence rows via API

    # Remove internal helper column before saving
    df = df.drop(columns=["_subject_score"], errors="ignore")

    df = _fill_missing_values(df)
    _validate_output(df)

    log.info("Pipeline complete. Final shape: %s", df.shape)
    log.info("─────────────────────────────────────────────────────────────")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 12. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(BASE_DIR_PROCESSED, exist_ok=True)
    courses_df = run_preprocessing_pipeline()
    courses_df.to_csv(OUTPUT_FILE, index=False)
    log.info("✅  Saved clean data → %s", OUTPUT_FILE)