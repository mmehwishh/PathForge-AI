"""
preprocessing.py
================
AirForge ML Service – Data Preprocessing Pipeline

Loads raw course data from Coursera, EdX, and Udemy, cleans and
standardises it into a single DataFrame, detects each course's subject,
fills missing values, and saves the result as a CSV ready for model training.

Author : Mehwish
Project: AirForge Learning-Path Recommender
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import logging

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Centralise every path and constant here so nothing is buried in functions.

BASE_DIR   = "ml_service/data/raw"
BASE_DIR_PROCESSED = "ml_service/data/processed"
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 3. SUBJECT-KEYWORD TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────
# Each key is a human-readable subject label.
# Each value is a list of lowercase keywords that signal that subject.
# get_subject() scans a course title and returns the first matching subject.

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
        "scala",
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
        "appsec", "threat modeling", "osint",
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
        "platform engineering", "devsecops",
    ],

    "Cloud Infrastructure": [
        "aws", "azure", "gcp", "google cloud", "cloud computing",
        "cloud architecture", "ec2", "s3", "rds", "lambda", "cloudformation",
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
        "edge computing", "sensor fusion", "actuator", "industrial iot",
        "scada", "home automation", "smart devices", "wearables",
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
        "machine learning", "ml", "scikit-learn", "sklearn",
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
        "stable diffusion", "medical imaging", "dicom", "pathology",
        "radiology ai",
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
        "model versioning", "a b testing model", "canary deployment",
        "shadow deployment", "bentoml", "torchserve", "triton inference",
        "onnx", "tensorrt", "model registry", "experiment tracking",
        "weights and biases", "wandb", "dvc", "data version control",
        "model explainability", "shap", "lime",
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
        "analytics", "business intelligence", "bi", "tableau", "power bi",
        "looker", "google analytics", "google data studio", "qlik",
        "metabase", "superset", "kpi", "dashboard", "report", "metrics",
        "data storytelling", "excel analytics", "pivot table", "vlookup",
        "power query", "dax", "web analytics", "funnel analysis",
        "cohort analysis", "retention analysis", "customer analytics",
        "market basket analysis", "rfm analysis",
    ],

    # ── Business & Management ─────────────────────────────────────────────────

    "Digital Marketing": [
        "seo", "search engine optimization", "sem", "ppc", "google ads",
        "facebook ads", "social media marketing", "content marketing",
        "email marketing", "copywriting", "branding", "brand strategy",
        "influencer marketing", "affiliate marketing",
        "conversion rate optimization", "cro", "landing page",
        "lead generation", "marketing funnel", "customer journey", "personas",
        "target audience", "analytics", "google analytics", "growth hacking",
        "viral marketing", "youtube marketing", "instagram", "tiktok marketing",
        "linkedin marketing", "marketing automation", "hubspot", "mailchimp",
        "crm marketing",
    ],

    "Finance & Accounting": [
        "finance", "accounting", "financial analysis", "financial modeling",
        "stock", "trading", "investment", "portfolio", "equity", "bonds",
        "derivatives", "options", "futures", "forex",
        "cryptocurrency trading", "algorithmic trading", "excel finance",
        "financial excel", "tax", "taxation", "corporate finance",
        "valuation", "dcf", "discounted cash flow", "financial statements",
        "balance sheet", "income statement", "cash flow", "ratio analysis",
        "risk management", "hedge", "quantitative finance", "fintech",
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
        "risk management", "jira", "trello", "asana", "monday", "confluence",
        "notion", "entrepreneurship", "startup", "product management",
        "roadmap", "sprint planning", "retrospective", "velocity",
        "backlog grooming", "okr", "kpi", "change management",
        "strategic planning",
    ],

    "Human Resources": [
        "hr", "human resources", "talent acquisition", "recruiting", "hiring",
        "onboarding", "employee engagement", "performance management",
        "payroll", "compensation", "benefits", "labor law", "compliance",
        "organizational development", "training development",
        "learning development", "diversity inclusion", "dei",
        "employee relations", "hr analytics", "succession planning",
        "workforce planning", "hris", "workday", "sap hr",
    ],

    "Sales & Customer Success": [
        "sales", "b2b sales", "b2c sales", "salesforce", "crm",
        "hubspot crm", "cold calling", "lead generation", "prospecting",
        "pipeline management", "negotiation", "closing deals",
        "objection handling", "sales funnel", "account management",
        "customer success", "customer retention", "upselling",
        "cross selling", "revenue operations", "revops", "customer service",
        "support", "zendesk", "intercom",
    ],

    "Supply Chain & Operations": [
        "supply chain", "logistics", "operations management",
        "inventory management", "procurement", "vendor management", "erp",
        "sap", "oracle erp", "demand forecasting", "warehouse management",
        "last mile delivery", "lean manufacturing", "six sigma",
        "total quality management", "tqm", "import export", "customs",
        "freight", "incoterms", "operations research", "linear programming",
        "optimization",
    ],

    "Entrepreneurship & Strategy": [
        "entrepreneurship", "startup", "business model", "lean startup", "mvp",
        "business strategy", "competitive analysis", "porter five forces",
        "swot", "market research", "market analysis", "go to market",
        "product launch", "fundraising", "pitch deck", "investor",
        "angel investor", "venture capital", "bootstrapping",
        "growth strategy", "scaling", "business development", "innovation",
        "design thinking", "business plan", "feasibility study",
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
        "after effects", "motion graphics", "color grading",
        "color correction", "videography", "cinematography", "camera work",
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
        "studio recording", "podcast", "foley", "sound effects",
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
        "writing skills", "business writing", "technical writing",
        "storytelling", "interview preparation", "resume writing",
        "linkedin profile", "job search", "networking", "personal branding",
        "career development", "mentoring",
    ],

    "Language Learning": [
        "english", "ielts", "toefl", "grammar", "vocabulary", "pronunciation",
        "spanish", "french", "german", "arabic", "chinese", "japanese",
        "korean", "business english", "academic writing", "esl",
        "language fluency", "translation", "interpretation", "linguistics",
    ],

    "Teaching & Education": [
        "curriculum design", "instructional design", "elearning", "lms",
        "moodle", "canvas", "google classroom", "online teaching",
        "blended learning", "assessment design", "rubric",
        "differentiated instruction", "stem education", "coding for kids",
        "educational technology", "edtech", "higher education", "k12",
        "corporate training", "learning development",
    ],

    # ── Emerging & Specialised Fields ─────────────────────────────────────────

    "Robotics & Automation": [
        "robotics", "ros", "robot operating system", "autonomous systems",
        "computer vision robotics", "slam", "localization", "mapping",
        "motion planning", "inverse kinematics", "robot arm",
        "collaborative robot", "industrial automation", "plc", "scada",
        "process automation", "rpa", "robotic process automation", "uipath",
        "automation anywhere", "blue prism", "drone", "uav",
        "autonomous vehicle", "self driving", "lidar",
    ],

    "AR & VR Development": [
        "augmented reality", "virtual reality", "mixed reality", "xr", "ar",
        "vr", "unity xr", "unreal vr", "arkit", "arcore", "hololens",
        "meta quest", "openxr", "webxr", "spatial computing",
        "immersive experience", "360 video", "vr training", "ar marketing",
    ],

    "Legal & Compliance": [
        "legal", "contract law", "intellectual property", "copyright",
        "trademark", "patent", "gdpr", "privacy law", "data protection",
        "compliance", "regulatory", "corporate law", "employment law",
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
    """
    Convert enrollment strings like '12K' or '1.5M' to a plain float.
    Returns np.nan for anything that cannot be parsed.
    """
    text = str(value).strip().upper()
    try:
        if "K" in text:
            return float(text.replace("K", "")) * 1_000
        if "M" in text:
            return float(text.replace("M", "")) * 1_000_000
        return float(text)
    except ValueError:
        return np.nan


def _get_subject(title: str) -> str:
    """
    Return the first SUBJECT_KEYWORDS key whose keyword list contains
    any word found in *title*. Falls back to 'Other'.
    """
    normalised = str(title).lower()
    for subject, keywords in SUBJECT_KEYWORDS.items():
        if any(kw in normalised for kw in keywords):
            return subject
    return "Other"


# ──────────────────────────────────────────────────────────────────────────────
# 5. LOAD
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
# 6. CLEAN (per-source transformations)
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
        "course_title":           "title",
        "course_Certificate_type": "certificate_type",
        "course_rating":          "rating",
        "course_difficulty":      "difficulty_level",
        "course_students_enrolled": "no_students",
        "course_overview":        "description",
    })

    edx = edx.rename(columns={
        "Name":               "title",
        "Difficulty Level":   "difficulty_level",
        "Course Description": "description",
        "Link":               "url",
    })

    udemy = udemy.rename(columns={
        "course_title":    "title",
        "level":           "difficulty_level",
        "Course Description": "description",
        "num_subscribers": "no_students",
    })

    merged = pd.concat([coursera, udemy, edx], axis=0, ignore_index=True)
    log.info("Merged dataset: %d rows.", len(merged))
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 7. ENRICH
# ──────────────────────────────────────────────────────────────────────────────

def _add_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and attach a subject label to every course."""
    df["subject"] = df["title"].apply(_get_subject)
    subject_counts = df["subject"].value_counts()
    log.info("Top 5 detected subjects:\n%s", subject_counts.head())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 8. IMPUTE MISSING VALUES
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

    # --- numeric columns ---
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        global_median = df[col].median()
        df[col] = df.groupby(group_keys)[col].transform(
            lambda x: x.fillna(x.median() if x.notna().any() else global_median)
        )

    # --- categorical columns ---
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        global_mode = df[col].mode()
        global_fallback = global_mode.iloc[0] if not global_mode.empty else "Unknown"
        df[col] = df.groupby(group_keys)[col].transform(
            lambda x: x.fillna(
                x.mode().iloc[0] if not x.mode().empty else global_fallback
            )
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline() -> pd.DataFrame:
    """
    Orchestrate every step of the preprocessing pipeline and return
    the cleaned DataFrame.

    Steps
    -----
    1. Load raw CSVs
    2. Fix data types
    3. Drop irrelevant columns
    4. Rename to a shared schema and merge
    5. Drop any remaining unneeded columns (e.g. url)
    6. Detect subject
    7. Impute missing values
    """
    log.info("── AirForge Preprocessing Pipeline ──────────────────────────")

    coursera, edx, udemy = load_data()

    udemy, coursera = _fix_dtypes(udemy, coursera)
    coursera, edx, udemy = _drop_columns(coursera, edx, udemy)

    df = _rename_and_merge(coursera, edx, udemy)
    df = df.drop(columns=["url"], errors="ignore")

    df = _add_subject_column(df)
    df = _fill_missing_values(df)

    log.info("Pipeline complete. Final shape: %s", df.shape)
    log.info("─────────────────────────────────────────────────────────────")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    courses_df = run_preprocessing_pipeline()
    courses_df.to_csv(OUTPUT_FILE, index=False)
    log.info("✅  Saved clean data → %s", OUTPUT_FILE)