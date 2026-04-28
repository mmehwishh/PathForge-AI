# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# =========================
# 2. LOAD DATA
# =========================
def load_data():
    coursera_data = pd.read_csv('/content/drive/MyDrive/AirForge_Data/coursea_data.csv')
    edx_data = pd.read_csv('/content/drive/MyDrive/AirForge_Data/EdX.csv')
    udemy_data = pd.read_csv('/content/drive/MyDrive/AirForge_Data/udemy_courses.csv')
    return coursera_data, edx_data, udemy_data


# =========================
# 3. HELPER FUNCTIONS
# =========================
def convert_students_enrolled(x):
    num = str(x).strip()
    if 'K' in num or 'k' in num:
        return float(num.replace('k', '').replace('K','')) * 1000
    elif 'M' in num or 'm' in num:
        return float(num.replace('m', '').replace('M','')) * 1000000
    else:
        try:
            return float(num)
        except:
            return np.nan


def convert_datatype(udemy_data, coursera_data):
    udemy_data['published_timestamp'] = pd.to_datetime(udemy_data['published_timestamp'], errors='coerce')
    coursera_data['course_students_enrolled'] = coursera_data['course_students_enrolled'].apply(convert_students_enrolled)
    return udemy_data, coursera_data


def drop_columns(coursera_data, edx_data, udemy_data):
    coursera_data = coursera_data.drop(['Unnamed: 0', 'course_organization'], axis=1, errors='ignore')
    edx_data = edx_data.drop(['University'], axis=1, errors='ignore')
    udemy_data = udemy_data.drop(['course_id', 'is_paid', 'price', 'num_reviews', 'num_lectures', 'published_timestamp'], axis=1, errors='ignore')
    return coursera_data, edx_data, udemy_data


def rename_columns(coursera_data, edx_data, udemy_data):

    coursera_data = coursera_data.rename(columns={
        'course_title': 'title',
        'course_Certificate_type': 'certificate_type',
        'course_rating': 'rating',
        'course_difficulty': 'difficulty_level',
        'course_students_enrolled': 'no_students',
        'course_overview': 'description',
    })

    edx_data = edx_data.rename(columns={
        'Name': 'title',
        'Difficulty Level': 'difficulty_level',
        'Course Description': 'description',
        'Link': 'url'
    })

    udemy_data = udemy_data.rename(columns={
        'course_title': 'title',
        'level': 'difficulty_level',
        'Course Description': 'description',
        'num_subscribers': 'no_students'
    })

    courses_data = pd.concat([coursera_data, udemy_data, edx_data], axis=0)
    return courses_data


# =========================
# 4. SUBJECT DETECTION
# =========================
subject_keywords = {

    # ── Technology & Development ──────────────────────────────────────────────

    'Web Development': [
        'frontend', 'backend', 'fullstack', 'full-stack', 'react', 'angular', 'vue', 'svelte',
        'html', 'html5', 'css', 'css3', 'javascript', 'js', 'typescript', 'ts', 'node', 'nodejs',
        'django', 'flask', 'fastapi', 'php', 'laravel', 'symfony', 'ruby', 'rails', 'ruby on rails',
        'express', 'expressjs', 'next.js', 'nextjs', 'nuxt', 'gatsby', 'remix', 'astro',
        'webpack', 'vite', 'babel', 'sass', 'scss', 'tailwind', 'bootstrap', 'materialui',
        'rest api', 'graphql', 'websocket', 'oauth', 'jwt', 'http', 'https', 'web api',
        'responsive design', 'pwa', 'progressive web app', 'single page application', 'spa',
        'ssr', 'server side rendering', 'static site', 'jamstack', 'cms', 'wordpress',
        'web scraping', 'selenium', 'puppeteer', 'playwright', 'cheerio',
    ],

    'Mobile Development': [
        'android', 'ios', 'swift', 'swiftui', 'objective-c', 'kotlin', 'java android',
        'flutter', 'dart', 'react native', 'ionic', 'xamarin', 'maui', 'cordova',
        'mobile app', 'mobile development', 'app development', 'cross-platform',
        'xcode', 'android studio', 'gradle', 'cocoapods', 'firebase mobile',
        'push notifications', 'in-app purchase', 'app store', 'google play',
        'mobile ui', 'jetpack compose', 'material design', 'human interface guidelines',
        'mobile security', 'offline first', 'mobile testing', 'appium',
    ],

    'Software Engineering': [
        'c#', 'c++', 'c programming', '.net', 'dotnet', 'asp.net', 'java', 'python',
        'oop', 'object oriented', 'design patterns', 'dsa', 'data structures', 'algorithms',
        'software architecture', 'solid principles', 'clean code', 'refactoring',
        'microservices', 'monolith', 'event driven', 'domain driven design', 'ddd',
        'tdd', 'test driven', 'bdd', 'unit testing', 'integration testing', 'jest', 'junit',
        'version control', 'git', 'github', 'gitlab', 'bitbucket', 'ci cd', 'devops pipeline',
        'code review', 'technical debt', 'scalability', 'concurrency', 'multithreading',
        'memory management', 'compilers', 'interpreters', 'operating systems', 'linux',
        'bash', 'shell scripting', 'makefile', 'cmake', 'rust', 'go', 'golang', 'scala',
    ],

    'Cybersecurity': [
        'security', 'hacking', 'penetration testing', 'pentest', 'cyber', 'network security',
        'cryptography', 'ethical hacking', 'white hat', 'bug bounty', 'ctf', 'capture the flag',
        'kali linux', 'metasploit', 'nmap', 'wireshark', 'burp suite', 'owasp',
        'sql injection', 'xss', 'cross site scripting', 'csrf', 'buffer overflow',
        'malware', 'ransomware', 'phishing', 'social engineering', 'reverse engineering',
        'forensics', 'digital forensics', 'incident response', 'soc', 'siem',
        'firewall', 'vpn', 'zero trust', 'iam', 'pki', 'ssl', 'tls', 'certificate',
        'compliance', 'gdpr', 'hipaa', 'iso 27001', 'nist', 'risk management', 'devsecops',
        'cloud security', 'application security', 'appsec', 'threat modeling', 'osint',
    ],

    'DevOps & Cloud Engineering': [
        'devops', 'docker', 'kubernetes', 'k8s', 'container', 'containerization',
        'ci cd', 'continuous integration', 'continuous delivery', 'continuous deployment',
        'jenkins', 'github actions', 'gitlab ci', 'circleci', 'travis',
        'terraform', 'ansible', 'puppet', 'chef', 'infrastructure as code', 'iac',
        'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'cloud platform',
        'helm', 'istio', 'service mesh', 'microservices', 'serverless', 'lambda', 'functions',
        'monitoring', 'observability', 'prometheus', 'grafana', 'elk stack', 'splunk',
        'nginx', 'apache', 'load balancing', 'auto scaling', 'high availability',
        'site reliability', 'sre', 'platform engineering', 'devsecops',
    ],

    'Cloud Infrastructure': [
        'aws', 'azure', 'gcp', 'google cloud', 'cloud computing', 'cloud architecture',
        'ec2', 's3', 'rds', 'lambda', 'cloudformation', 'vpc', 'iam roles',
        'azure devops', 'azure functions', 'azure blob', 'azure ad',
        'bigquery', 'cloud run', 'gke', 'cloud storage', 'pub sub',
        'multi cloud', 'hybrid cloud', 'private cloud', 'saas', 'paas', 'iaas',
        'cloud migration', 'cloud native', 'cloud cost optimization', 'finops',
        'cdn', 'cloudfront', 'route 53', 'dns', 'edge computing', 'fog computing',
    ],

    'Game Development': [
        'game development', 'unity', 'unreal engine', 'godot', 'gamemaker',
        'c# unity', 'blueprint', 'game design', 'game mechanics', '2d game', '3d game',
        'physics engine', 'collision detection', 'game ai', 'pathfinding',
        'shader', 'glsl', 'hlsl', 'opengl', 'directx', 'vulkan', 'webgl',
        'multiplayer', 'netcode', 'game server', 'matchmaking', 'steam',
        'mobile game', 'hyper casual', 'rpg', 'fps', 'platformer',
        'level design', 'procedural generation', 'vfx', 'particle system',
        'audio programming', 'animation rigging', 'inverse kinematics',
    ],

    'Embedded Systems & IoT': [
        'embedded', 'iot', 'internet of things', 'arduino', 'raspberry pi', 'esp32',
        'rtos', 'real time operating system', 'firmware', 'microcontroller', 'microprocessor',
        'uart', 'spi', 'i2c', 'gpio', 'pwm', 'adc', 'dac',
        'fpga', 'vhdl', 'verilog', 'arm cortex', 'stm32',
        'mqtt', 'zigbee', 'lora', 'bluetooth', 'ble', 'wifi module',
        'edge computing', 'sensor fusion', 'actuator', 'industrial iot', 'scada',
        'home automation', 'smart devices', 'wearables',
    ],

    'Blockchain & Web3': [
        'blockchain', 'ethereum', 'solidity', 'smart contracts', 'web3', 'defi',
        'nft', 'cryptocurrency', 'bitcoin', 'consensus mechanism', 'proof of work',
        'proof of stake', 'dao', 'dapp', 'decentralized', 'ipfs', 'filecoin',
        'hardhat', 'truffle', 'foundry', 'remix ide', 'metamask', 'wallet',
        'token', 'erc20', 'erc721', 'layer 2', 'polygon', 'solana', 'rust blockchain',
        'zero knowledge', 'zk proof', 'cross chain', 'bridge', 'oracle', 'chainlink',
    ],

    # ── Data Science & AI ─────────────────────────────────────────────────────

    'Data Science': [
        'data science', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'data cleaning', 'data wrangling', 'eda', 'exploratory data analysis',
        'data visualization', 'statistics', 'statistical analysis', 'hypothesis testing',
        'probability', 'bayesian', 'scipy', 'jupyter', 'notebook', 'anaconda',
        'feature engineering', 'feature selection', 'dimensionality reduction',
        'pca', 'principal component analysis', 'correlation', 'regression analysis',
        'time series', 'forecasting', 'anomaly detection', 'a b testing',
    ],

    'Machine Learning': [
        'machine learning', 'ml', 'scikit-learn', 'sklearn', 'supervised learning',
        'unsupervised learning', 'reinforcement learning', 'deep learning', 'neural network',
        'tensorflow', 'keras', 'pytorch', 'regression', 'classification', 'clustering',
        'random forest', 'gradient boosting', 'xgboost', 'lightgbm', 'catboost',
        'svm', 'support vector machine', 'naive bayes', 'knn', 'decision tree',
        'cross validation', 'hyperparameter tuning', 'grid search', 'model evaluation',
        'confusion matrix', 'roc curve', 'precision recall', 'overfitting', 'regularization',
        'ensemble methods', 'bagging', 'boosting', 'stacking',
    ],

    'Deep Learning': [
        'deep learning', 'neural network', 'cnn', 'convolutional neural network',
        'rnn', 'lstm', 'gru', 'transformer', 'attention mechanism', 'self attention',
        'bert', 'gpt', 'vit', 'vision transformer', 'resnet', 'vgg', 'efficientnet',
        'gan', 'generative adversarial network', 'vae', 'variational autoencoder',
        'diffusion model', 'stable diffusion', 'backpropagation', 'gradient descent',
        'batch normalization', 'dropout', 'activation function', 'relu', 'sigmoid',
        'transfer learning', 'fine tuning', 'pretrained model', 'model compression',
        'quantization', 'pruning', 'knowledge distillation', 'cuda', 'gpu training',
    ],

    'Natural Language Processing': [
        'nlp', 'natural language processing', 'text classification', 'sentiment analysis',
        'named entity recognition', 'ner', 'part of speech', 'pos tagging',
        'tokenization', 'stemming', 'lemmatization', 'stopwords', 'tfidf', 'bag of words',
        'word2vec', 'glove', 'fasttext', 'embeddings', 'language model', 'llm',
        'bert', 'gpt', 'transformers', 'hugging face', 'spacy', 'nltk', 'gensim',
        'machine translation', 'text summarization', 'question answering', 'chatbot',
        'speech recognition', 'text to speech', 'information retrieval', 'semantic search',
        'topic modeling', 'lda', 'coreference resolution', 'dependency parsing',
    ],

    'Computer Vision': [
        'computer vision', 'image processing', 'opencv', 'object detection', 'yolo',
        'image segmentation', 'semantic segmentation', 'instance segmentation',
        'face recognition', 'facial detection', 'pose estimation', 'optical flow',
        'image classification', 'feature extraction', 'sift', 'orb', 'hog',
        'depth estimation', '3d reconstruction', 'point cloud', 'lidar',
        'video analysis', 'action recognition', 'tracking', 'kalman filter',
        'augmented reality', 'mixed reality', 'image generation', 'stable diffusion',
        'medical imaging', 'dicom', 'pathology', 'radiology ai',
    ],

    'Generative AI & LLMs': [
        'generative ai', 'llm', 'large language model', 'gpt', 'chatgpt', 'claude',
        'gemini', 'llama', 'mistral', 'prompt engineering', 'prompt design',
        'rag', 'retrieval augmented generation', 'vector database', 'embedding',
        'langchain', 'llamaindex', 'fine tuning', 'lora', 'qlora', 'instruction tuning',
        'reinforcement learning from human feedback', 'rlhf', 'constitutional ai',
        'ai agents', 'autonomous agents', 'tool use', 'function calling',
        'multimodal', 'vision language model', 'vlm', 'text to image', 'text to video',
        'ai hallucination', 'ai safety', 'responsible ai', 'model alignment',
    ],

    'MLOps & AI Engineering': [
        'mlops', 'ml engineering', 'model deployment', 'model serving', 'inference',
        'mlflow', 'kubeflow', 'airflow', 'feature store', 'data pipeline',
        'model monitoring', 'data drift', 'concept drift', 'model versioning',
        'a b testing model', 'canary deployment', 'shadow deployment',
        'bentoml', 'torchserve', 'triton inference', 'onnx', 'tensorrt',
        'model registry', 'experiment tracking', 'weights and biases', 'wandb',
        'dvc', 'data version control', 'model explainability', 'shap', 'lime',
    ],

    'Big Data & Data Engineering': [
        'big data', 'hadoop', 'spark', 'apache spark', 'pyspark', 'hive', 'hbase',
        'kafka', 'apache kafka', 'flink', 'apache flink', 'airflow', 'luigi',
        'etl', 'elt', 'data pipeline', 'data warehouse', 'data lake', 'data lakehouse',
        'snowflake', 'databricks', 'redshift', 'bigquery', 'azure synapse',
        'dbt', 'data build tool', 'data modeling', 'star schema', 'data vault',
        'streaming data', 'batch processing', 'real time analytics', 'lambda architecture',
        'delta lake', 'iceberg', 'parquet', 'avro', 'orc',
    ],

    'Databases & SQL': [
        'sql', 'database', 'mysql', 'postgresql', 'sqlite', 'oracle', 'sql server',
        'mongodb', 'cassandra', 'redis', 'elasticsearch', 'dynamodb', 'couchdb',
        'nosql', 'relational database', 'document database', 'graph database', 'neo4j',
        'database design', 'normalization', 'indexing', 'query optimization',
        'stored procedures', 'triggers', 'transactions', 'acid', 'cap theorem',
        'orm', 'sqlalchemy', 'hibernate', 'prisma', 'sequelize',
        'database administration', 'backup', 'replication', 'sharding',
    ],

    'Data Analytics & Business Intelligence': [
        'analytics', 'business intelligence', 'bi', 'tableau', 'power bi', 'looker',
        'google analytics', 'google data studio', 'qlik', 'metabase', 'superset',
        'kpi', 'dashboard', 'report', 'metrics', 'data storytelling',
        'excel analytics', 'pivot table', 'vlookup', 'power query', 'dax',
        'web analytics', 'funnel analysis', 'cohort analysis', 'retention analysis',
        'customer analytics', 'market basket analysis', 'rfm analysis',
    ],

    # ── Business & Management ─────────────────────────────────────────────────

    'Digital Marketing': [
        'seo', 'search engine optimization', 'sem', 'ppc', 'google ads', 'facebook ads',
        'social media marketing', 'content marketing', 'email marketing', 'copywriting',
        'branding', 'brand strategy', 'influencer marketing', 'affiliate marketing',
        'conversion rate optimization', 'cro', 'landing page', 'lead generation',
        'marketing funnel', 'customer journey', 'personas', 'target audience',
        'analytics', 'google analytics', 'growth hacking', 'viral marketing',
        'youtube marketing', 'instagram', 'tiktok marketing', 'linkedin marketing',
        'marketing automation', 'hubspot', 'mailchimp', 'crm marketing',
    ],

    'Finance & Accounting': [
        'finance', 'accounting', 'financial analysis', 'financial modeling',
        'stock', 'trading', 'investment', 'portfolio', 'equity', 'bonds', 'derivatives',
        'options', 'futures', 'forex', 'cryptocurrency trading', 'algorithmic trading',
        'excel finance', 'financial excel', 'tax', 'taxation', 'corporate finance',
        'valuation', 'dcf', 'discounted cash flow', 'financial statements',
        'balance sheet', 'income statement', 'cash flow', 'ratio analysis',
        'risk management', 'hedge', 'quantitative finance', 'fintech',
        'bookkeeping', 'quickbooks', 'xero', 'gaap', 'ifrs', 'audit',
    ],

    'Financial Analysis': [
        'financial analysis', 'quantitative analysis', 'quant', 'financial modeling',
        'excel modeling', 'scenario analysis', 'sensitivity analysis', 'monte carlo',
        'bloomberg terminal', 'capital markets', 'asset management', 'wealth management',
        'private equity', 'venture capital', 'mergers acquisitions', 'm&a',
        'credit analysis', 'credit risk', 'market risk', 'operational risk',
        'cfa', 'frm', 'cpa', 'chartered accountant', 'financial planning',
    ],

    'Project Management': [
        'project management', 'agile', 'scrum', 'kanban', 'waterfall', 'pmp',
        'prince2', 'lean', 'six sigma', 'project planning', 'gantt chart',
        'leadership', 'team management', 'stakeholder management', 'risk management',
        'jira', 'trello', 'asana', 'monday', 'confluence', 'notion',
        'entrepreneurship', 'startup', 'product management', 'roadmap',
        'sprint planning', 'retrospective', 'velocity', 'backlog grooming',
        'okr', 'kpi', 'change management', 'strategic planning',
    ],

    'Human Resources': [
        'hr', 'human resources', 'talent acquisition', 'recruiting', 'hiring',
        'onboarding', 'employee engagement', 'performance management',
        'payroll', 'compensation', 'benefits', 'labor law', 'compliance',
        'organizational development', 'training development', 'learning development',
        'diversity inclusion', 'dei', 'employee relations', 'hr analytics',
        'succession planning', 'workforce planning', 'hris', 'workday', 'sap hr',
    ],

    'Sales & Customer Success': [
        'sales', 'b2b sales', 'b2c sales', 'salesforce', 'crm', 'hubspot crm',
        'cold calling', 'lead generation', 'prospecting', 'pipeline management',
        'negotiation', 'closing deals', 'objection handling', 'sales funnel',
        'account management', 'customer success', 'customer retention',
        'upselling', 'cross selling', 'revenue operations', 'revops',
        'customer service', 'support', 'zendesk', 'intercom',
    ],

    'Supply Chain & Operations': [
        'supply chain', 'logistics', 'operations management', 'inventory management',
        'procurement', 'vendor management', 'erp', 'sap', 'oracle erp',
        'demand forecasting', 'warehouse management', 'last mile delivery',
        'lean manufacturing', 'six sigma', 'total quality management', 'tqm',
        'import export', 'customs', 'freight', 'incoterms',
        'operations research', 'linear programming', 'optimization',
    ],

    'Entrepreneurship & Strategy': [
        'entrepreneurship', 'startup', 'business model', 'lean startup', 'mvp',
        'business strategy', 'competitive analysis', 'porter five forces', 'swot',
        'market research', 'market analysis', 'go to market', 'product launch',
        'fundraising', 'pitch deck', 'investor', 'angel investor', 'venture capital',
        'bootstrapping', 'growth strategy', 'scaling', 'business development',
        'innovation', 'design thinking', 'business plan', 'feasibility study',
    ],

    # ── Arts, Design & Media ──────────────────────────────────────────────────

    'Graphic Design': [
        'graphic design', 'photoshop', 'illustrator', 'indesign', 'affinity designer',
        'figma', 'sketch', 'adobe xd', 'ui design', 'ux design', 'user experience',
        'typography', 'color theory', 'layout design', 'logo design', 'branding design',
        'print design', 'packaging design', 'poster design', 'infographic',
        'vector illustration', 'raster', 'digital illustration', 'procreate',
        'motion graphics', 'after effects', 'animation', 'lottie',
        'user research', 'wireframing', 'prototyping', 'usability testing',
    ],

    'Video Production & Editing': [
        'video editing', 'premiere pro', 'final cut pro', 'davinci resolve',
        'after effects', 'motion graphics', 'color grading', 'color correction',
        'videography', 'cinematography', 'camera work', 'lighting', 'audio production',
        'youtube channel', 'content creation', 'vlog', 'podcast production',
        'screencast', 'screen recording', 'obs studio', 'streaming', 'twitch',
        'visual effects', 'vfx', 'compositing', 'green screen', 'chroma key',
        '3d animation', 'blender', 'maya', 'cinema 4d',
    ],

    '3D Modeling & Animation': [
        '3d modeling', 'blender', 'maya', 'cinema 4d', '3ds max', 'zbrush',
        'character modeling', 'hard surface modeling', 'sculpting', 'retopology',
        'uv mapping', 'texturing', 'pbr materials', 'substance painter',
        'rigging', 'skinning', 'animation', 'keyframe', 'motion capture',
        '3d printing', 'cad', 'autocad', 'solidworks', 'fusion 360',
        'rendering', 'cycles', 'eevee', 'arnold', 'vray', 'redshift',
        'vr ar content', 'game ready assets',
    ],

    'Music & Audio Production': [
        'music production', 'daw', 'ableton', 'fl studio', 'logic pro', 'pro tools',
        'mixing', 'mastering', 'sound design', 'synthesis', 'sampling',
        'music theory', 'beat making', 'hip hop production', 'electronic music',
        'audio engineering', 'recording', 'studio recording', 'podcast',
        'foley', 'sound effects', 'film scoring', 'composition',
        'midi', 'vst plugins', 'audio fx', 'eq', 'compression',
    ],

    'Photography': [
        'photography', 'lightroom', 'photoshop retouching', 'camera settings',
        'exposure', 'aperture', 'shutter speed', 'iso', 'composition',
        'portrait photography', 'landscape photography', 'product photography',
        'street photography', 'wildlife photography', 'drone photography',
        'photo editing', 'photo retouching', 'color grading photography',
        'studio lighting', 'natural light', 'flash photography',
    ],

    # ── Health, Science & Education ───────────────────────────────────────────

    'Healthcare & Medicine': [
        'medical', 'healthcare', 'clinical', 'nursing', 'pharmacology',
        'anatomy', 'physiology', 'pathology', 'diagnosis', 'treatment',
        'ehr', 'electronic health records', 'telemedicine', 'health informatics',
        'medical imaging', 'radiology', 'mri', 'ct scan', 'ultrasound',
        'public health', 'epidemiology', 'biostatistics', 'clinical trials',
        'mental health', 'psychiatry', 'psychology', 'therapy', 'counseling',
        'nutrition', 'dietetics', 'fitness', 'rehabilitation',
    ],

    'Bioinformatics & Computational Biology': [
        'bioinformatics', 'genomics', 'proteomics', 'transcriptomics',
        'sequence alignment', 'blast', 'bwa', 'bowtie', 'samtools',
        'variant calling', 'snp', 'gwas', 'gene expression', 'rna seq',
        'scrnaseq', 'single cell', 'metagenomics', 'phylogenetics',
        'r bioconductor', 'biopython', 'structural biology', 'molecular docking',
        'drug discovery', 'cheminformatics', 'rdkit', 'protein structure',
        'alphafold', 'cryo em', 'flow cytometry',
    ],

    'Data Science in Healthcare': [
        'health data', 'clinical data', 'ehr data', 'health analytics',
        'predictive modeling healthcare', 'survival analysis', 'clinical nlp',
        'medical image analysis', 'radiology ai', 'pathology ai',
        'drug discovery ai', 'genomic data science', 'real world evidence',
        'population health', 'risk stratification', 'readmission prediction',
    ],

    'Physics & Engineering Sciences': [
        'physics', 'mechanics', 'thermodynamics', 'electromagnetism', 'optics',
        'quantum mechanics', 'quantum computing', 'qiskit', 'quantum algorithms',
        'signal processing', 'dsp', 'fourier transform', 'filter design',
        'control systems', 'pid controller', 'matlab', 'simulink',
        'finite element analysis', 'fea', 'computational fluid dynamics', 'cfd',
        'materials science', 'nanotechnology', 'photonics',
    ],

    'Environmental Science & Sustainability': [
        'sustainability', 'climate change', 'carbon footprint', 'renewable energy',
        'solar energy', 'wind energy', 'energy storage', 'battery technology',
        'environmental monitoring', 'remote sensing', 'gis', 'geospatial',
        'green building', 'leed', 'circular economy', 'life cycle assessment',
        'esg', 'environmental reporting', 'carbon market',
    ],

    # ── Soft Skills & Professional Development ────────────────────────────────

    'Soft Skills & Communication': [
        'communication', 'public speaking', 'presentation skills', 'negotiation',
        'conflict resolution', 'active listening', 'interpersonal skills',
        'emotional intelligence', 'eq', 'empathy', 'teamwork', 'collaboration',
        'time management', 'productivity', 'goal setting', 'habit building',
        'critical thinking', 'problem solving', 'creativity', 'innovation mindset',
        'writing skills', 'business writing', 'technical writing', 'storytelling',
        'interview preparation', 'resume writing', 'linkedin profile', 'job search',
        'networking', 'personal branding', 'career development', 'mentoring',
    ],

    'Language Learning': [
        'english', 'ielts', 'toefl', 'grammar', 'vocabulary', 'pronunciation',
        'spanish', 'french', 'german', 'arabic', 'chinese', 'japanese', 'korean',
        'business english', 'academic writing', 'esl', 'language fluency',
        'translation', 'interpretation', 'linguistics',
    ],

    'Teaching & Education': [
        'curriculum design', 'instructional design', 'elearning', 'lms',
        'moodle', 'canvas', 'google classroom', 'online teaching', 'blended learning',
        'assessment design', 'rubric', 'differentiated instruction',
        'stem education', 'coding for kids', 'educational technology', 'edtech',
        'higher education', 'k12', 'corporate training', 'learning development',
    ],

    # ── Emerging & Specialized Fields ─────────────────────────────────────────

    'Robotics & Automation': [
        'robotics', 'ros', 'robot operating system', 'autonomous systems',
        'computer vision robotics', 'slam', 'localization', 'mapping',
        'motion planning', 'inverse kinematics', 'robot arm', 'collaborative robot',
        'industrial automation', 'plc', 'scada', 'process automation',
        'rpa', 'robotic process automation', 'uipath', 'automation anywhere', 'blue prism',
        'drone', 'uav', 'autonomous vehicle', 'self driving', 'lidar',
    ],

    'AR & VR Development': [
        'augmented reality', 'virtual reality', 'mixed reality', 'xr', 'ar', 'vr',
        'unity xr', 'unreal vr', 'arkit', 'arcore', 'hololens', 'meta quest',
        'openxr', 'webxr', 'spatial computing', 'immersive experience',
        '360 video', 'vr training', 'ar marketing',
    ],

    'Legal & Compliance': [
        'legal', 'contract law', 'intellectual property', 'copyright', 'trademark',
        'patent', 'gdpr', 'privacy law', 'data protection', 'compliance',
        'regulatory', 'corporate law', 'employment law', 'business law',
        'legal writing', 'legal research', 'paralegal', 'legaltech',
    ],

    'Real Estate': [
        'real estate', 'property investment', 'real estate investing', 'rental property',
        'house flipping', 'commercial real estate', 'property management',
        'real estate finance', 'mortgage', 'property valuation', 'appraisal',
        'real estate development', 'urban planning', 'zoning',
    ],
}

def get_subject(title):
    title = str(title).lower()
    for subject, keywords in subject_keywords.items():
        if any(word in title for word in keywords):
            return subject
    return 'Other'


# =========================
# 5. FILL MISSING VALUES
# =========================
def fill_missing_values(df):

    # numeric columns
    num_cols = ['rating', 'no_students', 'content_duration']

    for col in num_cols:
        if col in df.columns:
            df[col] = df.groupby(['subject', 'difficulty_level'])[col].transform(
                lambda x: x.fillna(
                    x.median() if not x.isnull().all()
                    else df[col].median()
                )
            )

    # categorical column
    if 'certificate_type' in df.columns:
        df['certificate_type'] = df.groupby(['subject', 'difficulty_level'])['certificate_type'].transform(
            lambda x: x.fillna(
                x.mode()[0] if not x.mode().empty
                else df['certificate_type'].mode()[0]
            )
        )

    return df


# =========================
# 6. MAIN PIPELINE
# =========================
def preprocess_pipeline():

    # load
    coursera_data, edx_data, udemy_data = load_data()

    # datatype fix
    udemy_data, coursera_data = convert_datatype(udemy_data, coursera_data)

    # drop cols
    coursera_data, edx_data, udemy_data = drop_columns(coursera_data, edx_data, udemy_data)

    # rename + merge
    courses_data = rename_columns(coursera_data, edx_data, udemy_data)

    # drop unnecessary
    courses_data = courses_data.drop(['url'], axis=1, errors='ignore')

    # subject column
    courses_data['subject'] = courses_data['title'].apply(get_subject)

    # fill missing
    courses_data = fill_missing_values(courses_data)

    return courses_data


# =========================
# 7. RUN PIPELINE
# =========================
courses_data = preprocess_pipeline()

# =========================
# 8. SAVE (IMPORTANT)
# =========================
courses_data.to_csv('/content/drive/MyDrive/AirForge_Data/clean_courses.csv', index=False)

print("✅ Data preprocessing complete and saved!")