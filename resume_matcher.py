"""
Redrob AI Campus Hackathon — Resume Matching Engine
"""

import math

# ─────────────────────────────────────────────
# SKILL_ALIASES  (exact as provided)
# ─────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    # Web — Frontend
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql",
    "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android", "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# ─────────────────────────────────────────────
# RESUME DATASET
# ─────────────────────────────────────────────
RESUMES = [
    {"id": "01", "name": "Arjun Sharma",    "raw": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "name": "Priya Nair",      "raw": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "name": "Rahul Gupta",     "raw": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "name": "Sneha Patel",     "raw": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "name": "Vikram Singh",    "raw": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "name": "Ananya Krishnan", "raw": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "name": "Karan Mehta",     "raw": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "name": "Deepika Rao",     "raw": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "name": "Aditya Kumar",    "raw": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "name": "Meera Iyer",      "raw": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

# ─────────────────────────────────────────────
# JOB DESCRIPTIONS
# ─────────────────────────────────────────────
JDS = [
    {
        "id": "JD-1", "company": "Kakao (Seoul)", "role": "ML Engineer",
        "required": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
        "preferred": "NLP, BERT, Feature Engineering, Statistics",
    },
    {
        "id": "JD-2", "company": "Naver (Seongnam)", "role": "Backend Engineer",
        "required": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
        "preferred": "REST API, CI/CD, Redis",
    },
    {
        "id": "JD-3", "company": "Line (Seoul)", "role": "Frontend Engineer",
        "required": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "preferred": "Node.js, GraphQL, Redux, Jest, AWS",
    },
]


# ─────────────────────────────────────────────
# STEP 1 & 2: Normalize + Deduplicate skills
# ─────────────────────────────────────────────
def normalize_skills(raw: str) -> list:
    """
    1. Split on commas.
    2. Lowercase and strip each token.
    3. Apply exact alias mapping using SKILL_ALIASES.
    4. Discard tokens not found in the alias map.
    5. Deduplicate while preserving order.
    """
    tokens = [token.strip().lower() for token in raw.split(",")]

    seen = set()
    normalized = []

    for token in tokens:
        if token in SKILL_ALIASES:
            canonical = SKILL_ALIASES[token]

            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)

    return normalized


# ─────────────────────────────────────────────
# STEP 3: Build shared vocabulary
# ─────────────────────────────────────────────
def build_vocabulary(normalized_resumes: list) -> list:
    """Sorted alphabetical vocabulary from all resume skills."""
    vocab_set = set()
    for r in normalized_resumes:
        vocab_set.update(r["skills"])
    return sorted(vocab_set)


# ─────────────────────────────────────────────
# STEP 4: TF-IDF vectors for resumes
# ─────────────────────────────────────────────
def compute_tfidf(skills: list, vocab: list, df: dict, total_docs: int) -> list:
    """
    TF  = 1 / N  (N = number of unique skills in this resume, after dedup)
    IDF = ln(total_docs / df[skill])   — no smoothing
    TF-IDF = TF * IDF   (0 if skill absent)
    """
    n = len(skills)
    vec = []
    for skill in vocab:
        if skill in skills:
            tf = 1 / n
            idf = math.log(total_docs / df[skill])
            vec.append(tf * idf)
        else:
            vec.append(0.0)
    return vec


# ─────────────────────────────────────────────
# STEP 5: JD binary vectors
# ─────────────────────────────────────────────
def build_jd_vector(jd: dict, vocab: list) -> list:
    """Binary vector over shared vocabulary."""
    combined = jd["required"] + ", " + jd["preferred"]
    jd_skills = normalize_skills(combined)
    return [1 if skill in jd_skills else 0 for skill in vocab]


# ─────────────────────────────────────────────
# STEP 6: Cosine similarity
# ─────────────────────────────────────────────
def cosine_similarity(a: list, b: list) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    TOTAL_DOCS = len(RESUMES)  # 10

    # --- Normalize resumes ---
    normalized = []
    for r in RESUMES:
        skills = normalize_skills(r["raw"])
        normalized.append({**r, "skills": skills})

    # --- Build vocabulary ---
    vocab = build_vocabulary(normalized)
    print(f"\n{'='*60}")
    print(f"Vocabulary ({len(vocab)} terms): {vocab}")

    # --- Document frequency ---
    df = {skill: sum(1 for r in normalized if skill in r["skills"]) for skill in vocab}

    # --- TF-IDF vectors ---
    print(f"\n{'='*60}")
    print("Normalized resume skills:")
    for r in normalized:
        print(f"  {r['id']} {r['name']:20s}: {r['skills']}")

    resume_vectors = []
    for r in normalized:
        vec = compute_tfidf(r["skills"], vocab, df, TOTAL_DOCS)
        resume_vectors.append({**r, "vec": vec})

    # --- JD binary vectors ---
    jd_vectors = []
    for jd in JDS:
        vec = build_jd_vector(jd, vocab)
        jd_skills = normalize_skills(jd["required"] + ", " + jd["preferred"])
        jd_vectors.append({**jd, "vec": vec, "skills": jd_skills})

    print(f"\n{'='*60}")
    print("JD skills after normalization:")
    for jd in jd_vectors:
        print(f"  {jd['id']}: {jd['skills']}")

    # --- Compute similarities & rank ---
    print(f"\n{'='*60}")
    print("RESULTS\n")
    for jd in jd_vectors:
        scores = []
        for r in resume_vectors:
            sim = cosine_similarity(r["vec"], jd["vec"])
            scores.append((r["name"], sim))

        # Sort: descending score, then ascending name (alphabetical tie-break)
        scores.sort(key=lambda x: (-x[1], x[0]))
        top3 = scores[:3]

        label = f"{jd['id']} — {jd['company']} ({jd['role']})"
        result = ", ".join(f"{name}({score:.2f})" for name, score in top3)
        print(label)
        print(result)
        print()


if __name__ == "__main__":
    main()
