# 🎬 AI Movie Intelligence Platform

### Production-Grade Recommendation System & Investment Decision Engine

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Ranking-orange)]()
[![ALS](https://img.shields.io/badge/ALS-Recommendation-purple)]()
[![Scale](https://img.shields.io/badge/Data-7.9M%20Relationships-black)]()

---

# 🚀 Project Summary

This project builds a **production-grade AI Movie Intelligence Platform** that solves two real-world business problems:

### 1. User Intelligence (Recommendation System)

Recommend personalized movies using collaborative filtering and ranking models.

### 2. Investment Intelligence (Business Decision Engine)

Identify which actors and genres studios should invest in to maximize success.

This system processes **millions of interactions**, builds advanced features, and serves predictions via a production API.

---

# 🎯 Business Problems Solved

## Problem 1 — Movie Recommendation

Platforms need to answer:

> What movie should this user watch next?

Solution:

```
ALS Candidate Generation → XGBoost Ranking → Final Recommendations
```

---

## Problem 2 — Movie Investment Decision

Studios need to answer:

> Which actor should we invest in?
> Which genre attracts the most audience?

Solution:

```
Large-scale actor intelligence using IMDb + MovieLens
```

---

# 📊 Dataset Scale

| Dataset                   | Size       |
| ------------------------- | ---------- |
| MovieLens Ratings         | 33,832,162 |
| Users                     | 330,975    |
| Movies                    | 83,239     |
| Actor-Movie Relationships | 7,920,568  |
| Actors analyzed           | 478,352    |

This is production-scale data.

---

# 🧠 System Architecture

```
                Data Layer
        ┌────────────────────────┐
        │ MovieLens Dataset      │
        │ IMDb Dataset           │
        └────────────────────────┘
                    │
                    ▼
            Feature Engineering
        ┌────────────────────────┐
        │ User Features          │
        │ Actor Features         │
        │ Genre Features         │
        │ Engagement Features    │
        └────────────────────────┘
                    │
                    ▼
                ML Models
        ┌────────────────────────┐
        │ ALS Recommendation     │
        │ XGBoost Ranking       │
        │ Actor Intelligence     │
        │ Investment Scoring     │
        └────────────────────────┘
                    │
                    ▼
              FastAPI Backend
                    │
                    ▼
           Streamlit Dashboard
```

---

# 🤖 Machine Learning Models

## Recommendation Model

Models used:

* ALS (Collaborative Filtering)
* XGBoost Ranking Model

Pipeline:

```
User → ALS → Candidate Movies → XGBoost → Ranked Recommendations
```

Performance:

```
Precision@10: 0.2044
```

---

## Investment Intelligence Model

Analyzes actors using:

* Role importance weighting
* Audience engagement signals
* Career longevity analysis
* Recency scoring

Investment Score Formula:

```
score =
  0.30 × audience demand
+ 0.25 × audience trust
+ 0.20 × rating quality
+ 0.15 × recency relevance
+ 0.10 × experience
```

---

# ⭐ Example Results

Top Investment Actors:

```
Tom Hanks
Brad Pitt
Tom Cruise
Leonardo DiCaprio
Harrison Ford
```

These rankings match real industry value.

---

# 🏗️ Production Features

✔ FastAPI backend
✔ Streamlit dashboard
✔ ML model serving
✔ Feature engineering pipeline
✔ Large-scale dataset processing
✔ Modular architecture

---

# 📡 API Example

Recommendation Endpoint:

```
GET /recommend/{user_id}
```

Response:

```json
{
  "recommendations": [
    "The Dark Knight",
    "Inception",
    "Interstellar"
  ]
}
```

---

# 💻 Tech Stack

Machine Learning:

```
Python
ALS (implicit)
XGBoost
scikit-learn
```

Backend:

```
FastAPI
Uvicorn
```

Frontend:

```
Streamlit
```

Data Processing:

```
pandas
numpy
scipy
```

---

# 📂 Project Structure

```
movie-intelligence-platform/

├── api/                FastAPI backend
├── app/                Streamlit frontend
├── src/
│   ├── recommendation/
│   ├── investment/
│   ├── features/
│   └── models/
│
├── notebooks/
├── data/
├── models/
├── requirements.txt
└── README.md
```

---

# ▶️ Run Locally

Backend:

```
uvicorn api.main:app --reload
```

Frontend:

```
streamlit run app/streamlit_app.py
```

---

# 💼 Skills Demonstrated

This project demonstrates:

* Recommender Systems
* Ranking Models
* Feature Engineering at Scale
* ML Model Deployment
* FastAPI Production Backend
* Data Pipeline Design
* Large-Scale Data Processing

---

# 🎯 Real-World Applications

Used by:

* Streaming platforms (Netflix-like)
* Movie studios
* Investment firms
* Production companies

---

# 👨‍💻 Author

Tej Pratap
Machine Learning Engineer

---

# 📜 License

MIT License
