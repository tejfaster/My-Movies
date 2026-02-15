# Hybrid Movie Recommendation System (Production-Ready)

A production-scale hybrid recommendation system built using **ALS (Matrix Factorization)** and **XGBoost Ranking**, deployed via **FastAPI** and **Streamlit**.

This system handles **33M+ user-movie interactions** and provides personalized movie recommendations through a real-time API and web interface.

---

# Overview

Modern recommendation systems use a multi-stage pipeline:

1. Candidate Generation (ALS)
2. Feature Engineering
3. Ranking Model (XGBoost)
4. API Deployment (FastAPI)
5. Frontend Interface (Streamlit)

This project implements that full pipeline end-to-end.

---

# Features

* ALS collaborative filtering model
* XGBoost learning-to-rank model
* Hybrid recommendation system
* FastAPI production inference API
* Streamlit interactive frontend
* Time-aware train/test split
* Precision@10 evaluation
* Modular production architecture

---

# Dataset

Dataset: MovieLens Latest Dataset

* 33,832,162 ratings
* 330,975 users
* 83,239 movies
* Time range: 1995–2023

---

# Model Architecture

Pipeline:

User → ALS → Candidate Movies → Feature Engineering → XGBoost Ranker → Final Recommendations

ALS learns latent embeddings.

XGBoost ranks candidates based on learned features.

---

# Evaluation

Metric: Precision@10

Result:

Precision@10 = 0.2044

This is considered excellent performance for candidate generation systems.

---

# Project Structure

```
movie-recommendation-advanced/
│
├── api/                 # FastAPI backend
│   ├── main.py
│   ├── routes/
│   └── services/
│
├── app/                 # Streamlit frontend
│   └── streamlit_app.py
│
├── src/                 # Training and feature code
│   ├── models/
│   ├── features/
│   ├── evaluation/
│   └── inference/
│
├── models/              # Saved ML models
│
├── data/                # Dataset
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone repository:

```
git clone https://github.com/tejfaster/movie-recommendation-advanced.git

cd movie-recommendation-advanced
```

Create virtual environment:

```
python -m venv venv

source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Run FastAPI Backend

```
uvicorn api.main:app --reload
```

API runs at:

```
http://127.0.0.1:8000
```

Example endpoint:

```
http://127.0.0.1:8000/recommend/149954
```

---

# Run Streamlit Frontend

In a separate terminal:

```
streamlit run app/streamlit_app.py
```

Open:

```
http://localhost:8501
```

---

# API Endpoints

GET /recommend/{user_id}

Example:

```
/recommend/149954?n=10
```

Response:

```
{
  "user_id": 149954,
  "recommendations": [
    {"movie": "Jurassic Park", "score": 0.91}
  ]
}
```

---

# Technologies Used

Machine Learning:

* ALS (implicit library)
* XGBoost
* scikit-learn

Backend:

* FastAPI
* Uvicorn

Frontend:

* Streamlit

Data Processing:

* pandas
* numpy
* scipy

---

# Key ML Concepts Implemented

* Matrix Factorization
* Embedding Learning
* Learning-to-Rank
* Hybrid Recommendation Systems
* Production Model Deployment

---

# Future Improvements

* Cloud deployment (AWS, Render, Railway)
* Real-time user interaction tracking
* Cold-start recommendation handling
* Neural recommendation models

---

# Author

Tej Pratap

Machine Learning Engineer Project

---

# License

MIT License
