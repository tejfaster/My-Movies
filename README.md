# 🎬 AI Movie Intelligence Platform

### Recommendation System + Investment Decision Engine

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Ranking-orange)
![ALS](https://img.shields.io/badge/ALS-Recommendation-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 🚀 Overview

This project is a **production-ready AI Movie Intelligence Platform** that combines:

* 🎯 Personalized Movie Recommendation System
* 💰 Movie Investment Decision Support System

It helps:

* Users discover movies
* Studios choose profitable actors
* Investors decide which movies to fund

Built using **ALS, XGBoost, FastAPI, and Streamlit**, handling millions of interactions.

---

# 🧠 System Capabilities

## 1. Recommendation Engine (User Intelligence)

Predicts movies users will like.

Pipeline:

```
User → ALS Candidate Generation → XGBoost Ranking → Final Recommendations
```

Endpoint:

```
/recommend/{user_id}
```

---

## 2. Investment Intelligence Engine (Business Intelligence)

Predicts which actors, genres, and movies are profitable.

Capabilities:

* Actor profitability prediction
* Movie revenue prediction
* Genre profitability analysis
* Investment risk assessment

Example output:

```
Best Actor Investment:
Zendaya → Predicted revenue impact: $520M
Confidence: 87%
Risk: Low
```

---

# 🏗️ Unified System Architecture

```
                    Data Sources
         ┌────────────────────────────┐
         │ MovieLens Dataset          │
         │ TMDB Dataset               │
         └────────────────────────────┘
                      │
                      ▼
              Feature Engineering Layer
      ┌────────────────────────────────────┐
      │ User Features                      │
      │ Movie Features                     │
      │ Actor Features                     │
      │ Genre Features                     │
      └────────────────────────────────────┘
                      │
                      ▼
                  Model Layer
      ┌────────────────────────────────────┐
      │ ALS Recommendation Model           │
      │ XGBoost Ranking Model              │
      │ Revenue Prediction Model           │
      │ Actor Profitability Model          │
      └────────────────────────────────────┘
                      │
                      ▼
                 FastAPI Backend
                      │
                      ▼
              Streamlit Dashboard UI
```

---

# 📊 Datasets Used

## MovieLens Dataset

* 33,832,162 ratings
* 330,975 users
* 83,239 movies

Used for:

* User recommendation training

## TMDB Dataset

Contains:

* Actors
* Revenue
* Budget
* Popularity
* Genres

Used for:

* Investment prediction

---

# 📈 Model Performance

Recommendation Model:

```
Precision@10: 0.2044
```

This is considered excellent performance.

---

# ⚙️ Tech Stack

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

# 📂 Project Structure

```
movie-intelligence-platform/
│
├── api/                # FastAPI backend
│
├── app/                # Streamlit frontend
│
├── src/
│   ├── recommendation/
│   ├── investment/
│   ├── features/
│   └── models/
│
├── models/
│
├── data/
│   ├── movielens/
│   └── tmdb/
│
├── requirements.txt
└── README.md
```

---

# ▶️ Run Backend

```
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# ▶️ Run Frontend

```
streamlit run app/streamlit_app.py
```

Open:

```
http://localhost:8501
```

---

# 📡 API Endpoints

Recommendation:

```
GET /recommend/{user_id}
```

Actor Investment Analysis:

```
GET /actor-profitability/{actor}
```

Revenue Prediction:

```
POST /predict-revenue
```

---

# 💡 Real-World Applications

This system can be used by:

* Streaming platforms
* Movie studios
* Film investors
* Production companies

To answer questions like:

* Which actor should I cast?
* Which genre is profitable?
* Will this movie succeed?

---

# 🔮 Future Improvements

* Cloud deployment (AWS / Render)
* Neural recommendation models
* Real-time retraining pipeline
* Investment risk modeling

---

# 👨‍💻 Author

Tej Pratap
Machine Learning Engineer Project

---

# 📜 License

MIT License
