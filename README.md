# 🎬 Hybrid Movie Recommendation System (Production-Ready)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Ranking-orange)
![ALS](https://img.shields.io/badge/ALS-Collaborative_Filtering-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-scale hybrid recommendation system using **ALS (Matrix Factorization)** and **XGBoost Learning-to-Rank**, deployed via **FastAPI** and **Streamlit**.

Handles **33M+ ratings** and provides real-time personalized recommendations.

---

# 🚀 Live System Architecture

```
            User
             │
             ▼
      Streamlit Frontend
             │
             ▼
        FastAPI Backend
             │
             ▼
  Hybrid Recommendation Engine
       │              │
       ▼              ▼
   ALS Model     XGBoost Ranker
       │              │
       └──────► Final Recommendations
```

---

# 📊 Dataset

MovieLens Latest Dataset

* 33,832,162 ratings
* 330,975 users
* 83,239 movies
* Time span: 1995–2023
* Sparsity: 99.88%

---

# 🧠 Machine Learning Pipeline

## Stage 1 — Candidate Generation (ALS)

Matrix factorization learns latent embeddings:

```
User Embedding Vector
Movie Embedding Vector
```

Used to generate candidate movies.

---

## Stage 2 — Feature Engineering

Features include:

* ALS score
* Embedding similarity
* User-movie interaction signals

---

## Stage 3 — Ranking Model (XGBoost)

XGBoost predicts probability:

```
P(user likes movie)
```

Used to rank candidate recommendations.

---

## Stage 4 — Hybrid Recommendation

Final pipeline:

```
ALS → Candidate Movies
      ↓
Feature Engineering
      ↓
XGBoost Ranking
      ↓
Final Recommendations
```

---

# 📈 Evaluation

Metric: Precision@10

Result:

```
Precision@10 = 0.2044
```

This is considered excellent performance for recommendation systems.

---

# 🏗️ Project Structure

```
movie-recommendation-advanced/
│
├── api/
│   ├── main.py
│   ├── routes/
│   └── services/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── models/
│   ├── features/
│   ├── evaluation/
│   └── inference/
│
├── models/
├── data/
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone repository:

```
git clone https://github.com/tejfaster/My-Movies.git

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

# ▶️ Run Backend API

```
uvicorn api.main:app --reload
```

API endpoint:

```
http://127.0.0.1:8000/recommend/149954
```

---

# ▶️ Run Frontend UI

```
streamlit run app/streamlit_app.py
```

Open:

```
http://localhost:8501
```

---

# 📡 API Example

Request:

```
GET /recommend/149954?n=10
```

Response:

```
{
  "user_id": 149954,
  "recommendations": [
    {
      "movie": "Jurassic Park (1993)",
      "score": 0.91
    }
  ]
}
```

---

# 🛠️ Technologies Used

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

# 💡 Key ML Concepts Demonstrated

* Collaborative Filtering
* Matrix Factorization
* Embedding Learning
* Learning-to-Rank
* Hybrid Recommendation Systems
* Production ML Deployment

---

# 🎯 Production Features

* Modular architecture
* Real-time inference API
* Interactive frontend
* Scalable ML pipeline
* Deployment-ready code

---

# 🔮 Future Improvements

* Cloud deployment (AWS / Render)
* Real-time user feedback integration
* Neural recommendation models
* Cold-start handling

---

# 👨‍💻 Author

Tej Pratap

Machine Learning Engineer Project

---

# 📜 License

MIT License
