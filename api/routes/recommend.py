from fastapi import APIRouter
from api.services.recommender import recommend_movies

router = APIRouter()

@router.get("/recommend/{user_id}")
def recommend(user_id: int,n: int = 10):

    results = recommend_movies(user_id,n)

    return {
        "user_id": user_id,
        "recommendations": results
    }