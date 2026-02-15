from fastapi import FastAPI
from api.routes.recommend import router as recommend_router

app = FastAPI(title="Movie Recommendation API")

@app.get("/")
def root():
    return {"message":"API running successfully"}

app.include_router(recommend_router)