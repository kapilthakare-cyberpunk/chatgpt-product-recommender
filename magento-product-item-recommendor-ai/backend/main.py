from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from recommendor.core import get_recommendations

app = FastAPI(title="Magento Product Item Recommendor AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend_products(
    item_id: int = Query(..., description="Magento Item ID"),
    limit: int = Query(5, description="Number of recommendations to return")
):
    results = get_recommendations(item_id=item_id, limit=limit)
    return {"item_id": item_id, "recommendations": results}