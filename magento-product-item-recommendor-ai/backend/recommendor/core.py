import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import PRICE_LIST_PATH, MIN_SIMILARITY_THRESHOLD
from .utils import load_pricelist

def get_recommendations(item_id: int, limit: int = 5):
    df = load_pricelist(PRICE_LIST_PATH)

    if "id" not in df.columns or "description" not in df.columns:
        raise Exception("CSV must contain at least 'id' and 'description' columns")

    if item_id not in df["id"].values:
        raise Exception(f"Item ID {item_id} not found in the price list")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["description"].fillna(""))

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = df.index[df["id"] == item_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, score in sim_scores[1:] if score > MIN_SIMILARITY_THRESHOLD][:limit]

    return df.iloc[top_indices][["id", "name", "price"]].to_dict(orient="records")