# Magento Product Recommender AI - Backend

This is the backend service for the Magento Product Recommender AI application. It uses FastAPI to provide a REST API for product recommendations based on similarity analysis.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /recommend` - Get product recommendations based on an item ID
  - Parameters:
    - `item_id`: The ID of the item to find recommendations for (required)
    - `limit`: Number of recommendations to return (default: 5)

## Configuration

- Update `recommendor/config.py` to point to your price list CSV file.
- The CSV file should have columns: 'id', 'description', 'name', and 'price'.