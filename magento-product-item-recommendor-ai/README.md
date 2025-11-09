# Magento Product Item Recommender AI

An AI-powered product recommendation engine that connects to Magento price lists to provide similar product recommendations based on item descriptions.

## Quick Start

The recommended way to start the application is using our interactive setup script:

```bash
# Make sure the script is executable
chmod +x setup.sh

# Run the interactive setup
./setup.sh
```

The script will:
1. Check for and optionally stop existing instances
2. Start the backend server on port 8000
3. Start the frontend server on port 3000
4. Verify everything is working

## Manual Start

If you prefer to start the servers manually:

### Backend (FastAPI)

```bash
cd backend
source venv/bin/activate  # Activate virtual environment
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`.

### Frontend (Next.js)

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## Usage

1. Enter a Magento item ID in the input field (try IDs 1-15 from the sample data)
2. Click "Fetch" to get recommendations
3. The recommended products will be displayed below the form

## API Endpoint

- `GET /recommend` - Get product recommendations based on an item ID
  - Parameters:
    - `item_id`: The ID of the item to find recommendations for (required)
    - `limit`: Number of recommendations to return (default: 5)

Example: `http://localhost:8000/recommend?item_id=1&limit=3`

## Configuration

- Update `backend/recommendor/config.py` to change the path to your price list CSV
- Adjust `MIN_SIMILARITY_THRESHOLD` in the same file to control recommendation similarity

## Management

- To stop all servers: `pkill -f "uvicorn main:app" && pkill -f "npm run dev"`
- Backend logs: `tail -f backend/backend.log`
- Frontend logs: `tail -f frontend/frontend.log`

## Sample Data

The application comes with sample data in `data/pricelist.csv` containing 15 products including:
- Cameras (Nikon, Canon, Sony)
- Lenses
- Video equipment
- Photo/video editing software
- Accessories

## Architecture

This application consists of two main components:

1. **Backend** - A FastAPI application that processes recommendations using scikit-learn
2. **Frontend** - A Next.js application that provides a user interface to interact with the recommendation engine