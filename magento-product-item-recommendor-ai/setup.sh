#!/bin/bash

# Magento Product Recommender AI - Interactive Setup Script
# This script helps you start the backend and frontend servers

PROJECT_ROOT="/home/kapilt/Projects/pnz-projects/chatgpt-product-recommender/magento-product-item-recommendor-ai"

echo "Magento Product Recommender AI - Interactive Setup"
echo "=================================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a process is running
is_running() {
    local port=$1
    lsof -Pi :$port -sTCP:LISTEN -t >/dev/null
}

# Function to start the backend server
start_backend() {
    echo "Starting backend server (FastAPI) on port 8000..."
    cd "$PROJECT_ROOT/backend" || { echo "Error: Backend directory not found"; exit 1; }
    
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Error: Virtual environment not found. Please run setup first."
        exit 1
    fi
    
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "Backend server started with PID: $BACKEND_PID"
    
    # Wait a moment for the server to start
    sleep 3
    
    # Verify the backend is running
    if is_running 8000; then
        echo "✓ Backend server is running on http://localhost:8000"
    else
        echo "✗ Backend server failed to start. Check backend.log for details."
        exit 1
    fi
}

# Function to start the frontend server
start_frontend() {
    echo "Starting frontend server (Next.js) on port 3000..."
    cd "$PROJECT_ROOT/frontend" || { echo "Error: Frontend directory not found"; exit 1; }
    
    nohup npm run dev > frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "Frontend server started with PID: $FRONTEND_PID"
}

# Function to check if servers are already running
check_running() {
    echo "Checking for running instances..."
    
    if is_running 8000; then
        echo "⚠️  Backend server is already running on port 8000"
        read -p "Stop existing backend? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pkill -f "uvicorn main:app" 2>/dev/null || true
            echo "Stopped existing backend server"
        else
            echo "Using existing backend server"
        fi
    fi
    
    if is_running 3000; then
        echo "⚠️  Frontend server is already running on port 3000"
        read -p "Stop existing frontend? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pkill -f "npm run dev" 2>/dev/null || true
            echo "Stopped existing frontend server"
        else
            echo "Using existing frontend server"
        fi
    fi
}

# Function to test the backend API
test_backend_api() {
    echo ""
    echo "Testing backend API..."
    if curl -s "http://localhost:8000/docs" | grep -q "Swagger UI"; then
        echo "✓ Backend API is responding correctly"
        echo "  API Documentation: http://localhost:8000/docs"
        echo "  Health check: http://localhost:8000/"
    else
        echo "✗ Backend API is not responding"
        echo "  Please check backend logs and configuration"
        return 1
    fi
}

# Main execution
echo "This script will:"
echo "1. Check for and optionally stop existing instances"
echo "2. Start the backend server on port 8000"
echo "3. Start the frontend server on port 3000"
echo "4. Verify everything is working"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Check for existing instances
check_running

# Start backend if not already running
if ! is_running 8000; then
    start_backend
else
    echo "Using existing backend server on port 8000"
fi

# Wait for backend to be fully ready
echo "Waiting for backend to be ready..."
sleep 5

# Test backend API
if ! test_backend_api; then
    echo "Backend API test failed. Exiting."
    exit 1
fi

# Start frontend if not already running
if ! is_running 3000; then
    start_frontend
    echo "Waiting for frontend to be ready..."
    sleep 8
    echo "✓ Frontend server started on http://localhost:3000"
else
    echo "Using existing frontend server on port 3000"
fi

echo ""
echo "Application started successfully!"
echo ""
echo "🔗 Access the application:"
echo "   Frontend (UI): http://localhost:3000"
echo "   Backend (API): http://localhost:8000/recommend?item_id=1&limit=3"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📋 Usage:"
echo "   - Enter a Magento item ID (1-15) in the frontend UI"
echo "   - Get AI-powered product recommendations"
echo ""
echo "🛠️  Management commands:"
echo "   - To stop: pkill -f 'uvicorn main:app' && pkill -f 'npm run dev'"
echo "   - Backend logs: tail -f $PROJECT_ROOT/backend/backend.log"
echo "   - Frontend logs: tail -f $PROJECT_ROOT/frontend/frontend.log"
echo ""