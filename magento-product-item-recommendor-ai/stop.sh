#!/bin/bash

# Magento Product Recommender AI - Stop Script
# This script stops all running instances of the application

echo "Stopping Magento Product Recommender AI..."
echo "=========================================="

# Kill backend and frontend processes
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

# Also try to kill any node processes that might be related to the dev server
pkill -f "node.*next" 2>/dev/null || true

echo "All processes stopped."
echo ""
echo "✓ Backend server (uvicorn) stopped"
echo "✓ Frontend server (npm run dev) stopped"
echo ""
echo "To start the application again, run:"
echo "  ./setup.sh"