#!/bin/bash
echo "Starting Production RAG System..."
docker-compose up --build -d
echo "Services started."
echo "API: http://localhost:8000/docs"
echo "Qdrant Dashboard: http://localhost:6333/dashboard"
