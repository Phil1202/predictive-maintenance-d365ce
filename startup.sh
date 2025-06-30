#!/bin/bash
echo "Starting FastAPI app with gunicorn..."

# In das Verzeichnis wechseln, das die main.py enthält
cd api

# Gunicorn starten mit Pfad zu main:app
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
