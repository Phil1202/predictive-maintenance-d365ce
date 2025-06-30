#!/bin/bash
echo "Starting FastAPI app with gunicorn..."
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
