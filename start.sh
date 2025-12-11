# Production startup script for AgriLink context based
uvicorn main:app --host 0.0.0.0 --port $PORT