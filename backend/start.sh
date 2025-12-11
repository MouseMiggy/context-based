# Production startup script for AgriLink context based
gunicorn app:app --bind 0.0.0.0:$PORT