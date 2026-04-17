FROM python:3.10-slim

WORKDIR /app

# Install dependencies from your existing requirements.txt + flask + gunicorn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask gunicorn

# Copy project files
COPY src/         ./src/
COPY models/      ./models/
COPY params.yaml  ./params.yaml
COPY app.py       ./app.py

EXPOSE 5001

# gunicorn: 2 workers, binds to port 5001, points to app object in app.py
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "app:app"]
