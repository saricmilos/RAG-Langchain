# Use a slim Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /code

# 1. Install dependencies first (better for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the application code
COPY app/ ./app/

# 3. WE REMOVE THE LOCAL DATA COPY
# Pinecone will be accessed via os.getenv("PINECONE_API_KEY")

# Expose port for FastAPI
EXPOSE 8080

# Run main.py as a module so 'app' imports work
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
