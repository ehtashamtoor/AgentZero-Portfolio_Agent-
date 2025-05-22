FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set environment variables for Uvicorn to bind properly
ENV PORT=8000
ENV HOST=0.0.0.0

# System dependencies (faiss, psycopg, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your code to the container
COPY . .

RUN pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "AgentZero:app", "--host", "0.0.0.0", "--port", "8000"]
