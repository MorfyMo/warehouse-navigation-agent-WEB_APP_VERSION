FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY Web_App/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RL modules from parent directory
COPY Env ./Env
COPY DQN_test ./DQN_test
COPY PPO_test ./PPO_test
COPY main_module.py ./main_module.py

# Copy the application code
COPY Web_App/ .

# Create logs directory
RUN mkdir -p logs/Test_DQN/Environment logs/PPO-program/Environment

# Expose the port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV MPLBACKEND=Agg
ENV TF_CPP_MIN_LOG_LEVEL=2

# Run the application
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080", "--ws", "websockets", "--ws-ping-interval", "15.0", "--ws-ping-timeout", "10.0"]
