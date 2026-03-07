FROM python:3.11

WORKDIR /app

# Copy dependency file first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY score.py .
COPY CSVC_best_model.pkl .

# Expose Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]