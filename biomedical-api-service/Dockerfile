# Use the official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# 🚀 THE MAGIC LINE: Force CPU-only AI libraries, drop caches, and increase timeout
RUN pip install --default-timeout=1000 --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the enterprise server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]