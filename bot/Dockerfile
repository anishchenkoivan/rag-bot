FROM pytorch/pytorch:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/bot

# Set working directory
WORKDIR /bot

# Install system dependencies (adjust if your bot needs more)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your bot code
COPY . .

# Command to run your bot (adjust main file name if needed)
CMD ["python", "main.py"]
