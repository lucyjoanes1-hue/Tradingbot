# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure folders exist
RUN mkdir -p models logs

# Run the bot
CMD ["bash", "start.sh"]
