# Use Python 3.12.6 as the base image
FROM python:3.12.6-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Run Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]