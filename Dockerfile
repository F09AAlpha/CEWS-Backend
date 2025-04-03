# Use Python 3.12.6 as the base image
FROM python:3.12.6-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install PostgreSQL dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        postgresql-client \
        libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install wkhtmltopdf 
RUN apt-get update && apt-get install -y wkhtmltopdf

# Expose the correct port
EXPOSE 3000

# Copy project
COPY . /app/

# Run Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:3000"]