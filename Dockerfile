# Use official Python image as a parent image
FROM python:3.12.3

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    && pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files into the container
COPY . .
RUN curl -o models/random_forest_model.pk1 "https://drive.google.com/file/d/1TajXmXgHp3jzIqKRhCDlxy9wUbli9uBf/view?usp=sharing"
#kkk
# Run FastAPI on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Expose the port used by the FastAPI app
EXPOSE 8000