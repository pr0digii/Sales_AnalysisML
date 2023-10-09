# Use the FastAPI Uvicorn base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY ./app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application
COPY ./app/main.py .

# Copy the model artifacts both predictive and forecastive
COPY ./models/predictive/prediction_artefacts.joblib/ ./models/predictive/



COPY ./models/forecastive/holt_winters_model.joblib ./models/forecastive/

# Expose the port the app runs on
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
