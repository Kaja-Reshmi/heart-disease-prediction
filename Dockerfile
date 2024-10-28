# Use a supported Python runtime
FROM python:3.9-slim  # or python:3.7-slim

# Rest of the Dockerfile remains unchanged
WORKDIR /app

COPY . /app
COPY lung_cancer_project.zip /app/

RUN apt-get update && apt-get install -y unzip && \
    unzip C:\Users\Achuth Kaja\heart_disease_prediction
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]
