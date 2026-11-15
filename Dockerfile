FROM python:3.12-slim
ENV GRADIO_ANALYTICS_ENABLED=False PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY prepare.py metrics.py feedback.py app.py ./
COPY data/subsidence_dataset_processed.csv data/label_build_log.json ./data/
COPY data/v2/subsidence_dataset_processed.csv data/v2/label_build_log.json ./data/v2/
# which label to serve: v2 (default, reset-aware) or v1 (as built for Milestone 4):  docker build --build-arg DATA_PATH=data/subsidence_dataset_processed.csv .
ARG DATA_PATH=data/v2/subsidence_dataset_processed.csv
ENV SUBSIDENCE_DATA=$DATA_PATH
# train the selected Random Forest once at build time so the container starts fast
RUN python prepare.py
RUN mkdir -p feedback
EXPOSE 7860 8000
CMD ["python", "app.py"]
