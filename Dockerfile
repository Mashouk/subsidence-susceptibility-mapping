FROM python:3.12-slim
ENV GRADIO_ANALYTICS_ENABLED=False PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY prepare.py metrics.py feedback.py app.py ./
COPY data/subsidence_dataset_processed.csv data/label_build_log.json ./data/
# train the selected Random Forest once at build time so the container starts fast
RUN python prepare.py
RUN mkdir -p feedback
EXPOSE 7860 8000
CMD ["python", "app.py"]
