FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Match the serialized pipeline's scikit-learn version. CPU inference uses NumPy and SciPy.
RUN pip install --no-cache-dir fastapi==0.115.12 pydantic==2.10.6 uvicorn==0.34.0 \
        joblib==1.4.2 numpy==2.2.6 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.6.1 \
    && pip install --no-cache-dir --no-deps xgboost==2.1.4

COPY src/api.py src/api.py
COPY logs/final_run.json logs/final_run.json
COPY models/final_xgb.joblib models/final_xgb.joblib

USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
