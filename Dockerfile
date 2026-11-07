# Recommendation API: FastAPI + FAISS + the user tower on CPU, with the serving bundle baked into the image.
#   docker build --build-arg BUNDLE=models/serve/full-main-6ep -t twotower-api .
#   az acr build -r <registry> -t twotower-api:<tag> --build-arg BUNDLE=models/serve/full-main-6ep .
FROM python:3.12-slim

WORKDIR /app
# One thread per process for torch and FAISS: each request is tiny, so threads only add contention.
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BUNDLE_DIR=/app/bundle

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY web/ web/
# Fail the build, not the deployment, if the serving import chain needs a package the image lacks.
RUN python -c "import src.serve"
ARG BUNDLE
COPY ${BUNDLE}/ bundle/

EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
