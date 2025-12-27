
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user for security (required by some HF spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Create data directories with correct permissions
RUN mkdir -p data/pdfs vector_store

CMD ["python", "app.py"]
