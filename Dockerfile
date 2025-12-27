
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user for security (required by some HF spaces)
RUN useradd -m -u 1000 user

# Create data directories with correct permissions
RUN mkdir -p data/pdfs vector_store && \
    chown -R user:user /app/data /app/vector_store

USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt

# Install torch CPU version first to avoid downloading huge CUDA binaries
# Then install other requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

CMD ["python", "app.py"]
