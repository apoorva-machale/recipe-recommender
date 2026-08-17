FROM python:3.11-slim

WORKDIR /app

# psycopg2-binary ships its own libpq, but build tools are still needed for a
# couple of source-only deps pulled in transitively by sentence-transformers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PGHOST=db \
    PGPORT=5432 \
    PGDATABASE=recipes \
    PGUSER=recipes \
    PGPASSWORD=recipes \
    PYTHONUNBUFFERED=1

CMD ["python3", "run.py"]
