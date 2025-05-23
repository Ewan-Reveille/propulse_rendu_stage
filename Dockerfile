FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=fr_FR.UTF-8 \
    LC_ALL=fr_FR.UTF-8

# Install system dependencies with locales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    locales \
    && sed -i '/fr_FR.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen fr_FR.UTF-8 \
    && update-locale LANG=fr_FR.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

    
# Download language resources
RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet punkt_tab && \
python -m spacy download fr_core_news_md

# Copy application
COPY . .
RUN cp /app/nam_dict.txt $(python -c "import gender_guesser; import os; print(os.path.join(os.path.dirname(gender_guesser.__file__), 'data', 'nam_dict.txt'))")

# Expose port
EXPOSE 5000

# Launch application
CMD ["gunicorn", "-w", "4", "--timeout", "360", "-b", "0.0.0.0:5000", "app:app"]