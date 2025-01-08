FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY src ./src
COPY pyproject.toml .
COPY uv.lock . 

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN uv sync

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/streamlit-app.py"]