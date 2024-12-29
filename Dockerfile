# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy requirements.txt for dependency installation
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the application code
COPY src ./src

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "src/streamlit-app.py"]